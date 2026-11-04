# fit_vignetting and score_seams on synthetic frames that follow the model exactly: I_k(p) = L(R_k·f(p))·V(u(p)) at every pixel
# centre (no interpolation anywhere), through a Kannala–Brandt fisheye, frames rotated in yaw AND pitch. The true V is recovered
# (a radial V about a centre, and an off-centre, asymmetric 2-D V by the tensor space), the host and the device agree, CG and
# Cholesky agree, and the seam score of the true V is far below that of no correction.

@testsetup module VignettingSequence
using LinearAlgebra: I, normalize
using StaticArrays: SVector, SMatrix, SA
using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, px, backproject, exp_so3, Circle, Point2
export vignetting_sequence

# the scene: a smooth log radiance on the sphere (wavelengths of about 30°; Cartesian, so no pole is singular)
scene(d) = 0.05 * exp(0.6 * sin(3d[1] + 2d[2]) * cos(2d[3]) + 0.3 * sin(9d[1] - 7d[3] + 5d[2]))

# `gains`: the frames' exposure gains g_k (I_k = g_k·L·V; all 1 by default)
function vignetting_sequence(logV; gains = ones(6))
    W = H = 800; s = 2; Wf, Hf = W ÷ s, H ÷ s
    model = CameraModel(260.0px, SA[400.0px, 400.0px]; lens = KannalaBrandt(-0.02))
    circle = Circle(Point2(400.0, 400.0), 390.0)
    angles = [(0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (15.0, 25.0), (45.0, 25.0), (30.0, -20.0)]   # (yaw, pitch) degrees
    R = [SMatrix{3, 3}(exp_so3(SVector(0.0, deg2rad(y), 0.0))) * SMatrix{3, 3}(exp_so3(SVector(deg2rad(p), 0.0, 0.0))) for (y, p) in angles]
    frames = map(enumerate(R)) do (k, Rk)
        img = zeros(Float32, Hf, Wf, 3)
        for j in 1:Wf, i in 1:Hf
            u = SVector((j - 0.5) * s, (i - 0.5) * s)
            hypot(u[1] - 400.0, u[2] - 400.0) < 390.0 || continue
            f = backproject(model, u)
            img[i, j, :] .= Float32(gains[k] * scene(Rk * f) * exp(logV(u)))
        end
        img
    end
    return (; frames, model, circle, raw_scale = s, R, saturated = [falses(Hf, Wf) for _ in R])
end
end

@testitem "fit_vignetting: a radial and an off-centre 2-D V recovered, host and device, CG and Cholesky" setup = [VignettingSequence] begin
    using Test, Statistics, StaticArrays
    using PanoramaStitching: fit_vignetting, RadialVignetting, TensorVignetting, log_vignetting, score_seams, panorama_view,
        ConjugateGradient, SparseCholesky
    import VisualGeometryToolkitCore as VgtkCore
    # the domain's pixel centres (camera pixels) and the error of a fit against the truth, both in the gauge mean log V = 0
    function error_vs(v, logV, q; margin = 8.0)
        us = [SVector((j - 0.5) * 2, (i - 0.5) * 2) for j in 1:400, i in 1:400]
        us = filter(u -> hypot(u[1] - 400, u[2] - 400) < 390 - margin, vec(us))
        a = [log_vignetting(v, u) for u in us]; b = [logV(u) for u in us]
        a .-= mean(a); b .-= mean(b)
        r = exp.(a .- b) .- 1
        return maximum(abs, r), sqrt(mean(abs2, r))
    end
    kw = (; circle = VgtkCore.Circle(VgtkCore.Point2(400.0, 400.0), 390.0), raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0,
          plane = 2, plane_offsets = nothing, chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01,
          vignetting_penalty_weight = 1.0e-6, frame_gains = false)
    cgs = ConjugateGradient(; tolerance = 1.0e-10, max_iter = 20000)

    # a radial V about a centre 30 px off the image's centre
    c = SVector(430.0, 390.0)
    radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2 - 0.3 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^4
    q = vignetting_sequence(radial)
    space = RadialVignetting(c, 420.0, 12)                     # the domain reaches ρ ≈ 413 px from c
    cg = fit_vignetting(space, q.model, q.frames, q.R; saturated = q.saturated, solver = cgs, kw...)
    ch = fit_vignetting(space, q.model, q.frames, q.R; saturated = q.saturated, solver = SparseCholesky(), kw...)
    e_cg = error_vs(VgtkCore.estimate(cg), radial, q); e_ch = error_vs(VgtkCore.estimate(ch), radial, q)
    @info "fit_vignetting radial: max/rms error CG $(e_cg), Cholesky $(e_ch); $(cg.iterations) CG iterations, rms $(cg.rms)"
    @test VgtkCore.retcode(cg) === :converged && VgtkCore.retcode(ch) === :solved
    @test e_cg[1] < 2.0e-3 && e_ch[1] < 2.0e-3
    @test maximum(abs.(VgtkCore.estimate(cg).coefficients .- VgtkCore.estimate(ch).coefficients)) < 1.0e-5
    @test cg.rms < 0.02

    # an off-centre, asymmetric 2-D V (a quartic about a displaced point and a tilt), by the tensor space
    d = SVector(460.0, 350.0)
    tilted(u) = -1.4 * (hypot(u[1] - d[1], u[2] - d[2]) / 400)^4 - 0.6 * (hypot(u[1] - d[1], u[2] - d[2]) / 400)^2 +
                0.05 * (u[1] - 400) / 400 - 0.04 * (u[2] - 400) / 400
    q2 = vignetting_sequence(tilted)
    t = fit_vignetting(TensorVignetting((10.0, 10.0), (780.0, 780.0), (16, 16)), q2.model, q2.frames, q2.R;
                       saturated = q2.saturated, solver = SparseCholesky(), kw...)
    e2 = error_vs(VgtkCore.estimate(t), tilted, q2)
    @info "fit_vignetting tensor: max/rms error $(e2)"
    @test e2[1] < 1.0e-2 && e2[2] < 1.0e-3                     # measured 0.65 % (at the rim) and 0.03 %

    # the seam score: the true V leaves (almost) nothing, no correction leaves the vignetting's steps
    view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.5)
    sk = (; saturated = q.saturated, circle = kw.circle, raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2,
          plane_offsets = nothing, chunk = 20_000)
    fitted = score_seams(q.model, q.frames, q.R, view; vignetting = VgtkCore.estimate(cg), sk...)
    none = score_seams(q.model, q.frames, q.R, view; vignetting = (u, f) -> 1.0, sk...)
    @info "score_seams: fitted V rms $(fitted.rms) mad $(fitted.mad), none rms $(none.rms) mad $(none.mad), $(fitted.n) pixels"
    @test fitted.n > 10_000 && fitted.n == none.n
    @test fitted.mad < 0.1 * none.mad && fitted.rms < 0.1 * none.rms
end

@testitem "fit_vignetting, score_seams: the device agrees with the host" setup = [VignettingSequence] begin
    using Test, StaticArrays, CUDA
    using PanoramaStitching: fit_vignetting, RadialVignetting, score_seams, panorama_view, ConjugateGradient
    import VisualGeometryToolkitCore as VgtkCore
    if !CUDA.functional()
        @test_skip "no CUDA device"
    else
        c = SVector(430.0, 390.0)
        radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2
        q = vignetting_sequence(radial)
        kw = (; circle = VgtkCore.Circle(VgtkCore.Point2(400.0, 400.0), 390.0), raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0,
              plane = 2, plane_offsets = nothing, chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01,
              vignetting_penalty_weight = 1.0e-6, frame_gains = true, solver = ConjugateGradient(; tolerance = 1.0e-10, max_iter = 20000))
        space = RadialVignetting(c, 420.0, 12)
        h = fit_vignetting(space, q.model, q.frames, q.R; saturated = q.saturated, kw...)
        df, ds = CuArray.(q.frames), CuArray.(q.saturated)
        d = fit_vignetting(space, q.model, df, q.R; saturated = ds, kw...)
        @test VgtkCore.retcode(d) === :converged
        # the atomic sums run in another order on each backend (and from run to run): log V agrees to the solve's precision
        @test maximum(abs.(VgtkCore.estimate(d).coefficients .- VgtkCore.estimate(h).coefficients)) < 1.0e-6   # measured 2e-10 … 1.5e-7
        @test abs(d.rms - h.rms) < 1.0e-8
        view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.5)
        sk = (; circle = kw.circle, raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2, plane_offsets = nothing,
              chunk = 20_000, vignetting = VgtkCore.estimate(h))
        sh = score_seams(q.model, q.frames, q.R, view; saturated = q.saturated, sk...)
        sd = score_seams(q.model, df, q.R, view; saturated = ds, sk...)
        @test sh.n == sd.n && abs(sh.rms - sd.rms) < 1.0e-9 && abs(sh.mad - sd.mad) < 1.0e-9
        @test maximum(abs.(d.metadata.gains .- h.metadata.gains)) < 1.0e-8 && d.metadata.gains[1] == 1.0
    end
end

# The frames' exposure gains: frames of the same scene and V at different exposures, I_k = g_k·L·V. With frame_gains the fit
# recovers the gains (gauge g₁ = 1, exactly) and V; without them the exposure differences go into V and the scene, and V is
# wrong. score_seams and render_panorama divide the gains out.
@testitem "fit_vignetting: the frames' exposure gains recovered (gauge g₁ = 1); score_seams and render_panorama divide them out" setup = [VignettingSequence] begin
    using Test, Statistics, StaticArrays
    using PanoramaStitching: fit_vignetting, RadialVignetting, log_vignetting, score_seams, render_panorama, panorama_view,
        ConjugateGradient, SparseCholesky
    import VisualGeometryToolkitCore as VgtkCore
    c = SVector(430.0, 390.0)
    radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2 - 0.3 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^4
    g = [1.0, 1.12, 0.9, 1.2, 0.95, 1.07]
    q = vignetting_sequence(radial; gains = g)
    circle = VgtkCore.Circle(VgtkCore.Point2(400.0, 400.0), 390.0)
    kw = (; saturated = q.saturated, circle, raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2, plane_offsets = nothing,
          chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01, vignetting_penalty_weight = 1.0e-6)
    space = RadialVignetting(c, 420.0, 12)
    cgs = ConjugateGradient(; tolerance = 1.0e-10, max_iter = 20000)
    function v_error(v)
        us = filter(u -> hypot(u[1] - 400, u[2] - 400) < 382, vec([SVector((j - 0.5) * 2, (i - 0.5) * 2) for j in 1:400, i in 1:400]))
        a = [log_vignetting(v, u) for u in us]; b = [radial(u) for u in us]
        return maximum(abs, exp.((a .- mean(a)) .- (b .- mean(b))) .- 1)
    end

    with = fit_vignetting(space, q.model, q.frames, q.R; frame_gains = true, solver = cgs, kw...)
    chol = fit_vignetting(space, q.model, q.frames, q.R; frame_gains = true, solver = SparseCholesky(), kw...)
    without = fit_vignetting(space, q.model, q.frames, q.R; frame_gains = false, solver = cgs, kw...)
    gw = with.metadata.gains
    @info "frame gains: recovered $(round.(gw; digits = 5)) (true $g); V error with $(v_error(VgtkCore.estimate(with))), without $(v_error(VgtkCore.estimate(without)))"
    @test VgtkCore.retcode(with) === :converged
    @test gw[1] === 1.0 && length(gw) == length(g)                    # the gauge: the first frame's gain is 1 exactly
    @test maximum(abs.(gw ./ g .- 1)) < 1.0e-3
    @test maximum(abs.(chol.metadata.gains .- gw)) < 1.0e-5           # CG and Cholesky solve the same system
    @test v_error(VgtkCore.estimate(with)) < 2.0e-3
    @test v_error(VgtkCore.estimate(without)) > 5 * v_error(VgtkCore.estimate(with))
    @test all(==(1.0), without.metadata.gains)
    @test with.rms < 0.5 * without.rms

    # the seams: the fitted V with its gains leaves (almost) nothing; the same V with the frames taken at one exposure does not
    view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.5)
    sk = (; saturated = q.saturated, circle, raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2, plane_offsets = nothing,
          chunk = 20_000, vignetting = VgtkCore.estimate(with))
    sg = score_seams(q.model, q.frames, q.R, view; gains = gw, sk...)
    s1 = score_seams(q.model, q.frames, q.R, view; sk...)
    @info "score_seams with the gains: rms $(sg.rms) mad $(sg.mad); without: rms $(s1.rms) mad $(s1.mad)"
    @test sg.rms < 0.1 * s1.rms                                       # (the MAD can stay small: a gain step is constant per frame region)
    @test_throws DimensionMismatch score_seams(q.model, q.frames, q.R, view; gains = gw[1:3], sk...)

    # the render: dividing by g_k·V makes every frame show the same scene value
    rk = (; vignetting = VgtkCore.estimate(with), circle, raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0)
    a = render_panorama(q.model, q.frames, q.R, view; gains = gw, rk...)
    b = render_panorama(q.model, q.frames, q.R, view; rk...)
    o = a.owner; m = o .> 0
    @test all(i -> a.image[i, 2] * gw[o[i]] ≈ b.image[i, 2], findall(m)[1:97:end])
end
