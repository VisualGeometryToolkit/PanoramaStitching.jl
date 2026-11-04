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

function vignetting_sequence(logV)
    W = H = 800; s = 2; Wf, Hf = W ÷ s, H ÷ s
    model = CameraModel(260.0px, SA[400.0px, 400.0px]; lens = KannalaBrandt(-0.02))
    circle = Circle(Point2(400.0, 400.0), 390.0)
    angles = [(0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (15.0, 25.0), (45.0, 25.0), (30.0, -20.0)]   # (yaw, pitch) degrees
    R = [SMatrix{3, 3}(exp_so3(SVector(0.0, deg2rad(y), 0.0))) * SMatrix{3, 3}(exp_so3(SVector(deg2rad(p), 0.0, 0.0))) for (y, p) in angles]
    frames = map(R) do Rk
        img = zeros(Float32, Hf, Wf, 3)
        for j in 1:Wf, i in 1:Hf
            u = SVector((j - 0.5) * s, (i - 0.5) * s)
            hypot(u[1] - 400.0, u[2] - 400.0) < 390.0 || continue
            f = backproject(model, u)
            img[i, j, :] .= Float32(scene(Rk * f) * exp(logV(u)))
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
          vignetting_penalty_weight = 1.0e-6)
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
              vignetting_penalty_weight = 1.0e-6, solver = ConjugateGradient(; tolerance = 1.0e-10, max_iter = 20000))
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
    end
end
