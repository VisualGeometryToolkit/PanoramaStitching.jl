# fit_vignetting and score_seams on synthetic frames that follow the model exactly: I_k(p) = g_k·L(R_k·f(p))·V(u(p)) at every
# pixel centre (no interpolation anywhere), through a Kannala–Brandt fisheye, frames rotated in yaw AND pitch. The true V is
# recovered in VisualGeometryToolkitCore's model (a radial V about a centre, and an off-centre, asymmetric 2-D V by a tensor
# surface), the host and the device agree, CG and Cholesky agree, the gains are recovered with V free or held fixed (gain
# compensation, the same solve), and the seam score of the true V is far below that of no correction.

@testsetup module VignettingSequence
using LinearAlgebra: I, normalize
using StaticArrays: SVector, SMatrix, SA
using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, px, backproject, exp_so3, Circle, Point2
export vignetting_sequence

# the scene: a smooth log radiance on the sphere (wavelengths of about 30°; Cartesian, so no pole is singular)
scene(d) = 0.05 * exp(0.6 * sin(3d[1] + 2d[2]) * cos(2d[3]) + 0.3 * sin(9d[1] - 7d[3] + 5d[2]))

# `gains`: the frames' exposure gains, one per frame g_k or one per frame and colour plane g_kc (a K × 3 matrix): I_kc = g_kc·t_c·L·V,
# t = (0.8, 1, 1.3) the scene's colour (all 1 by default)
function vignetting_sequence(logV; gains = ones(6))
    G = gains isa AbstractMatrix ? gains : repeat(gains, 1, 3); tint = (0.8, 1.0, 1.3)
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
            for c in 1:3
                img[i, j, c] = Float32(G[k, c] * tint[c] * scene(Rk * f) * exp(logV(u)))
            end
        end
        img
    end
    return (; frames, model, circle, raw_scale = s, R, saturated = [falses(Hf, Wf) for _ in R])
end
end

@testitem "fit_vignetting: a radial and an off-centre 2-D V recovered, host and device, CG and Cholesky" setup = [VignettingSequence] begin
    using Test, Statistics, StaticArrays
    using LinearAlgebra: I
    using PanoramaStitching: PanoramaStitching, fit_vignetting, score_seams, panorama_view, ConjugateGradient, SparseCholesky
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: RadialVignetting, TensorVignetting, LogVignetting, pole_regular_basis, n_control_points,
        control_points, with_control_points
    logv(s) = LogVignetting(with_control_points(s, zeros(n_control_points(s))))      # log V ≡ 0: the start
    # the domain's pixel centres (camera pixels) and the error of a fit against the truth, both in the gauge mean log V = 0
    function error_vs(v, logV, q; margin = 8.0)
        us = [SVector((j - 0.5) * 2, (i - 0.5) * 2) for j in 1:400, i in 1:400]
        us = filter(u -> hypot(u[1] - 400, u[2] - 400) < 390 - margin, vec(us))
        a = [log(v(u)) for u in us]; b = [logV(u) for u in us]
        a .-= mean(a); b .-= mean(b)
        r = exp.(a .- b) .- 1
        return maximum(abs, r), sqrt(mean(abs2, r))
    end
    kw = (; circle = VgtkCore.Circle(VgtkCore.Point2(400.0, 400.0), 390.0), raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0,
          plane = 2, plane_offsets = nothing, chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01,
          vignetting_penalty_weight = 1.0e-6, gain_correction = nothing)
    cgs = ConjugateGradient(; tolerance = 1.0e-10, max_iter = 20000)

    # a radial V about a centre 30 px off the image's centre, smooth at the centre (the pole-regular subspace)
    c = SVector(430.0, 390.0)
    radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2 - 0.3 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^4
    q = vignetting_sequence(radial)
    v0 = logv(RadialVignetting(12, c, 420.0))                # the domain reaches ρ ≈ 413 px from c
    @test_throws ArgumentError fit_vignetting(VgtkCore.log_spline(v0), q.model, q.frames, q.R; free_directions = pole_regular_basis(v0),
                                              saturated = q.saturated, solver = cgs, kw...)      # a V-form spline is only held fixed
    Z = pole_regular_basis(v0)
    cg = fit_vignetting(v0, q.model, q.frames, q.R; free_directions = Z, saturated = q.saturated, solver = cgs, kw...)
    ch = fit_vignetting(v0, q.model, q.frames, q.R; free_directions = Z, saturated = q.saturated, solver = SparseCholesky(), kw...)
    e_cg = error_vs(VgtkCore.estimate(cg), radial, q); e_ch = error_vs(VgtkCore.estimate(ch), radial, q)
    @info "fit_vignetting radial: max/rms error CG $(e_cg), Cholesky $(e_ch); $(cg.iterations) CG iterations, rms $(cg.rms)"
    @test VgtkCore.retcode(cg) === :converged && VgtkCore.retcode(ch) === :solved
    @test cg.metadata.log_change == 0 && cg.metadata.weights === PanoramaStitching.IntensityWeights()
    @test e_cg[1] < 2.0e-3 && e_ch[1] < 2.0e-3
    @test maximum(abs.(control_points(VgtkCore.estimate(cg)) .- control_points(VgtkCore.estimate(ch)))) < 1.0e-5
    @test cg.rms < 0.02
    P = control_points(VgtkCore.estimate(cg)); @test P[1] == P[2]           # V′(0) = 0 at the centre

    # an off-centre, asymmetric 2-D V (a quartic about a displaced point and a tilt), by a tensor surface
    d = SVector(460.0, 350.0)
    tilted(u) = -1.4 * (hypot(u[1] - d[1], u[2] - d[2]) / 400)^4 - 0.6 * (hypot(u[1] - d[1], u[2] - d[2]) / 400)^2 +
                0.05 * (u[1] - 400) / 400 - 0.04 * (u[2] - 400) / 400
    q2 = vignetting_sequence(tilted)
    t0 = logv(TensorVignetting((16, 16), VgtkCore.Rect(10.0, 10.0, 780.0, 780.0); degree = 3))
    t = fit_vignetting(t0, q2.model, q2.frames, q2.R; free_directions = Matrix(1.0I, n_control_points(t0), n_control_points(t0)),
                       saturated = q2.saturated, solver = SparseCholesky(), kw...)
    e2 = error_vs(VgtkCore.estimate(t), tilted, q2)
    @info "fit_vignetting tensor: max/rms error $(e2)"
    @test e2[1] < 1.0e-2 && e2[2] < 1.0e-3

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
    using PanoramaStitching: fit_vignetting, score_seams, render_panorama, panorama_view, ConjugateGradient
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: RadialVignetting, LogVignetting, pole_regular_basis, control_points, with_control_points
    using PanoramaStitching: SampsonWeights, ChannelGains
    import Adapt
    # a calibration's V as a table over the radius (a host Vector inside: it needs its Adapt rule to reach a device)
    struct TableV{A}
        t::A
        c::SVector{2, Float64}
        r::Float64
    end
    Adapt.@adapt_structure TableV
    (v::TableV)(u, f) = (x = clamp(hypot(u[1] - v.c[1], u[2] - v.c[2]) / v.r, 0.0, 1.0) * (length(v.t) - 1);
                         i = min(unsafe_trunc(Int, x), length(v.t) - 2); v.t[i + 1] + (x - i) * (v.t[i + 2] - v.t[i + 1]))
    if !CUDA.functional()
        @test_skip "no CUDA device"
    else
        c = SVector(430.0, 390.0)
        radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2
        q = vignetting_sequence(radial)
        v0 = LogVignetting(with_control_points(RadialVignetting(12, c, 420.0), zeros(12)))
        kw = (; circle = VgtkCore.Circle(VgtkCore.Point2(400.0, 400.0), 390.0), raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0,
              plane = 2, plane_offsets = nothing, chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01,
              vignetting_penalty_weight = 1.0e-6, gain_correction = ChannelGains(), free_directions = pole_regular_basis(v0),
              solver = ConjugateGradient(; tolerance = 1.0e-10, max_iter = 20000))
        h = fit_vignetting(v0, q.model, q.frames, q.R; saturated = q.saturated, kw...)
        df, ds = CuArray.(q.frames), CuArray.(q.saturated)
        d = fit_vignetting(v0, q.model, df, q.R; saturated = ds, kw...)
        @test VgtkCore.retcode(d) === :converged
        # the atomic sums run in another order on each backend (and from run to run): V agrees to the solve's precision
        @test maximum(abs.(control_points(VgtkCore.estimate(d)) .- control_points(VgtkCore.estimate(h)))) < 1.0e-6
        @test abs(d.rms - h.rms) < 1.0e-8
        # the Sampson weights on the device (the observation's constraint, ForwardDiff and Core's whitening in the kernels)
        sw = SampsonWeights(; point_scale = 1.0, iterations = 2)
        hs = fit_vignetting(v0, q.model, q.frames, q.R; saturated = q.saturated, weights = sw, kw...)
        dsf = fit_vignetting(v0, q.model, df, q.R; saturated = ds, weights = sw, kw...)
        @test maximum(abs.(control_points(VgtkCore.estimate(dsf)) .- control_points(VgtkCore.estimate(hs)))) < 1.0e-6
        @test maximum(abs.(dsf.metadata.gains .- hs.metadata.gains)) < 1.0e-8 && abs(dsf.rms - hs.rms) < 1.0e-6
        view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.5)
        sk = (; circle = kw.circle, raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2, plane_offsets = nothing,
              chunk = 20_000, vignetting = VgtkCore.estimate(h))
        sh = score_seams(q.model, q.frames, q.R, view; saturated = q.saturated, sk...)
        sd = score_seams(q.model, df, q.R, view; saturated = ds, sk...)
        @test sh.n == sd.n && abs(sh.rms - sd.rms) < 1.0e-9 && abs(sh.mad - sd.mad) < 1.0e-9
        @test maximum(abs.(d.metadata.gains .- h.metadata.gains)) < 1.0e-8 && all(==(1.0), d.metadata.gains[1, :])
        # a V held fixed that holds HOST arrays (a calibration's table, with its Adapt rule) reaches the device in the fit, the
        # render and the seams (the functors are moved by OnBackend)
        tv = TableV(collect(range(1.0, 0.6; length = 64)), c, 420.0)
        hf = fit_vignetting(tv, q.model, q.frames, q.R; saturated = q.saturated, kw..., free_directions = zeros(0, 0))
        dfit = fit_vignetting(tv, q.model, df, q.R; saturated = ds, kw..., free_directions = zeros(0, 0))
        @test maximum(abs.(dfit.metadata.gains .- hf.metadata.gains)) < 1.0e-8
        rk = (; vignetting = tv, circle = kw.circle, raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0)
        a = render_panorama(q.model, q.frames, q.R, view; rk...); b = render_panorama(q.model, df, q.R, view; rk...)
        @test isequal(Array(b.owner), a.owner) && maximum(abs, filter(!isnan, vec(Array(b.image) .- a.image))) < 1.0e-5
        sdt = score_seams(q.model, df, q.R, view; saturated = ds, sk..., vignetting = tv)
        @test sdt.n == score_seams(q.model, q.frames, q.R, view; saturated = q.saturated, sk..., vignetting = tv).n
    end
end

# The frames' exposure gains: frames of the same scene and V at different exposures, I_k = g_k·L·V. With gain_correction the fit
# recovers the gains (gauge g₁ = 1, exactly) and V; without them the exposure differences go into V and the scene, and V is
# wrong. With V held fixed (no free direction: gain compensation under a calibration's V) the same solve recovers the gains
# alone. score_seams and render_panorama divide the gains out.
@testitem "fit_vignetting: the frames' exposure gains recovered (gauge g₁ = 1), V free or held fixed; score_seams and render_panorama divide them out" setup = [VignettingSequence] begin
    using Test, Statistics, StaticArrays
    using PanoramaStitching: PanoramaStitching, fit_vignetting, score_seams, render_panorama, panorama_view, ConjugateGradient,
        SparseCholesky, SampsonWeights, IntensityWeights, FrameGains
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: RadialVignetting, LogVignetting, pole_regular_basis, n_control_points, with_control_points
    c = SVector(430.0, 390.0)
    radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2 - 0.3 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^4
    g = [1.0, 1.12, 0.9, 1.2, 0.95, 1.07]
    q = vignetting_sequence(radial; gains = g)
    circle = VgtkCore.Circle(VgtkCore.Point2(400.0, 400.0), 390.0)
    v0 = LogVignetting(with_control_points(RadialVignetting(12, c, 420.0), zeros(12)))
    kw = (; saturated = q.saturated, circle, raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2, plane_offsets = nothing,
          chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01, vignetting_penalty_weight = 1.0e-6)
    cgs = ConjugateGradient(; tolerance = 1.0e-10, max_iter = 20000)
    function v_error(v)
        us = filter(u -> hypot(u[1] - 400, u[2] - 400) < 382, vec([SVector((j - 0.5) * 2, (i - 0.5) * 2) for j in 1:400, i in 1:400]))
        a = [log(v(u)) for u in us]; b = [radial(u) for u in us]
        return maximum(abs, exp.((a .- mean(a)) .- (b .- mean(b))) .- 1)
    end
    Z = pole_regular_basis(v0)
    with = fit_vignetting(v0, q.model, q.frames, q.R; free_directions = Z, gain_correction = FrameGains(), solver = cgs, kw...)
    chol = fit_vignetting(v0, q.model, q.frames, q.R; free_directions = Z, gain_correction = FrameGains(), solver = SparseCholesky(), kw...)
    without = fit_vignetting(v0, q.model, q.frames, q.R; free_directions = Z, gain_correction = nothing, solver = cgs, kw...)
    G = with.metadata.gains; gw = G[:, 2]
    @test size(G) == (6, 3) && G[:, 1] == G[:, 2] == G[:, 3]          # FrameGains: one per frame, every plane
    @info "frame gains: recovered $(round.(gw; digits = 5)) (true $g); V error with $(v_error(VgtkCore.estimate(with))), without $(v_error(VgtkCore.estimate(without)))"
    @test VgtkCore.retcode(with) === :converged
    @test gw[1] === 1.0 && length(gw) == length(g)                    # the gauge: the first frame's gain is 1 exactly
    @test maximum(abs.(gw ./ g .- 1)) < 1.0e-3
    @test maximum(abs.(chol.metadata.gains .- G)) < 1.0e-5            # CG and Cholesky solve the same system
    @test v_error(VgtkCore.estimate(with)) < 2.0e-3
    @test v_error(VgtkCore.estimate(without)) > 5 * v_error(VgtkCore.estimate(with))
    @test all(==(1.0), without.metadata.gains)
    @test with.rms < 0.5 * without.rms

    # gain compensation: V held fixed at the fitted one (no free direction), the gains alone, one linear solve
    fixed = fit_vignetting(VgtkCore.estimate(with), q.model, q.frames, q.R; free_directions = zeros(n_control_points(v0), 0),
                           gain_correction = FrameGains(), solver = SparseCholesky(), kw...)
    @test VgtkCore.estimate(fixed) === VgtkCore.estimate(with) && fixed.metadata.log_change == 0
    @test maximum(abs.(fixed.metadata.gains[:, 2] ./ g .- 1)) < 1.0e-3
    @test maximum(abs.(fixed.metadata.gains .- G)) < 1.0e-5

    # the Sampson weights (the observation's constraint whitened with the sensor point's noise): frames that follow the model
    # exactly give the same V and gains; the weights come from the I² solve's model, so V held fixed takes them too
    sw = SampsonWeights(; point_scale = 1.5, iterations = 2)
    sam = fit_vignetting(v0, q.model, q.frames, q.R; free_directions = Z, gain_correction = FrameGains(), solver = SparseCholesky(),
                         weights = sw, kw...)
    @test sam.metadata.weights === sw && sam.metadata.log_change > 0        # two Sampson iterations after the I² solve
    @test maximum(abs.(sam.metadata.gains[:, 2] ./ g .- 1)) < 1.0e-3 && v_error(VgtkCore.estimate(sam)) < 2.0e-3
    sf = fit_vignetting(VgtkCore.estimate(with), q.model, q.frames, q.R; free_directions = zeros(n_control_points(v0), 0),
                        gain_correction = FrameGains(), solver = SparseCholesky(), weights = SampsonWeights(; point_scale = 1.5), kw...)
    @test sf.metadata.weights.iterations == 1 && maximum(abs.(sf.metadata.gains[:, 2] ./ g .- 1)) < 1.0e-3
    @test_throws ArgumentError SampsonWeights()                                         # the point's noise is required
    @test_throws ArgumentError SampsonWeights(; point_scale = 1.0, point_covariance = [1.0 0; 0 1])
    @test_throws ArgumentError SampsonWeights(; point_scale = 1.0, iterations = 3)
    @test_throws ArgumentError SampsonWeights(; point_covariance = [1.0 2.0; 2.0 1.0])  # not positive definite

    # the observation's constraint (Core's protocol): G_x is the model's exact gradient in u, and the weight is Core's whitening
    m = with.metadata
    cL = replace(m.scene, NaN => 0.0)
    obs = PanoramaStitching._LogIntensityModel(VgtkCore.estimate(with), m.chart, cL, q.R[3], log(m.gains[3, 2]))
    for u in (SVector(300.0, 420.0), SVector(520.0, 260.0), SVector(410.0, 395.0))
        f, M = VgtkCore.backproject_with_jacobian(q.model, u)
        x = SVector(u[1], u[2], -2.0)
        gx, G_x, G_θ = VgtkCore.constraint_with_jacobians(obs, x; ray = f, ray_jacobian = M)
        gat(w) = VgtkCore.constraint(obs, SVector(w[1], w[2], -2.0); ray = VgtkCore.backproject(q.model, w),
                                     ray_jacobian = VgtkCore.backproject_with_jacobian(q.model, w)[2])
        h = 1.0e-4
        fd = SVector((gat(u + SVector(h, 0)) - gat(u - SVector(h, 0))) / 2h, (gat(u + SVector(0, h)) - gat(u - SVector(0, h))) / 2h)
        @test isapprox(SVector(G_x[1], G_x[2]), fd; rtol = 1.0e-5, atol = 1.0e-9) && G_x[3] == 1.0
        @test length(G_θ) == 16 + 4 + 1 && G_θ[end] == -1.0
        Σx = PanoramaStitching._measurement_covariance(SampsonWeights(; point_scale = 1.5))
        @test VgtkCore.whitening_factor(G_x, Σx)^2 ≈ 1 / (1.5^2 * (G_x[1]^2 + G_x[2]^2) + 0.01^2)
    end

    # the seams: the fitted V with its gains leaves (almost) nothing; the same V with the frames taken at one exposure does not
    view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.5)
    sk = (; saturated = q.saturated, circle, raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2, plane_offsets = nothing,
          chunk = 20_000, vignetting = VgtkCore.estimate(with))
    sg = score_seams(q.model, q.frames, q.R, view; gains = G, sk...)
    s1 = score_seams(q.model, q.frames, q.R, view; sk...)
    @info "score_seams with the gains: rms $(sg.rms) mad $(sg.mad); without: rms $(s1.rms) mad $(s1.mad)"
    @test sg.rms < 0.1 * s1.rms                                       # (the MAD can stay small: a gain step is constant per frame region)
    @test_throws DimensionMismatch score_seams(q.model, q.frames, q.R, view; gains = G[1:3, :], sk...)
    @test_throws DimensionMismatch score_seams(q.model, q.frames, q.R, view; gains = gw, sk...)      # K × 3, not a vector

    # the render: dividing by g_k·V makes every frame show the same scene value
    rk = (; vignetting = VgtkCore.estimate(with), circle, raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0)
    a = render_panorama(q.model, q.frames, q.R, view; gains = G, rk...)
    b = render_panorama(q.model, q.frames, q.R, view; rk...)
    o = a.owner; m = o .> 0
    @test all(i -> a.image[i, 2] * gw[o[i]] ≈ b.image[i, 2], findall(m)[1:97:end])
end

# The gain modes × the V modes: frames whose every colour plane has its own exposure gain, I_kc = g_kc·t_c·L·V (a white balance
# that drifts from frame to frame). gain_correction = nothing, FrameGains() or ChannelGains(), each with V fitted (a LogVignetting
# along its free directions) and with V held fixed (the true V as a plain callable: its log V(u) the fixed numbers). ChannelGains
# recovers every plane's gains and V; FrameGains the fitted plane's gains, every column the same; nothing, ones. score_seams and
# render_panorama read the K × 3 gains per plane.
@testitem "fit_vignetting: gain_correction none / frame / channel, V fitted or held fixed (per-plane gains recovered)" setup = [VignettingSequence] begin
    using Test, Statistics, StaticArrays
    using PanoramaStitching: fit_vignetting, score_seams, render_panorama, panorama_view, SparseCholesky, ConjugateGradient,
        FrameGains, ChannelGains, GAIN_CORRECTIONS
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: RadialVignetting, LogVignetting, pole_regular_basis, with_control_points
    c = SVector(430.0, 390.0)
    radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2 - 0.3 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^4
    G = [1.0 1.0 1.0; 1.10 1.12 1.15; 0.93 0.90 0.86; 1.25 1.20 1.12; 0.97 0.95 0.99; 1.02 1.07 1.11]   # g_1c = 1: the gauge
    q = vignetting_sequence(radial; gains = G)
    circle = VgtkCore.Circle(VgtkCore.Point2(400.0, 400.0), 390.0)
    v0 = LogVignetting(with_control_points(RadialVignetting(12, c, 420.0), zeros(12)))
    truth(u, f) = exp(radial(u))                                     # a V given as any callable
    kw = (; saturated = q.saturated, circle, raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2, plane_offsets = nothing,
          chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01, vignetting_penalty_weight = 1.0e-6, solver = SparseCholesky())
    function v_error(v)
        us = filter(u -> hypot(u[1] - 400, u[2] - 400) < 382, vec([SVector((j - 0.5) * 2, (i - 0.5) * 2) for j in 1:400, i in 1:400]))
        a = [log(v(u, nothing)) for u in us]; b = [radial(u) for u in us]
        return maximum(abs, exp.((a .- mean(a)) .- (b .- mean(b))) .- 1)
    end
    modes = (("fitted", v0, pole_regular_basis(v0)), ("held", truth, zeros(0, 0)))
    for (vm, v, Z) in modes, gc in (nothing, FrameGains(), ChannelGains())
        r = fit_vignetting(v, q.model, q.frames, q.R; free_directions = Z, gain_correction = gc, kw...)
        g = r.metadata.gains
        @info "gain_correction $(gc === nothing ? "none" : nameof(typeof(gc))), V $vm: gains error per plane " *
              "$(round.(vec(maximum(abs.(g ./ G .- 1); dims = 1)); sigdigits = 2))"
        @test size(g) == (6, 3) && all(==(1.0), g[1, :])                 # the gauge, exactly
        if gc === nothing
            @test all(==(1.0), g)
        elseif gc isa FrameGains
            @test g[:, 1] == g[:, 2] == g[:, 3] && maximum(abs.(g[:, 2] ./ G[:, 2] .- 1)) < 1.0e-3
        else
            @test maximum(abs.(g ./ G .- 1)) < 1.0e-3                     # every plane's own gains
        end
        vm == "fitted" && gc !== nothing && @test v_error(VgtkCore.estimate(r)) < 2.0e-3
        vm == "held" && @test VgtkCore.estimate(r) === truth
    end
    @test GAIN_CORRECTIONS["channel"] === ChannelGains && GAIN_CORRECTIONS["frame"] === FrameGains && GAIN_CORRECTIONS["none"] === nothing
    @test_throws ArgumentError GAIN_CORRECTIONS["per_channel"]

    # the seams and the render read each plane's gains: with ChannelGains every plane's seams vanish, with FrameGains only the fitted
    # plane's
    ch = fit_vignetting(v0, q.model, q.frames, q.R; free_directions = pole_regular_basis(v0), gain_correction = ChannelGains(), kw...)
    fr = fit_vignetting(v0, q.model, q.frames, q.R; free_directions = pole_regular_basis(v0), gain_correction = FrameGains(), kw...)
    view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.5)
    sk = (; saturated = q.saturated, circle, raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0, plane_offsets = nothing, chunk = 20_000)
    for p in (1, 3)
        sc = score_seams(q.model, q.frames, q.R, view; vignetting = VgtkCore.estimate(ch), gains = ch.metadata.gains, plane = p, sk...)
        sf = score_seams(q.model, q.frames, q.R, view; vignetting = VgtkCore.estimate(fr), gains = fr.metadata.gains, plane = p, sk...)
        @test sc.rms < 0.1 * sf.rms
    end
    rk = (; vignetting = VgtkCore.estimate(ch), circle, raw_scale = 2, rim_margin_px = 8.0, fov_deg = 120.0)
    a = render_panorama(q.model, q.frames, q.R, view; gains = ch.metadata.gains, rk...)
    b = render_panorama(q.model, q.frames, q.R, view; rk...)
    o = a.owner; m = findall(o .> 0)[1:97:end]
    @test all(i -> all(cc -> a.image[i, cc] * ch.metadata.gains[o[i], cc] ≈ b.image[i, cc], 1:3), m)

    # an other plane's unconverged solve is not hidden: its CG budget's end is the result's retcode, its iterations summed
    short = (; kw..., solver = ConjugateGradient(; tolerance = 1.0e-14, max_iter = 3))
    rc = fit_vignetting(v0, q.model, q.frames, q.R; free_directions = pole_regular_basis(v0), gain_correction = ChannelGains(), short...)
    @test VgtkCore.retcode(rc) === :max_iterations && rc.iterations == 3 * 3
    # a V held fixed that is not positive somewhere in the domain is refused by name (not Base's DomainError of the log)
    dark(u, f) = u[1] < 100 ? -0.5 : truth(u, f)
    @test_throws ArgumentError fit_vignetting(dark, q.model, q.frames, q.R; free_directions = zeros(0, 0), gain_correction = FrameGains(), kw...)
end
