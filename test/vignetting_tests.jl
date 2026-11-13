# fit_vignetting and score_seams on synthetic frames that follow the model exactly: I_k(p) = g_k·L(R_k·f(p))·V(u(p)) at every
# pixel centre (no interpolation anywhere), through a Kannala–Brandt fisheye, frames rotated in yaw AND pitch. The true V is
# recovered in VisualGeometryToolkitCore's model (a radial V about a centre, and an off-centre, asymmetric 2-D V by a tensor
# surface), the host and the device agree, CG and Cholesky agree, the gains are recovered with V free or held fixed (gain
# compensation, the same solve), and the seam score of the true V is far below that of no correction.

@testsetup module VignettingSequence
using LinearAlgebra: I, normalize
using StaticArrays: SVector, SMatrix, SA
using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, px, backproject, exp_so3, Circle, Point2, BoundedCamera, Size2
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
    camera = BoundedCamera(model, Size2(W, H); lens_coverage = circle)                 # the sensor lit inside the image circle
    return (; frames, camera, R, saturated = [falses(Hf, Wf) for _ in R])
end
end

@testitem "fit_vignetting: a radial and an off-centre 2-D V recovered, host and device, CG and Cholesky" setup = [VignettingSequence, BayerMaps] begin
    using Test, Statistics, StaticArrays
    using LinearAlgebra: I
    using PanoramaStitching: PanoramaStitching, fit_vignetting, score_seams, panorama_view, ConjugateGradient, SparseCholesky
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: RadialVignetting, TensorProductVignetting, LogVignetting, pole_regular_basis, n_control_points,
        control_points, with_control_points
    logv(s) = LogVignetting(with_control_points(s, zeros(n_control_points(s))))      # log V ≡ 0: the start
    # the domain's pixel centres (camera pixels) and the error of a fit against the truth, both in the gauge mean log V = 0
    function error_vs(v, logV, q; margin = 8.0)
        us = [FRAME_MAP(SVector(j - 0.5, i - 0.5)) for j in 1:400, i in 1:400]
        us = filter(u -> hypot(u[1] - 400, u[2] - 400) < 390 - margin, vec(us))
        a = [log(v(u)) for u in us]; b = [logV(u) for u in us]
        a .-= mean(a); b .-= mean(b)
        r = exp.(a .- b) .- 1
        return maximum(abs, r), sqrt(mean(abs2, r))
    end
    kw = (; frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0,
          plane = 2, chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01,
          vignetting_penalty_weight = 1.0e-6, gain_correction = nothing)
    cgs = ConjugateGradient(; tolerance = 1.0e-10, max_iter = 20000)

    # a radial V about a centre 30 px off the image's centre, smooth at the centre (the pole-regular subspace)
    c = SVector(430.0, 390.0)
    radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2 - 0.3 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^4
    q = vignetting_sequence(radial)
    v0 = logv(RadialVignetting(12, c, 420.0))                # the domain reaches ρ ≈ 413 px from c
    @test_throws ArgumentError fit_vignetting(VgtkCore.log_spline(v0), q.camera, q.frames, q.R; free_directions = pole_regular_basis(v0),
                                              saturated = q.saturated, solver = cgs, kw...)      # a V-form spline is only held fixed
    Z = pole_regular_basis(v0)
    cg = fit_vignetting(v0, q.camera, q.frames, q.R; free_directions = Z, saturated = q.saturated, solver = cgs, kw...)
    ch = fit_vignetting(v0, q.camera, q.frames, q.R; free_directions = Z, saturated = q.saturated, solver = SparseCholesky(), kw...)
    e_cg = error_vs(VgtkCore.estimate(cg), radial, q); e_ch = error_vs(VgtkCore.estimate(ch), radial, q)
    @info "fit_vignetting radial: max/rms error CG $(e_cg), Cholesky $(e_ch); $(cg.iterations) CG iterations, rms $(cg.rms)"
    @test VgtkCore.converged(cg) && VgtkCore.stop_rule(cg) == VgtkCore.ToleranceMet && VgtkCore.retcode_symbol(cg) === :converged
    @test VgtkCore.converged(ch) && VgtkCore.stop_rule(ch) == VgtkCore.ClosedForm && VgtkCore.retcode_symbol(ch) === :solved
    @test cg.metadata.log_change == 0 && cg.metadata.observation_noise === PanoramaStitching.IntensityNoise()
    # the noise keyword of 0.4–0.6 is refused naming its successor (spelled in fragments: check_naming retires its word)
    @test_throws "renamed `observation_noise`" fit_vignetting(v0, q.camera, q.frames, q.R; free_directions = Z, saturated = q.saturated,
                                                              solver = cgs, Symbol("measure" * "ment_noise") => PanoramaStitching.IntensityNoise(), kw...)
    @test e_cg[1] < 2.0e-3 && e_ch[1] < 2.0e-3
    @test maximum(abs.(control_points(VgtkCore.estimate(cg)) .- control_points(VgtkCore.estimate(ch)))) < 1.0e-5
    @test cg.rms < 0.02
    P = control_points(VgtkCore.estimate(cg)); @test P[1] == P[2]           # V′(0) = 0 at the centre

    # an off-centre, asymmetric 2-D V (a quartic about a displaced point and a tilt), by a tensor surface
    d = SVector(460.0, 350.0)
    tilted(u) = -1.4 * (hypot(u[1] - d[1], u[2] - d[2]) / 400)^4 - 0.6 * (hypot(u[1] - d[1], u[2] - d[2]) / 400)^2 +
                0.05 * (u[1] - 400) / 400 - 0.04 * (u[2] - 400) / 400
    q2 = vignetting_sequence(tilted)
    t0 = logv(TensorProductVignetting((16, 16), VgtkCore.Rect(10.0, 10.0, 780.0, 780.0); degree = 3))
    t = fit_vignetting(t0, q2.camera, q2.frames, q2.R; free_directions = Matrix(1.0I, n_control_points(t0), n_control_points(t0)),
                       saturated = q2.saturated, solver = SparseCholesky(), kw...)
    e2 = error_vs(VgtkCore.estimate(t), tilted, q2)
    @info "fit_vignetting tensor: max/rms error $(e2)"
    @test e2[1] < 1.0e-2 && e2[2] < 1.0e-3

    # the seam score: the true V leaves (almost) nothing, no correction leaves the vignetting's steps
    view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.5)
    sk = (; saturated = q.saturated, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2,
          chunk = 20_000)
    fitted = score_seams(q.camera, q.frames, q.R, view; vignetting = VgtkCore.estimate(cg), sk...)
    none = score_seams(q.camera, q.frames, q.R, view; vignetting = (u, f) -> 1.0, sk...)
    @info "score_seams: fitted V rms $(fitted.rms) mad $(fitted.mad), none rms $(none.rms) mad $(none.mad), $(fitted.n) pixels"
    @test fitted.n > 10_000 && fitted.n == none.n
    @test fitted.mad < 0.1 * none.mad && fitted.rms < 0.1 * none.rms
end

@testitem "fit_vignetting, score_seams: the device agrees with the host" setup = [VignettingSequence, BayerMaps] begin
    using Test, StaticArrays, CUDA
    using PanoramaStitching: fit_vignetting, score_seams, render_panorama, panorama_view, ConjugateGradient
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: RadialVignetting, LogVignetting, pole_regular_basis, control_points, with_control_points
    using PanoramaStitching: IntensityAndPositionNoise, ChannelGains
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
        kw = (; frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0,
              plane = 2, chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01,
              vignetting_penalty_weight = 1.0e-6, gain_correction = ChannelGains(), free_directions = pole_regular_basis(v0),
              solver = ConjugateGradient(; tolerance = 1.0e-10, max_iter = 20000))
        h = fit_vignetting(v0, q.camera, q.frames, q.R; saturated = q.saturated, kw...)
        df, ds = CuArray.(q.frames), CuArray.(q.saturated)
        d = fit_vignetting(v0, q.camera, df, q.R; saturated = ds, kw...)
        @test VgtkCore.converged(d)
        # the atomic sums run in another order on each backend (and from run to run): V agrees to the solve's precision
        @test maximum(abs.(control_points(VgtkCore.estimate(d)) .- control_points(VgtkCore.estimate(h)))) < 1.0e-6
        @test abs(d.rms - h.rms) < 1.0e-8
        # the value-and-position noise's weights on the device (the observation's constraint, ForwardDiff and Core's whitening
        # in the kernels)
        sw = IntensityAndPositionNoise(; point_covariance = VgtkCore.CovMat{2}(σ = 1.0), iterations = 2)
        hs = fit_vignetting(v0, q.camera, q.frames, q.R; saturated = q.saturated, observation_noise = sw, kw...)
        dsf = fit_vignetting(v0, q.camera, df, q.R; saturated = ds, observation_noise = sw, kw...)
        @test maximum(abs.(control_points(VgtkCore.estimate(dsf)) .- control_points(VgtkCore.estimate(hs)))) < 1.0e-6
        @test maximum(abs.(dsf.metadata.gains .- hs.metadata.gains)) < 1.0e-8 && abs(dsf.rms - hs.rms) < 1.0e-6
        view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.5)
        sk = (; frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2,
              chunk = 20_000, vignetting = VgtkCore.estimate(h))
        sh = score_seams(q.camera, q.frames, q.R, view; saturated = q.saturated, sk...)
        sd = score_seams(q.camera, df, q.R, view; saturated = ds, sk...)
        @test sh.n == sd.n && abs(sh.rms - sd.rms) < 1.0e-9 && abs(sh.mad - sd.mad) < 1.0e-9
        @test maximum(abs.(d.metadata.gains .- h.metadata.gains)) < 1.0e-8 && all(==(1.0), d.metadata.gains[1, :])
        # a V held fixed that holds HOST arrays (a calibration's table, with its Adapt rule) reaches the device in the fit, the
        # render and the seams (the functors are moved by OnBackend)
        tv = TableV(collect(range(1.0, 0.6; length = 64)), c, 420.0)
        hf = fit_vignetting(tv, q.camera, q.frames, q.R; saturated = q.saturated, kw..., free_directions = zeros(0, 0))
        dfit = fit_vignetting(tv, q.camera, df, q.R; saturated = ds, kw..., free_directions = zeros(0, 0))
        @test maximum(abs.(dfit.metadata.gains .- hf.metadata.gains)) < 1.0e-8
        rk = (; vignetting = tv, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0)
        a = render_panorama(q.camera, q.frames, q.R, view; rk...); b = render_panorama(q.camera, df, q.R, view; rk...)
        @test isequal(Array(b.owner), a.owner) && maximum(abs, filter(!isnan, vec(Array(b.image) .- a.image))) < 1.0e-5
        sdt = score_seams(q.camera, df, q.R, view; saturated = ds, sk..., vignetting = tv)
        @test sdt.n == score_seams(q.camera, q.frames, q.R, view; saturated = q.saturated, sk..., vignetting = tv).n
    end
end

# The frames' exposure gains: frames of the same scene and V at different exposures, I_k = g_k·L·V. With gain_correction the fit
# recovers the gains (gauge g₁ = 1, exactly) and V; without them the exposure differences go into V and the scene, and V is
# wrong. With V held fixed (no free direction: gain compensation under a calibration's V) the same solve recovers the gains
# alone. score_seams and render_panorama divide the gains out.
@testitem "fit_vignetting: the frames' exposure gains recovered (gauge g₁ = 1), V free or held fixed; score_seams and render_panorama divide them out" setup = [VignettingSequence, BayerMaps] begin
    using Test, Statistics, StaticArrays
    using PanoramaStitching: PanoramaStitching, fit_vignetting, score_seams, render_panorama, panorama_view, ConjugateGradient,
        SparseCholesky, IntensityAndPositionNoise, IntensityGains
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: RadialVignetting, LogVignetting, pole_regular_basis, n_control_points, with_control_points
    c = SVector(430.0, 390.0)
    radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2 - 0.3 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^4
    g = [1.0, 1.12, 0.9, 1.2, 0.95, 1.07]
    q = vignetting_sequence(radial; gains = g)
    v0 = LogVignetting(with_control_points(RadialVignetting(12, c, 420.0), zeros(12)))
    kw = (; saturated = q.saturated, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2,
          chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01, vignetting_penalty_weight = 1.0e-6)
    cgs = ConjugateGradient(; tolerance = 1.0e-10, max_iter = 20000)
    function v_error(v)
        us = filter(u -> hypot(u[1] - 400, u[2] - 400) < 382, vec([FRAME_MAP(SVector(j - 0.5, i - 0.5)) for j in 1:400, i in 1:400]))
        a = [log(v(u)) for u in us]; b = [radial(u) for u in us]
        return maximum(abs, exp.((a .- mean(a)) .- (b .- mean(b))) .- 1)
    end
    Z = pole_regular_basis(v0)
    with = fit_vignetting(v0, q.camera, q.frames, q.R; free_directions = Z, gain_correction = IntensityGains(), solver = cgs, kw...)
    chol = fit_vignetting(v0, q.camera, q.frames, q.R; free_directions = Z, gain_correction = IntensityGains(), solver = SparseCholesky(), kw...)
    without = fit_vignetting(v0, q.camera, q.frames, q.R; free_directions = Z, gain_correction = nothing, solver = cgs, kw...)
    G = with.metadata.gains; gw = G[:, 2]
    @test size(G) == (6, 3) && G[:, 1] == G[:, 2] == G[:, 3]          # IntensityGains: one per frame, every plane
    @info "frame gains: recovered $(round.(gw; digits = 5)) (true $g); V error with $(v_error(VgtkCore.estimate(with))), without $(v_error(VgtkCore.estimate(without)))"
    @test VgtkCore.converged(with)
    @test gw[1] === 1.0 && length(gw) == length(g)                    # the gauge: the first frame's gain is 1 exactly
    @test maximum(abs.(gw ./ g .- 1)) < 1.0e-3
    @test maximum(abs.(chol.metadata.gains .- G)) < 1.0e-5            # CG and Cholesky solve the same system
    @test v_error(VgtkCore.estimate(with)) < 2.0e-3
    @test v_error(VgtkCore.estimate(without)) > 5 * v_error(VgtkCore.estimate(with))
    @test all(==(1.0), without.metadata.gains)
    @test with.rms < 0.5 * without.rms

    # gain compensation: V held fixed at the fitted one (no free direction), the gains alone, one linear solve
    fixed = fit_vignetting(VgtkCore.estimate(with), q.camera, q.frames, q.R; free_directions = zeros(n_control_points(v0), 0),
                           gain_correction = IntensityGains(), solver = SparseCholesky(), kw...)
    @test VgtkCore.estimate(fixed) === VgtkCore.estimate(with) && fixed.metadata.log_change == 0
    @test maximum(abs.(fixed.metadata.gains[:, 2] ./ g .- 1)) < 1.0e-3
    @test maximum(abs.(fixed.metadata.gains .- G)) < 1.0e-5

    # the value-and-position noise (the observation's constraint whitened with the sensor point's noise too): frames that follow
    # the model exactly give the same V and gains; the weights come from the I² solve's model, so V held fixed takes them too
    sw = IntensityAndPositionNoise(; point_covariance = VgtkCore.CovMat{2}(σ = 1.5), iterations = 2)
    sam = fit_vignetting(v0, q.camera, q.frames, q.R; free_directions = Z, gain_correction = IntensityGains(), solver = SparseCholesky(),
                         observation_noise = sw, kw...)
    @test sam.metadata.observation_noise === sw && sam.metadata.log_change > 0   # two reweight-and-solve cycles after the I² solve
    @test maximum(abs.(sam.metadata.gains[:, 2] ./ g .- 1)) < 1.0e-3 && v_error(VgtkCore.estimate(sam)) < 2.0e-3
    # the termination is every solve's (Core's worst_retcode), not the last one's: the cold first solve capped by the CG
    # budget, the warm-started reweighted solves after it within it — the fit is MaxIters (its legacy Symbol :max_iterations)
    cg(n) = ConjugateGradient(; tolerance = 1.0e-10, max_iter = n)
    single = fit_vignetting(v0, q.camera, q.frames, q.R; free_directions = Z, gain_correction = IntensityGains(), (; kw..., solver = cg(100_000))...)
    @test VgtkCore.converged(single) && VgtkCore.stop_rule(single) == VgtkCore.ToleranceMet
    budget = single.iterations ÷ 2                           # the cold solve capped well short (its CG count varies by an iteration or so: threaded scatter-adds)
    capped = fit_vignetting(v0, q.camera, q.frames, q.R; free_directions = Z, gain_correction = IntensityGains(),
                            (; kw..., solver = cg(budget), observation_noise = sw)...)
    @test capped.iterations < 3 * budget                     # a later solve met the tolerance within the budget
    @test VgtkCore.retcode(capped) == VgtkCore.ReturnCode.MaxIters && VgtkCore.stop_rule(capped) == VgtkCore.IterationCap
    @test VgtkCore.retcode_symbol(capped) === :max_iterations && !VgtkCore.converged(capped)
    sf = fit_vignetting(VgtkCore.estimate(with), q.camera, q.frames, q.R; free_directions = zeros(n_control_points(v0), 0),
                        gain_correction = IntensityGains(), solver = SparseCholesky(),
                        observation_noise = IntensityAndPositionNoise(; point_covariance = VgtkCore.CovMat{2}(σ = 1.5)), kw...)
    @test sf.metadata.observation_noise.iterations == 1 && maximum(abs.(sf.metadata.gains[:, 2] ./ g .- 1)) < 1.0e-3
    @test_throws UndefKeywordError IntensityAndPositionNoise()                                     # the point's noise is required
    @test_throws TypeError IntensityAndPositionNoise(; point_covariance = [1.0 0; 0 1])            # a CovMat, not a bare matrix
    @test_throws ArgumentError IntensityAndPositionNoise(; point_covariance = VgtkCore.CovMat{2}(σ = 1.0), iterations = 3)
    @test_throws ArgumentError IntensityAndPositionNoise(; point_covariance = VgtkCore.CovMat(SMatrix{2, 2}(1.0, 2.0, 2.0, 1.0)))   # not positive definite

    # the observation's constraint (Core's protocol): G_x is the model's exact gradient in u, and the weight is Core's whitening
    m = with.metadata
    cL = replace(m.scene, NaN => 0.0)
    obs = PanoramaStitching._LogIntensityModel(VgtkCore.estimate(with), m.chart, cL, q.R[3], log(m.gains[3, 2]))
    for u in (SVector(300.0, 420.0), SVector(520.0, 260.0), SVector(410.0, 395.0))
        f, M = VgtkCore.backproject_with_jacobian(VgtkCore.model(q.camera), u)
        x = SVector(u[1], u[2], -2.0)
        gx, G_x, G_θ = VgtkCore.constraint_with_jacobians(obs, x; ray = f, ray_jacobian = M)
        gat(w) = VgtkCore.constraint(obs, SVector(w[1], w[2], -2.0); ray = VgtkCore.backproject(q.camera, w),
                                     ray_jacobian = VgtkCore.backproject_with_jacobian(VgtkCore.model(q.camera), w)[2])
        h = 1.0e-4
        fd = SVector((gat(u + SVector(h, 0)) - gat(u - SVector(h, 0))) / 2h, (gat(u + SVector(0, h)) - gat(u - SVector(0, h))) / 2h)
        @test isapprox(SVector(G_x[1], G_x[2]), fd; rtol = 1.0e-5, atol = 1.0e-9) && G_x[3] == 1.0
        @test length(G_θ) == 16 + 4 + 1 && G_θ[end] == -1.0
        Σx = PanoramaStitching._noise_covariance(IntensityAndPositionNoise(; point_covariance = VgtkCore.CovMat{2}(σ = 1.5)))
        @test VgtkCore.whitening_factor(G_x, Σx)^2 ≈ 1 / (1.5^2 * (G_x[1]^2 + G_x[2]^2) + 0.01^2)
    end

    # the seams: the fitted V with its gains leaves (almost) nothing; the same V with the frames taken at one exposure does not
    view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.5)
    sk = (; saturated = q.saturated, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2,
          chunk = 20_000, vignetting = VgtkCore.estimate(with))
    sg = score_seams(q.camera, q.frames, q.R, view; gains = G, sk...)
    s1 = score_seams(q.camera, q.frames, q.R, view; sk...)
    @info "score_seams with the gains: rms $(sg.rms) mad $(sg.mad); without: rms $(s1.rms) mad $(s1.mad)"
    @test sg.rms < 0.1 * s1.rms                                       # (the MAD can stay small: a gain step is constant per frame region)
    @test_throws DimensionMismatch score_seams(q.camera, q.frames, q.R, view; gains = G[1:3, :], sk...)
    @test_throws DimensionMismatch score_seams(q.camera, q.frames, q.R, view; gains = gw, sk...)      # K × 3, not a vector

    # the render: dividing by g_k·V makes every frame show the same scene value
    rk = (; vignetting = VgtkCore.estimate(with), frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0)
    a = render_panorama(q.camera, q.frames, q.R, view; gains = G, rk...)
    b = render_panorama(q.camera, q.frames, q.R, view; rk...)
    o = a.owner; m = o .> 0
    @test all(i -> a.image[i, 2] * gw[o[i]] ≈ b.image[i, 2], findall(m)[1:97:end])
end

# The gain modes × the V modes: frames whose every colour plane has its own exposure gain, I_kc = g_kc·t_c·L·V (a white balance
# that drifts from frame to frame). gain_correction = nothing, IntensityGains() or ChannelGains(), each with V fitted (a
# LogVignetting along its free directions) and with V held fixed (the true V as a plain callable: its log V(u) the fixed numbers).
# ChannelGains recovers every plane's gains and V; IntensityGains the fitted plane's gains, every column the same; nothing, ones.
# score_seams and render_panorama read the K × 3 gains per plane.
@testitem "fit_vignetting: gain_correction none / intensity / channel, V fitted or held fixed (per-plane gains recovered)" setup = [VignettingSequence, BayerMaps] begin
    using Test, Statistics, StaticArrays
    using PanoramaStitching: fit_vignetting, score_seams, render_panorama, panorama_view, SparseCholesky, ConjugateGradient,
        IntensityGains, ChannelGains, GAIN_CORRECTIONS
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: RadialVignetting, LogVignetting, pole_regular_basis, with_control_points
    c = SVector(430.0, 390.0)
    radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2 - 0.3 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^4
    G = [1.0 1.0 1.0; 1.10 1.12 1.15; 0.93 0.90 0.86; 1.25 1.20 1.12; 0.97 0.95 0.99; 1.02 1.07 1.11]   # g_1c = 1: the gauge
    q = vignetting_sequence(radial; gains = G)
    v0 = LogVignetting(with_control_points(RadialVignetting(12, c, 420.0), zeros(12)))
    truth(u, f) = exp(radial(u))                                     # a V given as any callable
    kw = (; saturated = q.saturated, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2,
          chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01, vignetting_penalty_weight = 1.0e-6, solver = SparseCholesky())
    function v_error(v)
        us = filter(u -> hypot(u[1] - 400, u[2] - 400) < 382, vec([FRAME_MAP(SVector(j - 0.5, i - 0.5)) for j in 1:400, i in 1:400]))
        a = [log(v(u, nothing)) for u in us]; b = [radial(u) for u in us]
        return maximum(abs, exp.((a .- mean(a)) .- (b .- mean(b))) .- 1)
    end
    modes = (("fitted", v0, pole_regular_basis(v0)), ("held", truth, zeros(0, 0)))
    for (vm, v, Z) in modes, gc in (nothing, IntensityGains(), ChannelGains())
        r = fit_vignetting(v, q.camera, q.frames, q.R; free_directions = Z, gain_correction = gc, kw...)
        g = r.metadata.gains
        @info "gain_correction $(gc === nothing ? "none" : nameof(typeof(gc))), V $vm: gains error per plane " *
              "$(round.(vec(maximum(abs.(g ./ G .- 1); dims = 1)); sigdigits = 2))"
        @test size(g) == (6, 3) && all(==(1.0), g[1, :])                 # the gauge, exactly
        if gc === nothing
            @test all(==(1.0), g)
        elseif gc isa IntensityGains
            @test g[:, 1] == g[:, 2] == g[:, 3] && maximum(abs.(g[:, 2] ./ G[:, 2] .- 1)) < 1.0e-3
        else
            @test maximum(abs.(g ./ G .- 1)) < 1.0e-3                     # every plane's own gains
        end
        vm == "fitted" && gc !== nothing && @test v_error(VgtkCore.estimate(r)) < 2.0e-3
        vm == "held" && @test VgtkCore.estimate(r) === truth
    end
    @test GAIN_CORRECTIONS["channel"] === ChannelGains && GAIN_CORRECTIONS["intensity"] === IntensityGains &&
          GAIN_CORRECTIONS["none"] === nothing
    @test_throws ArgumentError GAIN_CORRECTIONS["per_channel"]

    # the seams and the render read each plane's gains: with ChannelGains every plane's seams vanish, with IntensityGains only the
    # fitted plane's
    ch = fit_vignetting(v0, q.camera, q.frames, q.R; free_directions = pole_regular_basis(v0), gain_correction = ChannelGains(), kw...)
    fr = fit_vignetting(v0, q.camera, q.frames, q.R; free_directions = pole_regular_basis(v0), gain_correction = IntensityGains(), kw...)
    view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.5)
    sk = (; saturated = q.saturated, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, chunk = 20_000)
    for p in (1, 3)
        sc = score_seams(q.camera, q.frames, q.R, view; vignetting = VgtkCore.estimate(ch), gains = ch.metadata.gains, plane = p, sk...)
        sf = score_seams(q.camera, q.frames, q.R, view; vignetting = VgtkCore.estimate(fr), gains = fr.metadata.gains, plane = p, sk...)
        @test sc.rms < 0.1 * sf.rms
    end
    rk = (; vignetting = VgtkCore.estimate(ch), frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0)
    a = render_panorama(q.camera, q.frames, q.R, view; gains = ch.metadata.gains, rk...)
    b = render_panorama(q.camera, q.frames, q.R, view; rk...)
    o = a.owner; m = findall(o .> 0)[1:97:end]
    @test all(i -> all(cc -> a.image[i, cc] * ch.metadata.gains[o[i], cc] ≈ b.image[i, cc], 1:3), m)

    # an other plane's unconverged solve is not hidden: its CG budget's end is the result's retcode, its iterations summed
    short = (; kw..., solver = ConjugateGradient(; tolerance = 1.0e-14, max_iter = 3))
    rc = fit_vignetting(v0, q.camera, q.frames, q.R; free_directions = pole_regular_basis(v0), gain_correction = ChannelGains(), short...)
    @test VgtkCore.retcode(rc) == VgtkCore.ReturnCode.MaxIters && VgtkCore.stop_rule(rc) == VgtkCore.IterationCap && rc.iterations == 3 * 3
    # a V held fixed that is not positive somewhere in the domain is refused by name (not Base's DomainError of the log)
    dark(u, f) = u[1] < 100 ? -0.5 : truth(u, f)
    @test_throws ArgumentError fit_vignetting(dark, q.camera, q.frames, q.R; free_directions = zeros(0, 0), gain_correction = IntensityGains(), kw...)
end

# The TOML tags of the gain correction and the observation noise: the current tags select their types (the tag rule: snake_case
# of the type's name less the family's suffix), and the tags of earlier releases are refused with an ArgumentError naming the new
# one (no alias: an old config fails, it is never read as the new setting).
@testitem "GAIN_CORRECTIONS, OBSERVATION_NOISES: the tags, and the renamed tags refused by name" begin
    using Test
    using PanoramaStitching: PanoramaStitching, IntensityGains, ChannelGains, IntensityNoise, IntensityAndPositionNoise
    import VisualGeometryToolkitCore as VgtkCore
    G = PanoramaStitching.GAIN_CORRECTIONS; N = PanoramaStitching.OBSERVATION_NOISES
    @test G["intensity"] === IntensityGains && G["channel"] === ChannelGains && G["none"] === nothing
    @test N["intensity"] === IntensityNoise && N["intensity_and_position"] === IntensityAndPositionNoise
    @test VgtkCore.tag_of(G, IntensityGains) == "intensity" && VgtkCore.tag_of(N, IntensityAndPositionNoise) == "intensity_and_position"
    renamed(reg, tag) = try
        reg[tag]; ""
    catch e
        e isa ArgumentError ? sprint(showerror, e) : rethrow()
    end
    m = renamed(G, "frame"); @test occursin("renamed", m) && occursin("\"intensity\"", m)
    m = renamed(N, "sampson"); @test occursin("renamed", m) && occursin("\"intensity_and_position\"", m)
    @test_throws ArgumentError N["weights"]
    @test isempty(VgtkCore.tag_rule_violations(G)) && isempty(VgtkCore.tag_rule_violations(N))
end

# The example's TOML keys of earlier releases are refused by name (run.jl's `refuse_renamed_keys`, which `load_config` and
# compare_arms.jl call, and `refuse_circle`, which `load_config` calls: the image circle is the camera's lens coverage): the
# example's own environment is not the tests', so their definitions are read from run.jl alone; the configs of record and
# arms.toml hold the current keys and tags.
@testitem "panorama_vignetting example: the renamed and retired TOML keys refused by name; the configs of record are current" begin
    using Test, TOML
    using PanoramaStitching: PanoramaStitching
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    dir =joinpath(pkgdir(PanoramaStitching), "examples", "panorama_vignetting")
    code = Meta.parseall(read(joinpath(dir, "run.jl"), String))
    defines(x, name) = x isa Expr && (x.head === :const ? defines(x.args[1], name) : x.head === :(=) ? x.args[1] === name :
                                      x.head === :function && x.args[1] isa Expr && x.args[1].args[1] === name)
    m = Module()
    for name in (:RENAMED_KEYS, :refuse_renamed_keys, :refuse_circle, :RETIRED_REGISTRATION_KEYS, :refuse_retired_registration,
                 :refuse_output, :refuse_inert_pair_keys)
        Core.eval(m, only(filter(x -> defines(x, name), code.args)))
    end
    refuse(t) = Base.invokelatest(m.refuse_renamed_keys, t, "config")
    for (old, new) in (("weights", "observation_noise"), ("sampson_iterations", "observation_noise_iterations"),
                       ("measure" * "ment_noise", "observation_noise"), ("measure" * "ment_noise_iterations", "observation_noise_iterations"))
        err = try
            refuse(Dict{String, Any}(old => 1)); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("the key $old was renamed to $new", err.msg)
    end
    # `[inputs] circle` is refused, naming the camera's lens coverage
    err = try
        Base.invokelatest(m.refuse_circle, Dict{String, Any}("circle" => [1.0, 2.0, 3.0]), "config [inputs]"); nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("lens_coverage", err.msg) && occursin("circle", err.msg)
    # the top-level `output` folder is retired (the results go under Eval's results root, keyed): refused by name, and the
    # configs of record name none
    err = try
        Base.invokelatest(m.refuse_output, Dict{String, Any}("output" => "out/x"), "config"); nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("output is retired", err.msg) && occursin("results_root", err.msg)
    for f in ("config_fx3_14mm.toml", "config_fx3_8mm_rooftop.toml")
        @test !haskey(TOML.parsefile(joinpath(dir, f)), "output")
    end
    # `[registration] bundle_iterations` is retired: refused by name
    err = try
        Base.invokelatest(m.refuse_retired_registration, Dict{String, Any}("bundle_iterations" => 3), "config [registration]"); nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("bundle_iterations", err.msg)
    # the 0.6 noise constants are refused by name: the frames' noise is Core's resolver's
    for k in ("electrons_per_dn", "read_noise_dn")
        @test_throws "sensor_noise" Base.invokelatest(m.refuse_retired_registration, Dict{String, Any}(k => 1.0), "config [registration]")
    end
    # `max_lag` / `overlap_deg` with pairs = "all" are refused when the config is read (no inert setting, cached or not)
    inert(t) = Base.invokelatest(m.refuse_inert_pair_keys, Dict{String, Any}(t), "config [registration]")
    @test_throws "read only by pairs" inert(("pairs" => "all", "max_lag" => 1))
    @test_throws "read only by pairs" inert(("pairs" => "all", "overlap_deg" => 120.0))
    @test inert(("pairs" => "lag_overlap", "max_lag" => 1, "overlap_deg" => 120.0))["max_lag"] == 1
    G = PanoramaStitching.GAIN_CORRECTIONS; N = PanoramaStitching.OBSERVATION_NOISES
    for f in ("config_fx3_14mm.toml", "config_fx3_8mm_rooftop.toml")
        inputs = TOML.parsefile(joinpath(dir, f))["inputs"]
        @test Base.invokelatest(m.refuse_inert_pair_keys, TOML.parsefile(joinpath(dir, f))["registration"], f) isa AbstractDict
        @test Base.invokelatest(m.refuse_circle, inputs, f) === inputs
        # the view-graph filters' tolerances and the detector's variant are stated
        reg = TOML.parsefile(joinpath(dir, f))["registration"]
        @test all(k -> haskey(reg, k), ("cycle_tol_deg", "orientation_tol_deg", "selection", "detection_image", "local_optimization_strategy"))
        # the pairs' local optimization is an Estimation strategy table, read by Estimation's own loader
        @test VgtkEstimation.optimization_strategy_from_toml(reg["local_optimization_strategy"]; class = :rotation) isa VgtkCore.AbstractOptimizationStrategy
        v = TOML.parsefile(joinpath(dir, f))["vignetting"]
        @test refuse(v) === v && G[v["gain_correction"]] isa Any && N[v["observation_noise"]] isa Type
        # the V models are Core's VIGNETTING_TAGS of the spline log V is (the retired "tensor" refused by name)
        @test all(k -> VgtkCore.VIGNETTING_TAGS[k] in (VgtkCore.RadialVignetting, VgtkCore.TensorProductVignetting),
                  vcat(v["model"], v["compare_models"]))
    end
    @test_throws ArgumentError VgtkCore.VIGNETTING_TAGS["tensor"]
    arms = TOML.parsefile(joinpath(dir, "arms.toml"))
    @test VgtkCore.VIGNETTING_TAGS[arms["kind"]] === VgtkCore.RadialVignetting
    @test refuse(arms) === arms && length(arms["arms"]) == 12 && allunique(a["name"] for a in arms["arms"])
    for a in arms["arms"]
        @test refuse(a) === a && N[a["observation_noise"]] isa Type && (G[a["gain_correction"]]; true)
        @test !any(t -> occursin(t, a["name"]), ("intensity", "channel", "none", "frame", "sampson", "free", "fixed"))
    end
end

# The example's stage cache (the verified pairs reused when only the averaging or the bundle changes, recomputed when a pair
# setting changes) needs VisualGeometryToolkitEval, which is no test dependency (the stack's allow-list): it is checked by
# examples/panorama_vignetting/stage_cache_check.jl, run by run_checks.jl in the example's environment (README, "Maintainers").
@testitem "panorama_vignetting example: run.jl parses, and the detector's settings" begin
    using Test
    using PanoramaStitching: PanoramaStitching
    dir = joinpath(pkgdir(PanoramaStitching), "examples", "panorama_vignetting")
    code = Meta.parseall(read(joinpath(dir, "run.jl"), String))
    @test !any(x -> Meta.isexpr(x, (:error, :incomplete)), code.args)
    for f in ("stage_cache_check.jl", "autocalibration_check.jl", "run_checks.jl")       # the example's checks parse too
        @test !any(x -> Meta.isexpr(x, (:error, :incomplete)), Meta.parseall(read(joinpath(dir, f), String)).args)
    end
    named(x, name) = x === name || (x isa Expr && x.head === :call && x.args[1] === name)
    defines(x, name) = x isa Expr && (x.head === :const ? defines(x.args[1], name) : x.head in (:(=), :function) && named(x.args[1], name))
    m = Module()
    for name in (:DETECTION_NOISE_KEYS, :detector_settings)
        Core.eval(m, only(filter(x -> defines(x, name), code.args)))
    end
    call(f, args...; kw...) = Base.invokelatest(getfield(m, f), args...; kw...)
    # the detector's settings: "vst" reads the noise keys, "tone" refuses them (never read)
    det(r) = call(:detector_settings, Dict{String, Any}(r))
    @test det(Dict("selection" => "spread", "detection_image" => "tone")) ==
          (; selection = :spread, detection_image = :tone, noise_sigmas = nothing, record = nothing)
    v = det(Dict("selection" => "spread", "detection_image" => "vst", "noise_sigmas" => 3, "sensor_noise_record" => "none"))
    @test v == (; selection = :spread, detection_image = :vst, noise_sigmas = 3.0, record = "none")
    @test_throws ArgumentError det(Dict("selection" => "spread", "detection_image" => "tone", "noise_sigmas" => 3))
    @test_throws ArgumentError det(Dict("selection" => "spread", "detection_image" => "tone", "sensor_noise_record" => "none"))
    @test_throws KeyError det(Dict("selection" => "spread", "detection_image" => "vst"))
end
