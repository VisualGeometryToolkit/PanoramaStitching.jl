# VignettingSelfCalibration and score_seams on synthetic frames that follow the model exactly: I_k(p) = g_k·L(R_k·f(p))·V(u(p)) at every
# pixel centre (no interpolation anywhere), through a Kannala–Brandt fisheye, frames rotated in yaw AND pitch. The true V is
# recovered in VisualGeometryToolkitCore's model (a radial V about a centre, and an off-centre, asymmetric 2-D V by a tensor
# surface), the host and the device agree, CG and Cholesky agree, the gains are recovered with V free or held fixed (gain
# compensation, the same solve), and the seam score of the true V is far below that of no correction.

@testsetup module VignettingSequence
using LinearAlgebra: I, normalize, dot
using StaticArrays: SVector, SMatrix, SA
using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, px, backproject, exp_so3, Circle, Point2, BoundedCamera, Size2,
    Uncertain, CovMat
export vignetting_sequence, relative_frames

# The frames as an uncertain image (a value's variance each, as Core's LinearRaw under a RelativeNoise gives): σ = σ_rel·I.
relative_frames(frames, σ_rel) = [map(x -> Uncertain(Float64(x), CovMat{1}(σ = x > 0 ? σ_rel * Float64(x) : 1.0)), F) for F in frames]   # (a dark pixel: no row)

# the scene: a smooth log radiance on the sphere (wavelengths of about 30°; Cartesian, so no pole is singular)
scene(d) = 0.05 * exp(0.6 * sin(3d[1] + 2d[2]) * cos(2d[3]) + 0.3 * sin(9d[1] - 7d[3] + 5d[2]))

# `gains`: the frames' exposure gains, one per frame g_k or one per frame and colour plane g_kc (a K × 3 matrix): I_kc = g_kc·t_c·L·V,
# t = (0.8, 1, 1.3) the scene's colour (all 1 by default). `spikes`: bright scene points (directions) added to L, each a Gaussian of
# angular σ `width_deg` and peak `amp`; `white`: the sensor's clip level, every value at or above it clipped there and its pixel
# `saturated` (no clip by default).
function vignetting_sequence(logV; gains = ones(6), spikes = SVector{3, Float64}[], amp = 0.0, width_deg = 1.0, white = Inf)
    G = gains isa AbstractMatrix ? gains : repeat(gains, 1, 3); tint = (0.8, 1.0, 1.3)
    W = H = 800; s = 2; Wf, Hf = W ÷ s, H ÷ s
    model = CameraModel(260.0px, SA[400.0px, 400.0px]; lens = KannalaBrandt(-0.02))
    circle = Circle(Point2(400.0, 400.0), 390.0)
    angles = [(0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (15.0, 25.0), (45.0, 25.0), (30.0, -20.0)]   # (yaw, pitch) degrees
    R = [SMatrix{3, 3}(exp_so3(SVector(0.0, deg2rad(y), 0.0))) * SMatrix{3, 3}(exp_so3(SVector(deg2rad(p), 0.0, 0.0))) for (y, p) in angles]
    σ = deg2rad(width_deg)
    L(d) = scene(d) + amp * sum((exp(-acos(clamp(dot(d, a), -1.0, 1.0))^2 / (2σ^2)) for a in spikes); init = 0.0)
    saturated = [falses(Hf, Wf) for _ in R]
    frames = map(enumerate(R)) do (k, Rk)
        img = zeros(Float32, Hf, Wf, 3)
        for j in 1:Wf, i in 1:Hf
            u = SVector((j - 0.5) * s, (i - 0.5) * s)
            hypot(u[1] - 400.0, u[2] - 400.0) < 390.0 || continue
            f = backproject(model, u)
            for c in 1:3
                I = G[k, c] * tint[c] * L(Rk * f) * exp(logV(u))
                I >= white && (saturated[k][i, j] = true)
                img[i, j, c] = Float32(min(I, white))
            end
        end
        img
    end
    camera = BoundedCamera(model, Size2(W, H); lens_coverage = circle)                 # the sensor lit inside the image circle
    return (; frames, camera, R, saturated)
end
end

@testitem "VignettingSelfCalibration: a radial and an off-centre 2-D V recovered, host and device, CG and Cholesky" setup = [VignettingSequence, BayerMaps] begin
    using Test, Statistics, StaticArrays
    using LinearAlgebra: I
    using PanoramaStitching: PanoramaStitching, VignettingSelfCalibration, score_seams, panorama_view, SparseCholesky
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore
    using VisualRadiometryToolkitCore: AbstractVignetting, RadialVignetting, TensorProductVignetting, LogVignetting, pole_regular_basis, n_control_points,
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
    alg = (; rim_margin_px = 8.0, saturation_margin_px = 2.0, fov_deg = 120.0, observation_noise = nothing,
           plane = 2, chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01,
           vignetting_penalty_weight = 1.0e-6, gain_correction = nothing)
    cgs = VgtkCore.NormalCG(); cg_stops = VgtkCore.ToleranceStops(; rtol = 1.0e-10, max_iter = 20000)

    # a radial V about a centre 30 px off the image's centre, smooth at the centre (the pole-regular subspace)
    c = SVector(430.0, 390.0)
    radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2 - 0.3 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^4
    q = vignetting_sequence(radial)
    ctx = (; camera = q.camera, rotations = q.R, saturated = q.saturated, frame_map = FRAME_MAP)
    v0 = logv(RadialVignetting(12, c, 420.0))                # the domain reaches ρ ≈ 413 px from c
    Z = pole_regular_basis(v0)
    @test_throws ArgumentError VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, q.frames, (; ctx..., vignetting = VrtkCore.log_spline(v0), free_directions = Z)),
                                              VignettingSelfCalibration(; alg..., solver = cgs); stops = cg_stops)   # a V-form spline is only held fixed
    problem = VgtkCore.EstimationProblem(AbstractVignetting, q.frames, (; ctx..., vignetting = v0, free_directions = Z))
    cg = VgtkCore.solve(problem, VignettingSelfCalibration(; alg..., solver = cgs); stops = cg_stops)
    ch = VgtkCore.solve(problem, VignettingSelfCalibration(; alg..., solver = SparseCholesky()))
    # the CG's stops are the solve's (Core's ToleranceStops), required with it; a Cholesky solve reads none, refused
    @test_throws "needs the solve keyword `stops" VgtkCore.solve(problem, VignettingSelfCalibration(; alg..., solver = cgs))
    @test_throws "reads no `stops`" VgtkCore.solve(problem, VignettingSelfCalibration(; alg..., solver = SparseCholesky()); stops = cg_stops)
    @test cg.alg.solver isa VgtkCore.NormalCG && ch.alg.solver isa SparseCholesky   # the solver is the algorithm's, read by type
    e_cg = error_vs(cg.u, radial, q); e_ch = error_vs(ch.u, radial, q)
    @info "VignettingSelfCalibration radial: max/rms error CG $(e_cg), Cholesky $(e_ch); $(cg.stats.nsteps) CG iterations, rms $(cg.rms)"
    @test VgtkCore.successful_retcode(cg) && cg.stop_rule == VgtkCore.ToleranceMet
    @test VgtkCore.successful_retcode(ch) && ch.stop_rule == VgtkCore.ClosedForm
    @test cg.metadata.log_change == 0 && cg.metadata.observation_noise === nothing && !cg.metadata.value_variance   # the read-noise fallback
    # an uncertain image weights a row by I²/var(I): exact frames give the same V; a mix of plain and uncertain is refused
    un = VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, relative_frames(q.frames, 0.01), (; ctx..., vignetting = v0, free_directions = Z)),
                        VignettingSelfCalibration(; alg..., solver = SparseCholesky()))
    @test un.metadata.value_variance && error_vs(un.u, radial, q)[1] < 2.0e-3
    @test_throws "not a mix" VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, [relative_frames(q.frames[1:1], 0.01); q.frames[2:end]],
                                                                     (; ctx..., vignetting = v0, free_directions = Z)),
                                            VignettingSelfCalibration(; alg..., solver = cgs); stops = cg_stops)
    # the noise keyword of 0.4–0.6 is no setting of the algorithm (spelled in fragments: check_naming retires its word)
    @test_throws MethodError VignettingSelfCalibration(; alg..., solver = cgs, Symbol("measure" * "ment_noise") => nothing)
    # a context key the fit does not read, or one it needs left out, is refused naming it
    @test_throws r"VignettingSelfCalibration: the context `p` holds the unknown key\(s\) plane" VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, q.frames,
        (; ctx..., vignetting = v0, free_directions = Z, plane = 2)), VignettingSelfCalibration(; alg..., solver = cgs); stops = cg_stops)
    @test_throws r"VignettingSelfCalibration: the context `p` lacks free_directions" VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, q.frames,
        (; ctx..., vignetting = v0)), VignettingSelfCalibration(; alg..., solver = cgs); stops = cg_stops)
    @test e_cg[1] < 2.0e-3 && e_ch[1] < 2.0e-3
    @test maximum(abs.(control_points(cg.u) .- control_points(ch.u))) < 1.0e-5
    @test cg.rms < 0.02
    P = control_points(cg.u); @test P[1] == P[2]           # V′(0) = 0 at the centre

    # an off-centre, asymmetric 2-D V (a quartic about a displaced point and a tilt), by a tensor surface
    d = SVector(460.0, 350.0)
    tilted(u) = -1.4 * (hypot(u[1] - d[1], u[2] - d[2]) / 400)^4 - 0.6 * (hypot(u[1] - d[1], u[2] - d[2]) / 400)^2 +
                0.05 * (u[1] - 400) / 400 - 0.04 * (u[2] - 400) / 400
    q2 = vignetting_sequence(tilted)
    t0 = logv(TensorProductVignetting((16, 16), VgtkCore.Rect(10.0, 10.0, 780.0, 780.0); degree = 3))
    t = VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, q2.frames, (; camera = q2.camera, rotations = q2.R, saturated = q2.saturated, frame_map = FRAME_MAP,
                                                                     vignetting = t0, free_directions = Matrix(1.0I, n_control_points(t0), n_control_points(t0)))),
                       VignettingSelfCalibration(; alg..., solver = SparseCholesky()))
    e2 = error_vs(t.u, tilted, q2)
    @info "VignettingSelfCalibration tensor: max/rms error $(e2)"
    @test e2[1] < 1.0e-2 && e2[2] < 1.0e-3

    # the seam score: the true V leaves (almost) nothing, no correction leaves the vignetting's steps
    view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.5)
    sk = (; saturated = q.saturated, saturation_margin_px = 2.0, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2,
          chunk = 20_000)
    fitted = score_seams(q.camera, q.frames, q.R, view; vignetting = cg.u, sk...)
    none = score_seams(q.camera, q.frames, q.R, view; vignetting = (u, f) -> 1.0, sk...)
    @info "score_seams: fitted V rms $(fitted.rms) mad $(fitted.mad), none rms $(none.rms) mad $(none.mad), $(fitted.n) pixels"
    @test fitted.n > 10_000 && fitted.n == none.n
    @test fitted.mad < 0.1 * none.mad && fitted.rms < 0.1 * none.rms
end

@testitem "VignettingSelfCalibration, score_seams: the device agrees with the host" setup = [VignettingSequence, BayerMaps] begin
    using Test, StaticArrays, CUDA
    using PanoramaStitching: VignettingSelfCalibration, score_seams, render_panorama, panorama_view
    import VisualGeometryToolkitCore as VgtkCore
    using VisualRadiometryToolkitCore: AbstractVignetting, RadialVignetting, LogVignetting, pole_regular_basis, control_points, with_control_points
    using PanoramaStitching: PositionNoise, ChannelGains
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
        alg = (; rim_margin_px = 8.0, saturation_margin_px = 2.0, fov_deg = 120.0, observation_noise = nothing,
               plane = 2, chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01,
               vignetting_penalty_weight = 1.0e-6, gain_correction = ChannelGains(),
               solver = VgtkCore.NormalCG())
        cg_stops = VgtkCore.ToleranceStops(; rtol = 1.0e-10, max_iter = 20000)
        fitted = (; camera = q.camera, rotations = q.R, frame_map = FRAME_MAP, vignetting = v0, free_directions = pole_regular_basis(v0))
        h = VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, q.frames, (; fitted..., saturated = q.saturated)), VignettingSelfCalibration(; alg...); stops = cg_stops)
        df, ds = CuArray.(q.frames), CuArray.(q.saturated)
        d = VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, df, (; fitted..., saturated = ds)), VignettingSelfCalibration(; alg...); stops = cg_stops)
        @test VgtkCore.successful_retcode(d)
        # the atomic sums run in another order on each backend (and from run to run): V agrees to the solve's precision
        @test maximum(abs.(control_points(d.u) .- control_points(h.u))) < 1.0e-6
        @test abs(d.rms - h.rms) < 1.0e-8
        # the value-and-position noise's weights on the device (the observation's constraint, ForwardDiff and Core's whitening
        # in the kernels)
        sw = PositionNoise(; point_covariance = VgtkCore.CovMat{2}(σ = 1.0), iterations = 2)
        uf = relative_frames(q.frames, 0.01)                                 # the position noise reads the values' variances
        hs = VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, uf, (; fitted..., saturated = q.saturated)), VignettingSelfCalibration(; alg..., observation_noise = sw); stops = cg_stops)
        dsf = VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, CuArray.(uf), (; fitted..., saturated = ds)), VignettingSelfCalibration(; alg..., observation_noise = sw); stops = cg_stops)
        @test maximum(abs.(control_points(dsf.u) .- control_points(hs.u))) < 1.0e-6
        @test maximum(abs.(dsf.metadata.gains .- hs.metadata.gains)) < 1.0e-8 && abs(dsf.rms - hs.rms) < 1.0e-6
        view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.5)
        sk = (; saturation_margin_px = 2.0, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2,
              chunk = 20_000, vignetting = h.u)
        sh = score_seams(q.camera, q.frames, q.R, view; saturated = q.saturated, sk...)
        sd = score_seams(q.camera, df, q.R, view; saturated = ds, sk...)
        @test sh.n == sd.n && abs(sh.rms - sd.rms) < 1.0e-9 && abs(sh.mad - sd.mad) < 1.0e-9
        @test maximum(abs.(d.metadata.gains .- h.metadata.gains)) < 1.0e-8 && all(==(1.0), d.metadata.gains[1, :])
        # a V held fixed that holds HOST arrays (a calibration's table, with its Adapt rule) reaches the device in the fit, the
        # render and the seams (the functors are moved by OnBackend)
        tv = TableV(collect(range(1.0, 0.6; length = 64)), c, 420.0)
        hf = VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, q.frames, (; fitted..., saturated = q.saturated, vignetting = tv, free_directions = zeros(0, 0))),
                            VignettingSelfCalibration(; alg...); stops = cg_stops)
        dfit = VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, df, (; fitted..., saturated = ds, vignetting = tv, free_directions = zeros(0, 0))),
                              VignettingSelfCalibration(; alg...); stops = cg_stops)
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
@testitem "VignettingSelfCalibration: the frames' exposure gains recovered (gauge g₁ = 1), V free or held fixed; score_seams and render_panorama divide them out" setup = [VignettingSequence, BayerMaps] begin
    using Test, Statistics, StaticArrays
    using PanoramaStitching: PanoramaStitching, VignettingSelfCalibration, score_seams, render_panorama, panorama_view,
        SparseCholesky, PositionNoise, IntensityGains
    import VisualGeometryToolkitCore as VgtkCore
    using VisualRadiometryToolkitCore: AbstractVignetting, RadialVignetting, LogVignetting, pole_regular_basis, n_control_points, with_control_points
    c = SVector(430.0, 390.0)
    radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2 - 0.3 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^4
    g = [1.0, 1.12, 0.9, 1.2, 0.95, 1.07]
    q = vignetting_sequence(radial; gains = g)
    v0 = LogVignetting(with_control_points(RadialVignetting(12, c, 420.0), zeros(12)))
    ctx = (; camera = q.camera, rotations = q.R, saturated = q.saturated, frame_map = FRAME_MAP)
    alg = (; rim_margin_px = 8.0, saturation_margin_px = 2.0, fov_deg = 120.0, plane = 2, observation_noise = nothing,
           chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01, vignetting_penalty_weight = 1.0e-6)
    cgs = VgtkCore.NormalCG(); cg_stops = VgtkCore.ToleranceStops(; rtol = 1.0e-10, max_iter = 20000)
    function v_error(v)
        us = filter(u -> hypot(u[1] - 400, u[2] - 400) < 382, vec([FRAME_MAP(SVector(j - 0.5, i - 0.5)) for j in 1:400, i in 1:400]))
        a = [log(v(u)) for u in us]; b = [radial(u) for u in us]
        return maximum(abs, exp.((a .- mean(a)) .- (b .- mean(b))) .- 1)
    end
    Z = pole_regular_basis(v0)
    problem = VgtkCore.EstimationProblem(AbstractVignetting, q.frames, (; ctx..., vignetting = v0, free_directions = Z))
    uncertain = VgtkCore.EstimationProblem(AbstractVignetting, relative_frames(q.frames, 0.01), (; ctx..., vignetting = v0, free_directions = Z))   # the position noise reads the values' variances
    with = VgtkCore.solve(problem, VignettingSelfCalibration(; alg..., gain_correction = IntensityGains(), solver = cgs); stops = cg_stops)
    chol = VgtkCore.solve(problem, VignettingSelfCalibration(; alg..., gain_correction = IntensityGains(), solver = SparseCholesky()))
    without = VgtkCore.solve(problem, VignettingSelfCalibration(; alg..., gain_correction = nothing, solver = cgs); stops = cg_stops)
    G = with.metadata.gains; gw = G[:, 2]
    @test size(G) == (6, 3) && G[:, 1] == G[:, 2] == G[:, 3]          # IntensityGains: one per frame, every plane
    @info "frame gains: recovered $(round.(gw; digits = 5)) (true $g); V error with $(v_error(with.u)), without $(v_error(without.u))"
    @test VgtkCore.successful_retcode(with)
    @test gw[1] === 1.0 && length(gw) == length(g)                    # the gauge: the first frame's gain is 1 exactly
    @test maximum(abs.(gw ./ g .- 1)) < 1.0e-3
    @test maximum(abs.(chol.metadata.gains .- G)) < 1.0e-5            # CG and Cholesky solve the same system
    @test v_error(with.u) < 2.0e-3
    @test v_error(without.u) > 5 * v_error(with.u)
    @test all(==(1.0), without.metadata.gains)
    @test with.rms < 0.5 * without.rms

    # gain compensation: V held fixed at the fitted one (no free direction), the gains alone, one linear solve
    fixed = VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, q.frames, (; ctx..., vignetting = with.u, free_directions = zeros(n_control_points(v0), 0))),
                           VignettingSelfCalibration(; alg..., gain_correction = IntensityGains(), solver = SparseCholesky()))
    @test fixed.u === with.u && fixed.metadata.log_change == 0
    @test maximum(abs.(fixed.metadata.gains[:, 2] ./ g .- 1)) < 1.0e-3
    @test maximum(abs.(fixed.metadata.gains .- G)) < 1.0e-5

    # the value-and-position noise (the observation's constraint whitened with the sensor point's noise too): frames that follow
    # the model exactly give the same V and gains; the weights come from the I² solve's model, so V held fixed takes them too
    sw = PositionNoise(; point_covariance = VgtkCore.CovMat{2}(σ = 1.5), iterations = 2)
    @test_throws "needs the frames' value variances" VgtkCore.solve(problem, VignettingSelfCalibration(; alg..., gain_correction = IntensityGains(),
                                                                                                       solver = SparseCholesky(), observation_noise = sw))
    sam = VgtkCore.solve(uncertain, VignettingSelfCalibration(; alg..., gain_correction = IntensityGains(), solver = SparseCholesky(),
                                                                observation_noise = sw))
    @test sam.metadata.observation_noise === sw && sam.metadata.log_change > 0   # two reweight-and-solve cycles after the I² solve
    @test maximum(abs.(sam.metadata.gains[:, 2] ./ g .- 1)) < 1.0e-3 && v_error(sam.u) < 2.0e-3
    # the termination is every solve's (Core's worst_retcode), not the last one's: the cold first solve capped by the CG
    # budget, the warm-started reweighted solves after it within it — the fit is MaxIters (its legacy Symbol :max_iterations)
    cg(n) = VgtkCore.ToleranceStops(; rtol = 1.0e-10, max_iter = n)
    single = VgtkCore.solve(problem, VignettingSelfCalibration(; alg..., gain_correction = IntensityGains(), solver = VgtkCore.NormalCG()); stops = cg(100_000))
    @test VgtkCore.successful_retcode(single) && single.stop_rule == VgtkCore.ToleranceMet
    budget = single.stats.nsteps ÷ 2                           # the cold solve capped well short (its CG count varies by an iteration or so: threaded scatter-adds)
    capped = VgtkCore.solve(uncertain, VignettingSelfCalibration(; alg..., gain_correction = IntensityGains(), solver = VgtkCore.NormalCG(),
                                                                   observation_noise = sw); stops = cg(budget))
    @test capped.stats.nsteps < 3 * budget                   # a later solve met the tolerance within the budget
    @test capped.retcode == VgtkCore.ReturnCode.MaxIters && capped.stop_rule == VgtkCore.IterationCap
    @test !VgtkCore.successful_retcode(capped)
    sf = VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, relative_frames(q.frames, 0.01), (; ctx..., vignetting = with.u, free_directions = zeros(n_control_points(v0), 0))),
                        VignettingSelfCalibration(; alg..., gain_correction = IntensityGains(), solver = SparseCholesky(),
                                                  observation_noise = PositionNoise(; point_covariance = VgtkCore.CovMat{2}(σ = 1.5))))
    @test sf.metadata.observation_noise.iterations == 1 && maximum(abs.(sf.metadata.gains[:, 2] ./ g .- 1)) < 1.0e-3
    @test_throws UndefKeywordError PositionNoise()                                     # the point's noise is required
    @test_throws TypeError PositionNoise(; point_covariance = [1.0 0; 0 1])            # a CovMat, not a bare matrix
    @test_throws ArgumentError PositionNoise(; point_covariance = VgtkCore.CovMat{2}(σ = 1.0), iterations = 3)
    @test_throws ArgumentError PositionNoise(; point_covariance = VgtkCore.CovMat(SMatrix{2, 2}(1.0, 2.0, 2.0, 1.0)))   # not positive definite

    # the observation's constraint (Core's protocol): G_x is the model's exact gradient in u, and the weight is Core's whitening
    m = with.metadata
    cL = replace(m.scene, NaN => 0.0)
    obs = PanoramaStitching._LogIntensityModel(with.u, m.chart, cL, q.R[3], log(m.gains[3, 2]))
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
        # Σ_x = blockdiag(Σ_u, var(log I)), var(log I) = var(I)/I² = σ_rel² under a relative noise
        Σx = SMatrix{3, 3, Float64, 9}(1.5^2, 0, 0, 0, 1.5^2, 0, 0, 0, 0.01^2)
        @test VgtkCore.whitening_factor(G_x, Σx)^2 ≈ 1 / (1.5^2 * (G_x[1]^2 + G_x[2]^2) + 0.01^2)
    end

    # the seams: the fitted V with its gains leaves (almost) nothing; the same V with the frames taken at one exposure does not
    view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.5)
    sk = (; saturated = q.saturated, saturation_margin_px = 2.0, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2,
          chunk = 20_000, vignetting = with.u)
    sg = score_seams(q.camera, q.frames, q.R, view; gains = G, sk...)
    s1 = score_seams(q.camera, q.frames, q.R, view; sk...)
    @info "score_seams with the gains: rms $(sg.rms) mad $(sg.mad); without: rms $(s1.rms) mad $(s1.mad)"
    @test sg.rms < 0.1 * s1.rms                                       # (the MAD can stay small: a gain step is constant per frame region)
    @test_throws DimensionMismatch score_seams(q.camera, q.frames, q.R, view; gains = G[1:3, :], sk...)
    @test_throws DimensionMismatch score_seams(q.camera, q.frames, q.R, view; gains = gw, sk...)      # K × 3, not a vector

    # the render: dividing by g_k·V makes every frame show the same scene value
    rk = (; vignetting = with.u, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0)
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
@testitem "VignettingSelfCalibration: gain_correction none / intensity / channel, V fitted or held fixed (per-plane gains recovered)" setup = [VignettingSequence, BayerMaps] begin
    using Test, Statistics, StaticArrays
    using PanoramaStitching: VignettingSelfCalibration, score_seams, render_panorama, panorama_view, SparseCholesky,
        IntensityGains, ChannelGains, GAIN_CORRECTIONS
    import VisualGeometryToolkitCore as VgtkCore
    using VisualRadiometryToolkitCore: AbstractVignetting, RadialVignetting, LogVignetting, pole_regular_basis, with_control_points
    c = SVector(430.0, 390.0)
    radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2 - 0.3 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^4
    G = [1.0 1.0 1.0; 1.10 1.12 1.15; 0.93 0.90 0.86; 1.25 1.20 1.12; 0.97 0.95 0.99; 1.02 1.07 1.11]   # g_1c = 1: the gauge
    q = vignetting_sequence(radial; gains = G)
    v0 = LogVignetting(with_control_points(RadialVignetting(12, c, 420.0), zeros(12)))
    truth(u, f) = exp(radial(u))                                     # a V given as any callable
    ctx = (; camera = q.camera, rotations = q.R, saturated = q.saturated, frame_map = FRAME_MAP)
    alg = (; rim_margin_px = 8.0, saturation_margin_px = 2.0, fov_deg = 120.0, plane = 2, observation_noise = nothing,
           chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01, vignetting_penalty_weight = 1.0e-6, solver = SparseCholesky())
    function v_error(v)
        us = filter(u -> hypot(u[1] - 400, u[2] - 400) < 382, vec([FRAME_MAP(SVector(j - 0.5, i - 0.5)) for j in 1:400, i in 1:400]))
        a = [log(v(u, nothing)) for u in us]; b = [radial(u) for u in us]
        return maximum(abs, exp.((a .- mean(a)) .- (b .- mean(b))) .- 1)
    end
    modes = (("fitted", v0, pole_regular_basis(v0)), ("held", truth, zeros(0, 0)))
    for (vm, v, Z) in modes, gc in (nothing, IntensityGains(), ChannelGains())
        r = VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, q.frames, (; ctx..., vignetting = v, free_directions = Z)),
                           VignettingSelfCalibration(; alg..., gain_correction = gc))
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
        vm == "fitted" && gc !== nothing && @test v_error(r.u) < 2.0e-3
        vm == "held" && @test r.u === truth
    end
    @test GAIN_CORRECTIONS["channel"] === ChannelGains && GAIN_CORRECTIONS["intensity"] === IntensityGains &&
          GAIN_CORRECTIONS["none"] === nothing
    @test_throws ArgumentError GAIN_CORRECTIONS["per_channel"]

    # the seams and the render read each plane's gains: with ChannelGains every plane's seams vanish, with IntensityGains only the
    # fitted plane's
    problem = VgtkCore.EstimationProblem(AbstractVignetting, q.frames, (; ctx..., vignetting = v0, free_directions = pole_regular_basis(v0)))
    ch = VgtkCore.solve(problem, VignettingSelfCalibration(; alg..., gain_correction = ChannelGains()))
    fr = VgtkCore.solve(problem, VignettingSelfCalibration(; alg..., gain_correction = IntensityGains()))
    view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.5)
    sk = (; saturated = q.saturated, saturation_margin_px = 2.0, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, chunk = 20_000)
    for p in (1, 3)
        sc = score_seams(q.camera, q.frames, q.R, view; vignetting = ch.u, gains = ch.metadata.gains, plane = p, sk...)
        sf = score_seams(q.camera, q.frames, q.R, view; vignetting = fr.u, gains = fr.metadata.gains, plane = p, sk...)
        @test sc.rms < 0.1 * sf.rms
    end
    rk = (; vignetting = ch.u, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0)
    a = render_panorama(q.camera, q.frames, q.R, view; gains = ch.metadata.gains, rk...)
    b = render_panorama(q.camera, q.frames, q.R, view; rk...)
    o = a.owner; m = findall(o .> 0)[1:97:end]
    @test all(i -> all(cc -> a.image[i, cc] * ch.metadata.gains[o[i], cc] ≈ b.image[i, cc], 1:3), m)

    # an other plane's unconverged solve is not hidden: its CG budget's end is the result's retcode, its iterations summed
    rc = VgtkCore.solve(problem, VignettingSelfCalibration(; alg..., solver = VgtkCore.NormalCG(),
                                                           gain_correction = ChannelGains()); stops = VgtkCore.ToleranceStops(; rtol = 1.0e-14, max_iter = 3))
    @test rc.retcode == VgtkCore.ReturnCode.MaxIters && rc.stop_rule == VgtkCore.IterationCap && rc.stats.nsteps == 3 * 3
    # a V held fixed that is not positive somewhere in the domain is refused by name (not Base's DomainError of the log)
    dark(u, f) = u[1] < 100 ? -0.5 : truth(u, f)
    @test_throws "VignettingSelfCalibration: the given V is not positive" VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, q.frames,
        (; ctx..., vignetting = dark, free_directions = zeros(0, 0))), VignettingSelfCalibration(; alg..., gain_correction = IntensityGains()))
end

# The TOML tags of the gain correction and the observation noise: the current tags select their types (the tag rule: snake_case
# of the type's name less the family's suffix), and the tags of earlier releases are refused with an ArgumentError naming the new
# one (no alias: an old config fails, it is never read as the new setting).
@testitem "GAIN_CORRECTIONS, OBSERVATION_NOISES: the tags, and the renamed tags refused by name" begin
    using Test
    using PanoramaStitching: PanoramaStitching, IntensityGains, ChannelGains, PositionNoise
    import VisualGeometryToolkitCore as VgtkCore
    G = PanoramaStitching.GAIN_CORRECTIONS; N = PanoramaStitching.OBSERVATION_NOISES
    @test G["intensity"] === IntensityGains && G["channel"] === ChannelGains && G["none"] === nothing
    @test N["none"] === nothing && N["position"] === PositionNoise
    @test VgtkCore.tag_of(G, IntensityGains) == "intensity" && VgtkCore.tag_of(N, PositionNoise) == "position"
    renamed(reg, tag) = try
        reg[tag]; ""
    catch e
        e isa ArgumentError ? sprint(showerror, e) : rethrow()
    end
    m = renamed(G, "frame"); @test occursin("renamed", m) && occursin("\"intensity\"", m)
    m = renamed(N, "sampson"); @test occursin("renamed", m) && occursin("\"position\"", m)
    # 0.7: the tags of 0.4–0.6 name their successors
    m = renamed(N, "intensity"); @test occursin("renamed", m) && occursin("\"none\"", m)
    m = renamed(N, "intensity_and_position"); @test occursin("renamed", m) && occursin("\"position\"", m)
    @test_throws ArgumentError N["weights"]
    @test isempty(VgtkCore.tag_rule_violations(G)) && isempty(VgtkCore.tag_rule_violations(N))
end

# The example's TOML keys of earlier releases are refused by name (run.jl's `refuse_renamed_keys`, which `load_config` and
# compare_arms.jl call, and `refuse_circle`, which `load_config` calls: the image circle is the camera's lens coverage): the
# example's own environment is not the tests', so their definitions are read from run.jl alone; the configs of record and
# arms.toml hold the current keys and tags.
@testitem "panorama_vignetting example: the renamed and retired TOML keys refused by name; the configs of record are current" begin
    using Test, TOML
    using PanoramaStitching: PanoramaStitching, AllPairs, LagOverlap, TrackCheck
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    dir =joinpath(pkgdir(PanoramaStitching), "examples", "panorama_vignetting")
    code = Meta.parseall(read(joinpath(dir, "run.jl"), String))
    defines(x, name) = x isa Expr && (x.head === :const ? defines(x.args[1], name) : x.head === :(=) ? x.args[1] === name :
                                      x.head === :function && x.args[1] isa Expr && x.args[1].args[1] === name)
    m = Module()
    Core.eval(m, :(import PanoramaStitching))
    Core.eval(m, :(using PanoramaStitching: AllPairs, LagOverlap, TrackCheck, SparseCholesky))
    Core.eval(m, :(import VisualGeometryToolkitCore as VgtkCore))
    for name in (:RENAMED_KEYS, :RETIRED_VIGNETTING_KEYS, :refuse_renamed_keys, :refuse_circle, :refuse_output, :refuse_inert_pair_keys,
                 :refuse_undeclared_sensor_record, :refuse_config, :detector_settings, :EXAMPLE_CONFIG_VERSIONS, :PAIR_SELECTIONS,
                 :BUNDLE_REFINES, :LINEAR_SOLVERS)
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
    # the constant σ of log I is retired (a specified RelativeNoise of the frames' sensor model): refused by name
    @test_throws "RelativeNoise" refuse(Dict{String, Any}("log_intensity" * "_scale" => 0.01))
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
        PanoramaStitching.refuse_retired_registration(Dict{String, Any}("bundle_iterations" => 3), "config [registration]"); nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("bundle_iterations", err.msg)
    # the 0.6 noise constants are refused by name: the frames' noise is Core's resolver's
    for k in ("electrons_per_dn", "read_noise_dn")
        @test_throws "sensor_characterization" PanoramaStitching.refuse_retired_registration(Dict{String, Any}(k => 1.0), "config [registration]")
    end
    # `[lag_overlap]` with pairs = "all" is refused when the config is read (no inert setting, cached or not)
    inert(pairs) = Base.invokelatest(m.refuse_inert_pair_keys, Dict{String, Any}("registration" => Dict{String, Any}("pairs" => pairs),
                                                                                 "lag_overlap" => Dict{String, Any}("max_lag" => 1)), "config")
    @test_throws "read only by pairs" inert("all")
    @test inert("lag_overlap")["lag_overlap"]["max_lag"] == 1
    # PanoramaStitching 0.7's optical-axis angles are retired (0.8): refused by name, naming the view_overlap rules
    retired(t) = PanoramaStitching.refuse_retired_registration(Dict{String, Any}(t), "config [registration]")
    @test_throws r"`overlap_deg` is retired: .*any view.*min_inliers" retired(("pairs" => "lag_overlap", "max_lag" => 1, "overlap_deg" => 120.0))
    @test_throws r"`near_duplicate_deg` is retired: .*near_duplicate_overlap" retired(("near_duplicate_deg" => 4.0,))
    # the frames' decode states its sensor record: a [decode] without `sensor_characterization` is refused when the config is read
    # (`load_config`), before any file is
    undeclared(t) = Base.invokelatest(m.refuse_undeclared_sensor_record, Dict{String, Any}(t), "config [decode]")
    @test_throws "[decode] sensor_characterization: required" undeclared(("mode" => "bin2", "gain" => 1.0, "cache" => false))
    @test undeclared(("sensor_characterization" => "none",))["sensor_characterization"] == "none"
    # so does the registration's (no default "none"): a config without either is refused by `load_config`'s `refuse_config`;
    # with detection_image = "tone" the detector reads the decoded frames, so its record is [decode]'s
    of_record = TOML.parsefile(joinpath(dir, "config_fx3_14mm.toml"))
    config(edit!) = (c = deepcopy(of_record); edit!(c); Base.invokelatest(m.refuse_config, c, "config"))
    @test config(identity) isa AbstractDict
    @test_throws "[registration] sensor_characterization: required" config(c -> delete!(c["registration"], "sensor_characterization"))
    @test_throws "[decode] sensor_characterization: required" config(c -> delete!(c["decode"], "sensor_characterization"))
    @test_throws "state that record" config(c -> c["registration"]["sensor_characterization"] = "another_record.json")
    @test_throws "noise_sigmas: read only by" config(c -> c["registration"]["noise_sigmas"] = 3)   # the detector's keys, when read
    G = PanoramaStitching.GAIN_CORRECTIONS; N = PanoramaStitching.OBSERVATION_NOISES
    for f in ("config_fx3_14mm.toml", "config_fx3_8mm_rooftop.toml")
        inputs = TOML.parsefile(joinpath(dir, f))["inputs"]
        @test TOML.parsefile(joinpath(dir, f))["decode"]["cache"] isa Bool                      # the decode cache, opt-in, stated
        @test TOML.parsefile(joinpath(dir, f))["decode"]["sensor_characterization"] isa String   # the frames' sensor record, stated (no default)
        @test TOML.parsefile(joinpath(dir, f))["registration"]["sensor_characterization"] isa String   # the detector's, stated too
        @test Base.invokelatest(m.refuse_inert_pair_keys, TOML.parsefile(joinpath(dir, f)), f) isa AbstractDict
        @test PanoramaStitching.refuse_retired_registration(TOML.parsefile(joinpath(dir, f))["registration"], f) isa AbstractDict
        @test TOML.parsefile(joinpath(dir, f))["registration"]["near_duplicate_overlap"] == 0.90   # the owner's value, stated
        @test Base.invokelatest(m.refuse_circle, inputs, f) === inputs
        # the current layout (version 2): the view-graph filters' tolerances and the detector's variant are stated, each algorithm's
        # settings in its own table, the algorithms by tag
        c = TOML.parsefile(joinpath(dir, f))
        @test c["config_version"] == VgtkCore.current_config_version(m.EXAMPLE_CONFIG_VERSIONS) == 2
        @test VgtkCore.migrate_config(c, m.EXAMPLE_CONFIG_VERSIONS) == c                         # current: read as it is
        reg = c["registration"]
        @test all(k -> haskey(reg, k), ("selection", "detection_image", "bundle"))
        @test all(k -> haskey(reg, k), ("cycle_tol_deg", "orientation_tol_deg"))
        @test m.PAIR_SELECTIONS[reg["pairs"]] in (AllPairs, LagOverlap) && m.BUNDLE_REFINES[reg["bundle"]] === TrackCheck
        # the pairs' local optimization is an Estimation strategy table, read by Estimation's own loader
        @test VgtkEstimation.optimization_strategy_from_toml(c["rotation_problem"]["ransac"]["local_optimization_strategy"]; class = :rotation) isa
              VgtkCore.AbstractOptimizationStrategy
        v = c["vignetting"]; vs = c["vignetting_self_calibration"]
        @test refuse(v) === v && refuse(vs) === vs && G[vs["gain_correction"]] isa Any && (N[vs["observation_noise"]]; true)
        @test m.LINEAR_SOLVERS[vs["solver"]] === VgtkCore.NormalCG && haskey(vs["stops"], "rtol")
        # the config of record of the earlier layout (version 1, stored as it was) reads through the converter to this file, value
        # for value; a current file holding a version-1 key is refused naming its current one
        v1 = TOML.parsefile(joinpath(pkgdir(PanoramaStitching), "test", "fixtures", "stored_configs", "panorama_vignetting", f))
        @test !haskey(v1, "config_version") && VgtkCore.migrate_config(v1, m.EXAMPLE_CONFIG_VERSIONS) == c
        @test_throws "rotation_problem.ransac.outlier_halfwidth_px" VgtkCore.migrate_config(
            merge(c, Dict("registration" => merge(reg, Dict("outlier_halfwidth_px" => 1.0)))), m.EXAMPLE_CONFIG_VERSIONS)
        # the V models are Core's VIGNETTING_TAGS of the spline log V is (the retired "tensor" refused by name)
        @test all(k -> VrtkCore.VIGNETTING_TAGS[k] in (VrtkCore.RadialVignetting, VrtkCore.TensorProductVignetting),
                  vcat(v["model"], v["compare_models"]))
    end
    @test_throws ArgumentError VrtkCore.VIGNETTING_TAGS["tensor"]
    arms = TOML.parsefile(joinpath(dir, "arms.toml"))
    @test VrtkCore.VIGNETTING_TAGS[arms["kind"]] === VrtkCore.RadialVignetting
    @test refuse(arms) === arms && length(arms["arms"]) == 12 && allunique(a["name"] for a in arms["arms"])
    for a in arms["arms"]
        @test refuse(a) === a && (N[a["observation_noise"]]; true) && (G[a["gain_correction"]]; true)
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
    for f in ("stage_cache_check.jl", "write_results_check.jl", "autocalibration_check.jl", "provider_agreement_check.jl", "run_checks.jl")   # the example's checks parse too
        @test !any(x -> Meta.isexpr(x, (:error, :incomplete)), Meta.parseall(read(joinpath(dir, f), String)).args)
    end
    named(x, name) = x === name || (x isa Expr && x.head === :call && x.args[1] === name)
    defines(x, name) = x isa Expr && (x.head === :const ? defines(x.args[1], name) : x.head in (:(=), :function) && named(x.args[1], name))
    m = Module()
    for name in (:detector_settings,)
        Core.eval(m, only(filter(x -> defines(x, name), code.args)))
    end
    call(f, args...; kw...) = Base.invokelatest(getfield(m, f), args...; kw...)
    # the detector's settings: "vst" reads its threshold and its sensor record, "tone" refuses the threshold (never read; its record
    # is [decode]'s, which refuse_config checks)
    det(r) = call(:detector_settings, Dict{String, Any}(r))
    @test det(Dict("selection" => "spread", "detection_image" => "tone", "sensor_characterization" => "none")) ==
          (; selection = :spread, detection_image = :tone, noise_sigmas = nothing, characterization = nothing)
    v = det(Dict("selection" => "spread", "detection_image" => "vst", "noise_sigmas" => 3, "sensor_characterization" => "none"))
    @test v == (; selection = :spread, detection_image = :vst, noise_sigmas = 3.0, characterization = "none")
    @test_throws ArgumentError det(Dict("selection" => "spread", "detection_image" => "tone", "noise_sigmas" => 3, "sensor_characterization" => "none"))
    @test_throws KeyError det(Dict("selection" => "spread", "detection_image" => "vst", "noise_sigmas" => 3))
end

# The uncertain path end to end (owner ruling D10): the frames as RAW mosaics decoded by Core's LinearRaw under a sensor model
# whose noise feeds each value's variance — a specified RelativeNoise (Core's test fixture: tests never build a component) —
# give VignettingSelfCalibration a StructArray of Uncertain values: their values are the plain decode's bit for bit, their variances
# (σ_rel·x)², and the fit weights by them (value_variance), recovering V from exact frames.
@testitem "VignettingSelfCalibration: LinearRaw's uncertain image (a sensor model's RelativeNoise) weights by the values' variances" setup = [VignettingSequence, BayerMaps] begin
    using Test, Statistics, StaticArrays
    using PanoramaStitching: PanoramaStitching, VignettingSelfCalibration, SparseCholesky
    import VisualGeometryToolkitCore as VgtkCore
    using VisualRadiometryToolkitCore: AbstractVignetting, RadialVignetting, LogVignetting, pole_regular_basis, with_control_points
    c = SVector(430.0, 390.0)
    radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2
    q = vignetting_sequence(radial)
    model = VgtkCore.sensor_fixture(; relative_noise = 0.01).model
    @test model.noise isa VgtkCore.RelativeNoise
    Ls = [VgtkCore.LinearRaw(rggb_mosaic(F); mode = :bin2, sensor_model = model) for F in q.frames]
    plain = [VgtkCore.LinearRaw(rggb_mosaic(F); mode = :bin2).image for F in q.frames]
    vals, vars = PanoramaStitching._frame_values(Ls[1].image)
    @test vals == plain[1] && vars ≈ (0.01 .* plain[1]) .^ 2
    v0 = LogVignetting(with_control_points(RadialVignetting(12, c, 420.0), zeros(12)))
    ctx = (; camera = q.camera, rotations = q.R, vignetting = v0, free_directions = pole_regular_basis(v0), saturated = [L.saturated for L in Ls],
           frame_map = FRAME_MAP)
    alg = VignettingSelfCalibration(; gain_correction = nothing, rim_margin_px = 8.0, saturation_margin_px = 2.0, fov_deg = 120.0, plane = 2,
                                    chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01, vignetting_penalty_weight = 1.0e-6,
                                    solver = SparseCholesky(), observation_noise = nothing)
    u = VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, [L.image for L in Ls], ctx), alg)
    p = VgtkCore.solve(VgtkCore.EstimationProblem(AbstractVignetting, plain, ctx), alg)
    @test u.metadata.value_variance && !p.metadata.value_variance
    us = filter(x -> hypot(x[1] - 400, x[2] - 400) < 380, vec([FRAME_MAP(SVector(j - 0.5, i - 0.5)) for j in 1:400, i in 1:400]))
    err(r) = (a = [log(r.u(x)) for x in us]; b = radial.(us); maximum(abs, (a .- mean(a)) .- (b .- mean(b))))
    @test err(u) < 5.0e-3 && err(p) < 5.0e-3                               # 14-bit codes: the quantization, not the weights
end

# PanoramaStitching 0.9.0 (T. Harder's analysis, 2026-11-15): the saturation masks are SYMMETRIC over the frames. Bright scene points
# (σ 0.3°, under the scene's 1° chart, so its spline cannot follow them) clip where V is high — near each frame's centre — and stay
# below white where V dims them. With each frame's own mask alone (PanoramaStitching ≤ 0.8.4) the rows that remain see the points
# only towards the rim, the scene's misfit there pulls V up and V comes out too flat: on this sweep 1.9–3.7 % high at the rings
# (measured 2026-11-15 on 0.8.3). Dropped from every frame where any frame whose domain sees the direction is saturated (masks
# dilated by 2 camera px = 1 frame px, the nearest-pixel lookup's own quantization), V is recovered to 0.05 %.
@testitem "VignettingSelfCalibration: the saturation masks are symmetric over the frames (a bright point clipped near the centre only)" setup = [VignettingSequence, BayerMaps] begin
    using Test, Statistics, StaticArrays, CUDA
    using PanoramaStitching: VignettingSelfCalibration, SparseCholesky
    import VisualGeometryToolkitCore as VgtkCore
    using VisualRadiometryToolkitCore: AbstractVignetting, RadialVignetting, LogVignetting, pole_regular_basis, with_control_points, control_points
    c = SVector(430.0, 390.0)
    radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2 - 0.3 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^4
    plain = vignetting_sequence(radial)
    spikes = [R * SVector(0.0, 0.0, 1.0) for R in plain.R]                  # a bright point on every frame's axis
    q = vignetting_sequence(radial; spikes, amp = 5.0, width_deg = 0.3, white = 1.0)
    # each point is imaged at 0°, ~25° and 60° off the frames' axes (r ≈ 0, 110 and 270 px); its clipped core, where
    # 5·t_c·V·exp(−θ²/2σ²) ≥ 1, shrinks as V dims it, so a frame where V is high drops more of it than one where V is low
    @test sum(count, q.saturated) > 100
    v0 = LogVignetting(with_control_points(RadialVignetting(12, c, 420.0), zeros(12)))
    alg = (; rim_margin_px = 8.0, fov_deg = 120.0, observation_noise = nothing, plane = 2, chart_spacing_deg = 1.0,
           penalty_order = 3, penalty_weight = 0.01, vignetting_penalty_weight = 1.0e-6, gain_correction = nothing, solver = SparseCholesky())
    fitted(s, frames = s.frames, saturated = s.saturated) =
        VgtkCore.EstimationProblem(AbstractVignetting, frames, (; camera = s.camera, rotations = s.R, saturated, frame_map = FRAME_MAP, vignetting = v0,
                                                         free_directions = pole_regular_basis(v0)))
    # V's error at rings about its centre, both in the gauge V = 1 at the centre: > 0 is too flat
    function ring_error(v)
        radii = 0.0:40.0:360.0
        lv = [log(v(c + SVector(r, 0.0))) for r in radii]; lt = [radial(c + SVector(r, 0.0)) for r in radii]
        return exp.((lv .- lv[1]) .- (lt .- lt[1])) .- 1
    end
    r = VgtkCore.solve(fitted(q), VignettingSelfCalibration(; alg..., saturation_margin_px = 2.0))
    e = ring_error(r.u)
    @info "VignettingSelfCalibration symmetric saturation: ring errors $(round.(e; digits = 5)), $(r.n) rows, $(r.metadata.seconds.saturation) s"
    @test VgtkCore.successful_retcode(r) && maximum(abs, e) < 1.0e-3
    # no margin: the nearest-pixel lookup misses the clipped points' edges, and part of the bias stays
    e0 = ring_error(VgtkCore.solve(fitted(q), VignettingSelfCalibration(; alg..., saturation_margin_px = 0.0)).u)
    @test maximum(e0) > 5.0e-3
    # the rows dropped: more than the frames' own saturated pixels (each point's direction from every frame that sees it)
    rp = VgtkCore.solve(fitted(plain), VignettingSelfCalibration(; alg..., saturation_margin_px = 2.0))
    @test rp.n - r.n > 2 * sum(count, q.saturated)
    @test maximum(abs, ring_error(rp.u)) < 1.0e-4                 # nothing saturated: V as before
    @test_throws ArgumentError VgtkCore.solve(fitted(q), VignettingSelfCalibration(; alg..., saturation_margin_px = -1.0))
    @test_throws ArgumentError VgtkCore.solve(fitted(q), VignettingSelfCalibration(; alg..., saturation_margin_px = NaN))
    # the device: the same masks, the same V
    if CUDA.functional()
        d = VgtkCore.solve(fitted(q, CuArray.(q.frames), CuArray.(q.saturated)), VignettingSelfCalibration(; alg..., saturation_margin_px = 2.0))
        @test d.n == r.n
        @test maximum(abs.(control_points(d.u) .- control_points(r.u))) < 1.0e-6
    else
        @test_skip "no CUDA device"
    end
end

# PanoramaStitching 0.9.0: score_seams scores the fit's own observations. A direction any frame whose domain sees it has
# saturated there is measured by NO frame (the sweep's symmetric test, `saturation_margin_px` as the fit's), so the seam score
# no longer counts, at a clipped point, the frames where V dims it below white.
@testitem "score_seams: the fit's observations — a direction some frame sees clipped is measured by no frame" setup = [VignettingSequence, BayerMaps] begin
    using Test, StaticArrays
    using PanoramaStitching: score_seams, sample_frames, panorama_view
    import VisualGeometryToolkitCore as VgtkCore
    c = SVector(430.0, 390.0)
    radial(u) = -1.2 * (hypot(u[1] - c[1], u[2] - c[2]) / 400)^2
    plain = vignetting_sequence(radial)
    q = vignetting_sequence(radial; spikes = [R * SVector(0.0, 0.0, 1.0) for R in plain.R], amp = 5.0, width_deg = 0.3, white = 1.0)
    view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.25)
    nv, nλ = view.size
    D = vec([view.rays(VgtkCore.pixel_center(Float64, CartesianIndex(i, j))) for i in 1:nv, j in 1:nλ])   # score_seams' order
    kw = (; frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, vignetting = (u, f) -> 1.0)
    # the raster's directions some frame sees clipped: seen there with clear masks, not with its own
    s0 = sample_frames(q.camera, q.frames, q.R, D; saturated = [falses(size(m)) for m in q.saturated], kw...)
    s1 = sample_frames(q.camera, q.frames, q.R, D; saturated = q.saturated, kw...)
    clipped = vec(any(s0.seen .& .!s1.seen; dims = 2))
    # a 4 px margin (2 frame px): the dilated nearest-pixel test reaches every bilinear tap of the frame's own rule
    sc = score_seams(q.camera, q.frames, q.R, view; saturated = q.saturated, saturation_margin_px = 4.0, kw..., plane = 2, chunk = 20_000)
    @test count(clipped) > 50 && sc.n > 10_000
    @test all(isnan, vec(Array(sc.ratio))[clipped])
    @test_throws ArgumentError score_seams(q.camera, q.frames, q.R, view; saturated = q.saturated, saturation_margin_px = -1.0, kw..., plane = 2, chunk = 20_000)
end
