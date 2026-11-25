# =============================================================================
# The autocalibration of a rotating camera (`RotationBundleProblem(tracks; fixed)` refining the guess `(; camera, rotations)`: no
# target, the rotation bundle with the camera free but `fixed`), and the rotation-only bundle that holds the calibration
# (`RotationBundleProblem(camera, tracks)` refining the rotations), one bundle underneath.
# Exact data from a known Kannala–Brandt camera are fitted exactly from a wrong camera and a perturbed start; the reduced
# information (the Schur complement of the directions) shows nothing free but the gauge (the first frame's rotation); the camera block's Jacobian against
# ForwardDiff; the camera held, the bundle's numbers are those it had before the camera could be freed.
# =============================================================================

@testsetup module RotatingKB
using Random, LinearAlgebra, StaticArrays
import Rotations
using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, px, project
export TRUE_CAM, rows_rotations, exact_tracks
# a 1200 × 1200 sensor, its principal point off centre, a KB lens with four coefficients
const TRUE_CAM = CameraModel(300.0px, SA[612.0px, 588.0px]; lens = KannalaBrandt(0.05, -0.01, 0.002, -0.0005))
# two rows of frames (pitch 0 and 0.35 rad), yaw 0.45 rad apart, a little roll: rotations about more than one axis
rows_rotations() = Dict(k => SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotYXZ(0.45 * ((k - 1) % 5), k > 5 ? 0.35 : 0.0, 0.02 * ((k - 1) % 3)))) for k in 1:10)
# tracks of world directions seen within 80° of the axes of ≥ 2 frames, the TRUE camera's exact pixels inside the sensor
function exact_tracks(rng, R, n)
    out = Vector{Vector{Pair{Int, SVector{2, Float64}}}}()
    while length(out) < n
        d = normalize(randn(rng, SVector{3, Float64}))
        t = Pair{Int, SVector{2, Float64}}[]
        for k in sort!(collect(keys(R)))
            c = R[k]' * d
            acos(clamp(c[3], -1, 1)) < deg2rad(80) || continue
            u = project(TRUE_CAM, c)
            (all(isfinite, u) && 0 < u[1] < 1200 && 0 < u[2] < 1200) || continue
            push!(t, k => SVector{2, Float64}(u))
        end
        length(t) >= 2 && push!(out, t)
    end
    return out
end
end

@testitem "the autocalibration: a known KB lens recovered exactly from a rotating sequence; one gauge" setup = [RotatingKB] begin
    using Test, Random, LinearAlgebra, StaticArrays
    import Rotations
    using PanoramaStitching: RotationBundleProblem
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode
    rng = Xoshiro(7)
    R = rows_rotations()
    tracks = exact_tracks(rng, R, 600)
    # the start: an equidistant camera (k = 0), f 6 % short, the principal point at the sensor centre; rotations 1° off
    cam0 = VgtkCore.BoundedCamera(VgtkCore.CameraModel(282.0VgtkCore.px, SA[600.0VgtkCore.px, 600.0VgtkCore.px];
                                                       lens = VgtkCore.KannalaBrandt(0.0, 0.0, 0.0, 0.0)), VgtkCore.Size2(1200, 1200))
    start = Dict(k => (k == 1 ? R[1] : SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec((deg2rad(1) .* normalize(randn(rng, 3)))...))) * R[k]) for k in 1:10)
    free = (:f, :cx, :cy, :k1, :k2, :k3, :k4)                                   # every parameter but the aspect (held) and the skew
    lm = LevenbergMarquardt(; optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}())
    stops = LmStops(; max_iter = 1000, ftol = 1.0e-12, xtol = 1.0e-12)
    autocalibrate(fixed; stops = stops) = solve(RefineProblem(RotationBundleProblem(tracks; fixed), (; camera = cam0, rotations = start)), lm;
                                                width = 1.5, stops)
    res = autocalibrate((:aspect,))
    @test successful_retcode(res)
    # Converged by a tolerance, within the cap; the count of LM iterations is not pinned: it follows how each machine's
    # compiled code rounds (StaticArrays' `muladd` chains, contracted into FMAs as each LLVM and CPU target decides; the
    # held-calibration item below), and GitHub's runner ran other counts than dgx01. The observations are the fixture's.
    @test res.stop_rule != VgtkCore.IterationCap && 0 < res.stats.nsteps < stops.max_iter && res.n == 3624
    # On these exact tracks the rms is rounding alone: each residual is the difference of two roundings of a pixel
    # coordinate below 1200 px (the observation's projection and the fit's), each within eps(1200.0), so the rms is at most
    # 2eps(1200.0). The estimate is checked against the truth below (the camera, the rotations), not recorded sums.
    @test res.rms <= 2eps(1200.0)
    # a group holds its family: the four KB coefficients held keeps them at the start's, f, cx and cy free
    rk = autocalibrate((:aspect, :k); stops = LmStops(; max_iter = 3, ftol = 1.0e-12, xtol = 1.0e-12))
    @test rk.metadata.calibration.free_parameters == (:f, :cx, :cy)
    cam = res.u.camera
    @test cam isa VgtkCore.BoundedCamera && VgtkCore.extent(cam) == VgtkCore.extent(cam0)
    θ = VgtkCore.parameter_vector(VgtkCore.model(cam)); θt = VgtkCore.parameter_vector(TRUE_CAM)
    @test θ ≈ θt rtol = 1.0e-9                                            # every parameter, the held aspect and skew included
    @test res.rms < 1.0e-8                                                # exact data: an exact fit
    @test maximum(k -> norm(SMatrix{3, 3}(res.u.rotations[k]) - R[k]), 1:10) < 1.0e-10
    c = res.metadata.calibration
    @test c.free_parameters == free
    @test collect(values(c.parameters)) ≈ θt[[1, 4, 5, 6, 7, 8, 9]] rtol = 1.0e-9
    # one gauge: with the first frame's rotation held, the reduced information (the inverse of the rotations' and the camera's
    # marginal of the bundle's (JᵀJ)⁻¹) has no null direction (nothing else free); every kept direction is observed
    eg = c.reduced_eigenvalues_gauge_fixed
    @test length(eg) == 3 * 9 + 7
    @test eg[1] > 1.0e-8 * eg[end]
    @test !hasproperty(c, :reduced_eigenvalues)                         # the gauge-included spectrum: PanoramaStitching's Schur copy, deleted
    @test c.direction_min_eigenvalue > 0
    @test c.nu == 2 * res.n - (3 * 9 + 2 * length(tracks) + 7)
end

@testitem "the autocalibration: the camera block's Jacobian; the refusals" setup = [RotatingKB, SweepScene] begin
    using Test, LinearAlgebra, StaticArrays
    import ForwardDiff, NLLSsolver
    import PanoramaStitching
    using PanoramaStitching: RotationBundleProblem
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt, LmStops
    R = rows_rotations()
    free = (:f, :cx, :cy, :k1, :k2, :k3, :k4)
    chart = VgtkCore.restrict(VgtkCore.ParameterManifold(TRUE_CAM), free)
    cc = PanoramaStitching._CameraVariable(chart)            # the camera variable: its chart and its model's valid domain
    @test cc.domain == VgtkCore.valid_domain(TRUE_CAM)
    # an LM update recomputes the domain, of the variable's one type (a fold appearing or vanishing keeps it)
    cu = NLLSsolver.update(cc, [0.0, 0, 0, -0.3, 0, 0, 0])
    @test typeof(cu) === typeof(cc) && cu.domain == VgtkCore.valid_domain(VgtkCore.model(cu.chart))
    d = normalize(SVector(0.4, -0.3, 1.0))
    ch = VgtkCore.SO3Manifold(SMatrix{3, 3, Float64, 9}(R[3])); dv = VgtkCore.Eliminable{false}(VgtkCore.SphereManifold(d))
    x = VgtkCore.project(TRUE_CAM, R[3]' * d) + SVector(0.3, -0.2)
    obs = PanoramaStitching._CameraObservation{false, Nothing, Nothing}(x, SVector(1, 2, 3), nothing, nothing)
    _, J = NLLSsolver.computeresjac(NLLSsolver.static(7), obs, ch, dv, cc)
    # the frame and direction blocks through the variables' updates (the camera at its anchor); the camera block through the
    # chart's update (π without the valid-domain test, which Core evaluates on plain numbers only)
    f = updated_residual(obs, ch, dv, cc)
    @test J[:, 1:5] ≈ ForwardDiff.jacobian(f, zeros(5)) rtol = 1.0e-7
    g(δ) = PanoramaStitching._image(VgtkCore.model(NLLSsolver.update(chart, δ)), R[3]' * d)
    @test J[:, 6:12] ≈ ForwardDiff.jacobian(g, zeros(7)) rtol = 1.0e-9
    @test NLLSsolver.computeresidual(obs, ch, dv, cc) ≈ VgtkCore.project(TRUE_CAM, R[3]' * d) - x
    for (flags, cols) in ((1, 1:3), (2, 4:5), (3, 1:5), (4, 6:12), (5, [1:3; 6:12]), (6, 4:12))
        @test NLLSsolver.computeresjac(flags, obs, ch, dv, cc)[2] ≈ J[:, cols]
    end
    stops = LmStops(; max_iter = 1000, ftol = 1.0e-10, xtol = 1.0e-10)
    tr = [[1 => SVector(600.0, 600.0), 2 => SVector(500.0, 600.0)]]
    cauchy = LevenbergMarquardt(; optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}())
    autocalibration(fixed) = RefineProblem(RotationBundleProblem(tr; fixed), (; camera = TRUE_CAM, rotations = R))
    held = RefineProblem(RotationBundleProblem(TRUE_CAM, tr), R)
    @test_throws ArgumentError solve(autocalibration((:focal,)), cauchy; width = 1.5, stops)      # no such parameter
    @test_throws ArgumentError solve(autocalibration((:f, :f)), cauchy; width = 1.5, stops)       # named twice
    @test_throws r"holds every camera parameter" solve(autocalibration(VgtkCore.parameter_names(TRUE_CAM)), cauchy; width = 1.5, stops)
    # the autocalibration refines its guess's camera: a problem without one is refused on the rotations alone
    @test_throws r"\(; camera, rotations\)" solve(RefineProblem(RotationBundleProblem(tr), R), cauchy; width = 1.5, stops)
    # a keyword the solve does not take (the retired bundle keywords `camera_free`, `loss`, `loss_width` among them) has no method
    @test_throws MethodError solve(held, cauchy; width = 1.5, stops, camera_free = (:f,))
    @test_throws ArgumentError solve(autocalibration(()), LevenbergMarquardt(; optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(efficiency_constant = VgtkCore.efficiency_constant(VgtkCore.CauchyShape(), 2))); stops)   # no scale or width
    @test_throws ArgumentError solve(autocalibration(()), cauchy; scale = 1.0, width = 1.5, stops)                          # both
    @test_throws ArgumentError solve(autocalibration(()), LevenbergMarquardt(; optimization_strategy = nothing); width = 1.5, stops)
    @test_throws ArgumentError solve(held, cauchy; stops)                                  # neither scale nor width
    @test_throws ArgumentError solve(held, cauchy; stops, scale = 1.0, width = 1.5)        # both
    @test_throws ArgumentError solve(held, LevenbergMarquardt(; optimization_strategy = nothing); stops, width = 1.5)   # L2 takes none
    for k in (:bundle_loss, :averaging_loss)
        @test_throws r"optimization_strategy" PanoramaStitching._refuse_retired_registration_keywords("register_rotations", (; k => 1))
    end
    @test_throws r"RotationBundleProblem" PanoramaStitching._refuse_retired_registration_keywords("register_rotations",
                                                                                                       (; bundle_camera_free = ()))
    @test_throws r"bundle_optimization_strategy" PanoramaStitching._refuse_retired_registration_keywords("register_rotations",
                                                                                                     (; bundle_scale_multiple = 3.0))
    @test_throws r"local_optimization_strategy" PanoramaStitching._refuse_retired_registration_keywords("verify_pairs",
                                                                                                      (; local_optimization = nothing))
end

# The camera held (`RotationBundleProblem(camera, tracks)`, the rotation-only bundle): the numbers of the bundle before the camera could be freed — the fisheye
# sweep of rotation_bundle_tests.jl with outliers, its rms and its rotations as PanoramaStitching 69471ff computed them.
@testitem "the rotation bundle holds the calibration: no calibration, the given width, the rotations recovered" setup = [SweepScene, RotatingKB, BundleReference] begin
    using Test, Random, LinearAlgebra, StaticArrays
    import Rotations
    using PanoramaStitching: RotationBundleProblem
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode
    # (69471ff's pinned numbers of this fixture were those of the 3-run bundle that re-estimated its width; the bundle is one
    # run at a given width since the pooled RANSAC scale, so the held camera is checked on the result instead)
    rng = Xoshiro(31)
    R = sweep_rotations(rng, 5)
    ts, dirs, _ = sweep_tracks(rng, R, 400; outl = 0.05)                         # dirs: the tracks' true directions
    start = Dict(k => SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec((0.02 .* randn(rng, 3))...))) * R[k] for k in 1:5)
    start[1] = R[1]
    lm = LevenbergMarquardt(; optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}())
    stops = LmStops(; max_iter = 1000, ftol = 1.0e-12, xtol = 1.0e-10)
    res = solve(RefineProblem(RotationBundleProblem(CAM, [[k => x for (k, x) in t] for t in ts]), start), lm; width = 1.5, stops)
    @test res.prob.f.camera === CAM && res.u isa AbstractDict && res.metadata.calibration === nothing
    @test res.metadata.loss_width == 1.5 && successful_retcode(res) && 0 < res.stats.nsteps < stops.max_iter
    @test res.stop_rule != VgtkCore.IterationCap && res.n == 1489          # by a tolerance; the fixture's observations
    @test maximum(k -> rad2deg(norm(VgtkCore.log_so3(SMatrix{3, 3}(res.u[k])' * R[k]))), 1:5) < 0.05
    # The estimate against the truth, within the error the noise allows — no number recorded on one machine (the iterations
    # and last digits follow how each machine's compiled code rounds; GitHub's runner ran another count than dgx01 on 0.7.0).
    # The observations within 6σ of the truth are the inliers (σ = 0.5 px: a Gaussian residual exceeds 6σ with probability
    # e⁻¹⁸); the Cauchy M-estimator's asymptotic covariance at loss width 1.5 px is κσ²(JᵢₙᵀJᵢₙ)⁻¹ (J the bundle's Jacobian
    # at the truth, BundleReference); the outliers pull it by their first-order bias −H⁻¹g_out. The rotations' error, less
    # that bias, is within χ²₁₂(0.999) in the covariance's metric (12: four free frames).
    σ, c, free = 0.5, 1.5, collect(2:5)
    m = bundle_model(CAM, R, dirs, ts, free)
    nobs = length(m.cols)
    inl = [hypot(m.r[2i - 1], m.r[2i]) <= 6σ for i in 1:nobs]
    # a track with no inlier has no information on its direction, which then absorbs its outliers whole: its columns and its
    # observations are left out (the rotations are the first 3·|free| columns, always kept)
    keep = sort!(unique(reduce(vcat, [m.cols[i] for i in 1:nobs if inl[i]])))
    Jin = m.J[reduce(vcat, [[2i - 1, 2i] for i in 1:nobs if inl[i]]), keep]
    pulled = [i for i in 1:nobs if !inl[i] && m.cols[i][end] in keep]               # the outliers of tracks with an inlier
    (; κ, curvature) = cauchy_efficiency(c, σ)
    F = cholesky(Symmetric(Jin' * Jin))
    bias = -(F \ cauchy_gradient(m, c, pulled)[keep]) ./ curvature
    nR = 3 * length(free)
    CR = Symmetric(κ * σ^2 * (F \ Matrix(1.0I, length(keep), nR))[1:nR, :])          # the rotations' block of κσ²(JᵢₙᵀJᵢₙ)⁻¹
    e = coordinates_between(R, dirs, res.u, res.metadata.directions, free)[1:nR] - bias[1:nR]
    q = chisq_even_quantile(0.999, nR)
    @info "held bundle: the rotations' error in its covariance's metric $(round(dot(e, CR \ e); digits = 2)) (χ²₁₂(0.999) = $(round(q; digits = 2))), $(count(inl)) inliers"
    @test dot(e, CR \ e) <= q
    # and on the KB scene, the camera held at the wrong start camera
    rng = Xoshiro(7)
    Rk = rows_rotations()
    tracks = exact_tracks(rng, Rk, 600)
    cam0 = VgtkCore.BoundedCamera(VgtkCore.CameraModel(282.0VgtkCore.px, SA[600.0VgtkCore.px, 600.0VgtkCore.px];
                                                       lens = VgtkCore.KannalaBrandt(0.0, 0.0, 0.0, 0.0)), VgtkCore.Size2(1200, 1200))
    startk = Dict(k => (k == 1 ? Rk[1] : SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec((deg2rad(1) .* normalize(randn(rng, 3)))...))) * Rk[k]) for k in 1:10)
    stopsk = LmStops(; max_iter = 1000, ftol = 1.0e-12, xtol = 1.0e-12)
    rk = solve(RefineProblem(RotationBundleProblem(cam0, tracks), startk), lm; width = 1.5, stops = stopsk)
    @test rk.prob.f.camera === cam0
    @test successful_retcode(rk) && rk.stop_rule != VgtkCore.IterationCap && 0 < rk.stats.nsteps < stopsk.max_iter && rk.n == 3624
    # The wrong camera cannot fit the exact tracks (the freed camera fits them to 2eps(1200.0), the first item). No minimum is
    # asserted: this robust cost, its residuals far past the loss width, has many shallow minima (three starts 0.3°–1° off
    # stop at rms 13.49–13.63 px; measured on dgx01, 2026-11-14), so no start-independent reference holds.
    @test rk.rms > 2eps(1200.0)
end
