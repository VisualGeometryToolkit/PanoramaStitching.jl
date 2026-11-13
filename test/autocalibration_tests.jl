# =============================================================================
# The autocalibration of a rotating camera (`refine_camera_and_rotations`: no target, the rotation bundle with the camera free
# but `fixed`), and the rotation-only bundle that holds the calibration (`refine_rotations`), one bundle underneath.
# Exact data from a known Kannala–Brandt camera are fitted exactly from a wrong camera and a perturbed start; the reduced
# information (Core's implicit_covariance) shows nothing free but the gauge (the first frame's rotation); the camera block's Jacobian against
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

@testitem "refine_camera_and_rotations: a known KB lens recovered exactly from a rotating sequence; one gauge" setup = [RotatingKB] begin
    using Test, Random, LinearAlgebra, StaticArrays
    import Rotations
    using PanoramaStitching: refine_camera_and_rotations
    import VisualGeometryToolkitCore as VgtkCore
    rng = Xoshiro(7)
    R = rows_rotations()
    tracks = exact_tracks(rng, R, 600)
    # the start: an equidistant camera (k = 0), f 6 % short, the principal point at the sensor centre; rotations 1° off
    cam0 = VgtkCore.BoundedCamera(VgtkCore.CameraModel(282.0VgtkCore.px, SA[600.0VgtkCore.px, 600.0VgtkCore.px];
                                                       lens = VgtkCore.KannalaBrandt(0.0, 0.0, 0.0, 0.0)), VgtkCore.Size2(1200, 1200))
    start = Dict(k => (k == 1 ? R[1] : SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec((deg2rad(1) .* normalize(randn(rng, 3)))...))) * R[k]) for k in 1:10)
    free = (:f, :cx, :cy, :k1, :k2, :k3, :k4)                                   # every parameter but the aspect (held) and the skew
    kw = (; optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(), width = 1.5, max_iter = 1000, ftol = 1.0e-12,
          xtol = 1.0e-12)
    res = refine_camera_and_rotations(cam0, start, tracks; kw..., fixed = (:aspect,))
    @test VgtkCore.converged(res)
    # the same bundle as the camera_free keyword it replaces: PanoramaStitching ce2ce59's `refine_rotations(…; camera_free =
    # free, optimization_strategy = Triggs{CauchyShape}(efficiency_constant = 1.0), width = 1.5)` on this scene. Not bit for
    # bit: StaticArrays' `muladd` chains round as each compiled copy's LLVM contracts them (the held-calibration item below
    # says how), and on these exact tracks the rms is that rounding alone: each residual is the difference of two roundings
    # of a pixel coordinate below 1200 px (the observation's projection and the fit's), each within eps(1200.0), so the rms
    # is at most 2eps(1200.0). The estimate agrees to the stop's rtol = xtol; the counts are exact.
    @test (res.iterations, res.n) == (11, 3624) && res.rms <= 2eps(1200.0)
    @test isapprox(sum(sum, values(VgtkCore.estimate(res))), 19.8664444387427; rtol = kw.xtol) &&
          isapprox(sum(sum, res.metadata.directions), 231.78800370774763; rtol = kw.xtol)
    # a group holds its family: the four KB coefficients held keeps them at the start's, f, cx and cy free
    rk = refine_camera_and_rotations(cam0, start, tracks; kw..., fixed = (:aspect, :k), max_iter = 3)
    @test rk.metadata.calibration.free_parameters == (:f, :cx, :cy)
    cam = res.metadata.camera
    @test cam isa VgtkCore.BoundedCamera && VgtkCore.extent(cam) == VgtkCore.extent(cam0)
    θ = VgtkCore.parameter_vector(VgtkCore.model(cam)); θt = VgtkCore.parameter_vector(TRUE_CAM)
    @test θ ≈ θt rtol = 1.0e-9                                            # every parameter, the held aspect and skew included
    @test res.rms < 1.0e-8                                                # exact data: an exact fit
    @test maximum(k -> norm(SMatrix{3, 3}(VgtkCore.estimate(res)[k]) - R[k]), 1:10) < 1.0e-10
    c = res.metadata.calibration
    @test c.free_parameters == free
    @test collect(values(c.parameters)) ≈ θt[[1, 4, 5, 6, 7, 8, 9]] rtol = 1.0e-9
    # one gauge: with the first frame's rotation held, the reduced information (the inverse of the rotations' and the camera's
    # marginal of Core's implicit_covariance) has no null direction (nothing else free); every kept direction is observed
    eg = c.reduced_eigenvalues_gauge_fixed
    @test length(eg) == 3 * 9 + 7
    @test eg[1] > 1.0e-8 * eg[end]
    @test !hasproperty(c, :reduced_eigenvalues)                         # the gauge-included spectrum: PanoramaStitching's Schur copy, deleted
    @test c.direction_min_eigenvalue > 0
    @test c.nu == 2 * res.n - (3 * 9 + 2 * length(tracks) + 7)
end

@testitem "refine_camera_and_rotations: the camera block's Jacobian; the refusals" setup = [RotatingKB, SweepScene] begin
    using Test, LinearAlgebra, StaticArrays
    import ForwardDiff, NLLSsolver
    import PanoramaStitching
    using PanoramaStitching: refine_rotations, refine_camera_and_rotations
    import VisualGeometryToolkitCore as VgtkCore
    R = rows_rotations()
    free = (:f, :cx, :cy, :k1, :k2, :k3, :k4)
    cc = VgtkCore.restrict(VgtkCore.ParameterManifold(TRUE_CAM), free)
    d = normalize(SVector(0.4, -0.3, 1.0))
    ch = VgtkCore.SO3Manifold(SMatrix{3, 3, Float64, 9}(R[3])); dv = VgtkCore.Eliminable{false}(VgtkCore.SphereManifold(d))
    x = VgtkCore.project(TRUE_CAM, R[3]' * d) + SVector(0.3, -0.2)
    obs = PanoramaStitching._CameraObservation{false, Nothing, Nothing, typeof(cc)}(x, SVector(1, 2, 3), nothing, nothing)
    _, J = NLLSsolver.computeresjac(NLLSsolver.static(7), obs, ch, dv, cc)
    # the frame and direction blocks through the variables' updates (the camera at its anchor); the camera block through the
    # chart's update (π without the valid-domain test, which Core evaluates on plain numbers only)
    f = updated_residual(obs, ch, dv, cc)
    @test J[:, 1:5] ≈ ForwardDiff.jacobian(f, zeros(5)) rtol = 1.0e-7
    g(δ) = PanoramaStitching._image(VgtkCore.model(NLLSsolver.update(cc, δ)), R[3]' * d)
    @test J[:, 6:12] ≈ ForwardDiff.jacobian(g, zeros(7)) rtol = 1.0e-9
    @test NLLSsolver.computeresidual(obs, ch, dv, cc) ≈ VgtkCore.project(TRUE_CAM, R[3]' * d) - x
    for (flags, cols) in ((1, 1:3), (2, 4:5), (3, 1:5), (4, 6:12), (5, [1:3; 6:12]), (6, 4:12))
        @test NLLSsolver.computeresjac(flags, obs, ch, dv, cc)[2] ≈ J[:, cols]
    end
    kw = (; max_iter = 1000, ftol = 1.0e-10, xtol = 1.0e-10)
    tr = [[1 => SVector(600.0, 600.0), 2 => SVector(500.0, 600.0)]]
    ac = (; kw..., optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(), width = 1.5)
    @test_throws ArgumentError refine_camera_and_rotations(TRUE_CAM, R, tr; ac..., fixed = (:focal,))      # no such parameter
    @test_throws ArgumentError refine_camera_and_rotations(TRUE_CAM, R, tr; ac..., fixed = (:f, :f))       # named twice
    @test_throws r"holds every camera parameter" refine_camera_and_rotations(TRUE_CAM, R, tr; ac...,
                                                                               fixed = VgtkCore.parameter_names(TRUE_CAM))
    for k in (:camera_free, :free, :fix, :fixed_params, :hold)                 # the retired names, each naming `fixed`
        @test_throws r"`fixed = \(…\)`" refine_camera_and_rotations(TRUE_CAM, R, tr; ac..., (k => (:f,),)...)
    end
    @test_throws ArgumentError refine_camera_and_rotations(TRUE_CAM, R, tr; kw..., optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(efficiency_constant = VgtkCore.efficiency_constant(VgtkCore.CauchyShape(), 2)))   # no scale or width
    @test_throws ArgumentError refine_camera_and_rotations(TRUE_CAM, R, tr; ac..., scale = 1.0)                          # both
    @test_throws ArgumentError refine_camera_and_rotations(TRUE_CAM, R, tr; ac..., optimization_strategy = nothing)
    # the rotation-only bundle holds the calibration: camera_free is refused, pointing at the autocalibration; so is the
    # registration's bundle_camera_free (and the other retired registration keywords, each naming its successor)
    @test_throws r"refine_camera_and_rotations" refine_rotations(TRUE_CAM, R, tr; kw..., width = 1.5, camera_free = (:f,))
    # the retired loss keywords of the bundle: refused naming the strategy and its scale input
    @test_throws r"optimization_strategy" refine_rotations(TRUE_CAM, R, tr; kw..., width = 1.5, loss = VgtkCore.CauchyShape())
    @test_throws r"optimization_strategy" refine_rotations(TRUE_CAM, R, tr; kw..., loss_width = 1.5)
    @test_throws ArgumentError refine_rotations(TRUE_CAM, R, tr; kw...)                                  # neither scale nor width
    @test_throws ArgumentError refine_rotations(TRUE_CAM, R, tr; kw..., scale = 1.0, width = 1.5)        # both
    @test_throws ArgumentError refine_rotations(TRUE_CAM, R, tr; kw..., optimization_strategy = nothing, width = 1.5)   # L2 takes none
    for k in (:bundle_loss, :averaging_loss)
        @test_throws r"optimization_strategy" PanoramaStitching._refuse_retired_registration_keywords("register_rotations", (; k => 1))
    end
    @test_throws r"refine_camera_and_rotations" PanoramaStitching._refuse_retired_registration_keywords("register_rotations",
                                                                                                       (; bundle_camera_free = ()))
    @test_throws r"bundle_optimization_strategy" PanoramaStitching._refuse_retired_registration_keywords("register_rotations",
                                                                                                     (; bundle_scale_multiple = 3.0))
    @test_throws r"local_optimization_strategy" PanoramaStitching._refuse_retired_registration_keywords("verify_pairs",
                                                                                                      (; local_optimization = nothing))
end

# The camera held (`refine_rotations`, the rotation-only bundle): the numbers of the bundle before the camera could be freed — the fisheye
# sweep of rotation_bundle_tests.jl with outliers, its rms and its rotations as PanoramaStitching 69471ff computed them.
@testitem "refine_rotations holds the calibration: no calibration, the given width, the rotations recovered" setup = [SweepScene, RotatingKB] begin
    using Test, Random, LinearAlgebra, StaticArrays
    import Rotations
    using PanoramaStitching: refine_rotations
    import VisualGeometryToolkitCore as VgtkCore
    # (69471ff's pinned numbers of this fixture were those of the 3-run bundle that re-estimated its width; the bundle is one
    # run at a given width since the pooled RANSAC scale, so the held camera is checked on the result instead)
    rng = Xoshiro(31)
    R = sweep_rotations(rng, 5)
    ts, _, _ = sweep_tracks(rng, R, 400; outl = 0.05)
    start = Dict(k => SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec((0.02 .* randn(rng, 3))...))) * R[k] for k in 1:5)
    start[1] = R[1]
    kw = (; optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(), width = 1.5, max_iter = 1000, ftol = 1.0e-12, xtol = 1.0e-10)
    res = refine_rotations(CAM, start, [[k => x for (k, x) in t] for t in ts]; kw...)
    @test res.metadata.camera === CAM && res.metadata.calibration === nothing
    @test res.metadata.loss_width == 1.5 && VgtkCore.converged(res) && 0 < res.iterations < 1000
    @test maximum(k -> rad2deg(norm(VgtkCore.log_so3(SMatrix{3, 3}(VgtkCore.estimate(res)[k])' * R[k]))), 1:5) < 0.05
    # the rotation-only split leaves the held bundle's numbers as they were (PanoramaStitching ce2ce59). Not bit for bit:
    # StaticArrays' small-matrix products are `muladd` chains, which LLVM contracts into FMAs as each compiled copy and each
    # LLVM decides (Julia 1.12's LLVM 18 and 1.13's LLVM 20 round some 3 × 3 products one ulp apart: this fixture's start
    # rotations already differ, while its draws, every libm function, exp_so3, the camera's maps and BLAS/LAPACK agree bit
    # for bit; measured on dgx01, 2026-11-12). Two such roundings run the same LM iterations to the same minimum, and each
    # stops where its relative cost decrease is below ftol and its relative step below xtol: the rms agrees to
    # rtol = ftol, the estimate (the rotations, the directions) to rtol = xtol. The counts are exact.
    @test (res.iterations, res.n) == (19, 1489) && isapprox(res.rms, 109.24885291344255; rtol = kw.ftol)
    @test isapprox(sum(sum, values(VgtkCore.estimate(res))), 8.795668643015834; rtol = kw.xtol) &&
          isapprox(sum(sum, res.metadata.directions), 144.22098567182522; rtol = kw.xtol)
    # and on the KB scene, the camera held at the wrong start camera
    rng = Xoshiro(7)
    Rk = rows_rotations()
    tracks = exact_tracks(rng, Rk, 600)
    cam0 = VgtkCore.BoundedCamera(VgtkCore.CameraModel(282.0VgtkCore.px, SA[600.0VgtkCore.px, 600.0VgtkCore.px];
                                                       lens = VgtkCore.KannalaBrandt(0.0, 0.0, 0.0, 0.0)), VgtkCore.Size2(1200, 1200))
    startk = Dict(k => (k == 1 ? Rk[1] : SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec((deg2rad(1) .* normalize(randn(rng, 3)))...))) * Rk[k]) for k in 1:10)
    rk = refine_rotations(cam0, startk, tracks; optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(), width = 1.5, max_iter = 1000, ftol = 1.0e-12,
                          xtol = 1.0e-12)
    @test rk.metadata.camera === cam0                                          # the bounds as above (ftol, xtol 1e-12)
    @test (rk.iterations, rk.n) == (435, 3624) && isapprox(rk.rms, 13.626810529489516; rtol = 1.0e-12)
    @test isapprox(sum(sum, values(VgtkCore.estimate(rk))), 18.480086542920372; rtol = 1.0e-12) &&
          isapprox(sum(sum, rk.metadata.directions), 128.27182481038034; rtol = 1.0e-12)
end
