# =============================================================================
# The rotation bundle with the camera free (`refine_rotations(…; camera_free)`): the self-calibration of a rotating camera.
# Exact data from a known Kannala–Brandt camera are fitted exactly from a wrong camera and a perturbed start; the reduced
# information shows the one gauge (the first frame's rotation) and nothing else free; the camera block's Jacobian against
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

@testitem "refine_rotations, camera free: a known KB lens recovered exactly from a rotating sequence; one gauge" setup = [RotatingKB] begin
    using Test, Random, LinearAlgebra, StaticArrays
    import Rotations
    using PanoramaStitching: refine_rotations
    import VisualGeometryToolkitCore as VgtkCore
    rng = Xoshiro(7)
    R = rows_rotations()
    tracks = exact_tracks(rng, R, 600)
    # the start: an equidistant camera (k = 0), f 6 % short, the principal point at the sensor centre; rotations 1° off
    cam0 = VgtkCore.BoundedCamera(VgtkCore.CameraModel(282.0VgtkCore.px, SA[600.0VgtkCore.px, 600.0VgtkCore.px];
                                                       lens = VgtkCore.KannalaBrandt(0.0, 0.0, 0.0, 0.0)), VgtkCore.Size2(1200, 1200))
    start = Dict(k => (k == 1 ? R[1] : SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec((deg2rad(1) .* normalize(randn(rng, 3)))...))) * R[k]) for k in 1:10)
    free = (:f, :cx, :cy, :k1, :k2, :k3, :k4)
    kw = (; loss = VgtkCore.CauchyShape(), loss_width = 1.5, max_iter = 1000, ftol = 1.0e-12, xtol = 1.0e-12)
    res = refine_rotations(cam0, start, tracks; kw..., camera_free = free)
    @test res.retcode === :converged
    cam = res.metadata.camera
    @test cam isa VgtkCore.BoundedCamera && VgtkCore.extent(cam) == VgtkCore.extent(cam0)
    θ = VgtkCore.parameter_vector(VgtkCore.model(cam)); θt = VgtkCore.parameter_vector(TRUE_CAM)
    @test θ ≈ θt rtol = 1.0e-9                                            # every parameter, the held aspect and skew included
    @test res.rms < 1.0e-8                                                # exact data: an exact fit
    @test maximum(k -> norm(SMatrix{3, 3}(VgtkCore.estimate(res)[k]) - R[k]), 1:10) < 1.0e-10
    c = res.metadata.calibration
    @test c.free == free
    @test collect(values(c.parameters)) ≈ θt[[1, 4, 5, 6, 7, 8, 9]] rtol = 1.0e-9
    # one gauge: the reduced information with the first frame's rotation has exactly three null directions (a common rotation
    # of the frames and the directions); without it, none (nothing else free); every kept direction is observed
    e = c.reduced_eigenvalues; eg = c.reduced_eigenvalues_gauge_fixed
    @test length(e) == 3 * 10 + 7 && length(eg) == 3 * 9 + 7
    @test maximum(abs, e[1:3]) < 1.0e-12 * e[end] && e[4] > 1.0e-8 * e[end]
    @test eg[1] > 1.0e-8 * eg[end]
    @test c.direction_min_eigenvalue > 0
    @test c.nu == 2 * res.n - (3 * 9 + 2 * length(tracks) + 7)
end

@testitem "refine_rotations, camera free: the camera block's Jacobian; the refusals" setup = [RotatingKB] begin
    using Test, LinearAlgebra, StaticArrays
    import ForwardDiff, NLLSsolver
    import PanoramaStitching
    using PanoramaStitching: refine_rotations
    import VisualGeometryToolkitCore as VgtkCore
    R = rows_rotations()
    free = (:f, :cx, :cy, :k1, :k2, :k3, :k4)
    cc = VgtkCore.restrict(VgtkCore.ParameterManifold(TRUE_CAM), free)
    d = normalize(SVector(0.4, -0.3, 1.0))
    ch = VgtkCore.SO3Manifold(SMatrix{3, 3, Float64, 9}(R[3])); dv = VgtkCore.Eliminable{false}(VgtkCore.SphereManifold(d))
    x = VgtkCore.project(TRUE_CAM, R[3]' * d) + SVector(0.3, -0.2)
    obs = PanoramaStitching._CameraObservation{false, Nothing, Nothing}(x, SVector(1, 2, 3), nothing, nothing)
    _, J = NLLSsolver.computeresjac(NLLSsolver.static(7), obs, ch, dv, cc)
    # the frame and direction blocks through the variables' updates (the camera at its anchor); the camera block through the
    # chart's update (π without the valid-domain test, which Core evaluates on plain numbers only)
    f(δ) = NLLSsolver.computeresidual(obs, NLLSsolver.update(ch, δ[1:3]), NLLSsolver.update(dv, δ[4:5]), cc)
    @test J[:, 1:5] ≈ ForwardDiff.jacobian(f, zeros(5)) rtol = 1.0e-7
    g(δ) = PanoramaStitching._image(VgtkCore.model(NLLSsolver.update(cc, δ)), R[3]' * d)
    @test J[:, 6:12] ≈ ForwardDiff.jacobian(g, zeros(7)) rtol = 1.0e-9
    @test NLLSsolver.computeresidual(obs, ch, dv, cc) ≈ VgtkCore.project(TRUE_CAM, R[3]' * d) - x
    for (flags, cols) in ((1, 1:3), (2, 4:5), (3, 1:5), (4, 6:12), (5, [1:3; 6:12]), (6, 4:12))
        @test NLLSsolver.computeresjac(flags, obs, ch, dv, cc)[2] ≈ J[:, cols]
    end
    kw = (; loss = VgtkCore.CauchyShape(), loss_width = 1.5, max_iter = 1000, ftol = 1.0e-10, xtol = 1.0e-10)
    tr = [[1 => SVector(600.0, 600.0), 2 => SVector(500.0, 600.0)]]
    @test_throws ArgumentError refine_rotations(TRUE_CAM, R, tr; kw..., camera_free = (:focal,))       # no such parameter
    @test_throws ArgumentError refine_rotations(TRUE_CAM, R, tr; kw..., camera_free = (:f, :f))        # named twice
end

# The camera held (`camera_free = ()`, the default): the numbers of the bundle before the camera could be freed — the fisheye
# sweep of rotation_bundle_tests.jl with outliers, its rms and its rotations as PanoramaStitching 69471ff computed them.
@testitem "refine_rotations, camera held: no calibration, the given width, the rotations recovered" setup = [SweepScene] begin
    using Test, Random, LinearAlgebra, StaticArrays
    import Rotations
    using PanoramaStitching: refine_rotations
    import VisualGeometryToolkitCore as VgtkCore
    # (69471ff's pinned numbers of this fixture were those of the 3-run bundle that re-estimated its width; the bundle is one
    # run at a given width since the SIMFIT++ scale, so the held camera is checked on the result instead)
    rng = Xoshiro(31)
    R = sweep_rotations(rng, 5)
    ts, _, _ = sweep_tracks(rng, R, 400; outl = 0.05)
    start = Dict(k => SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec((0.02 .* randn(rng, 3))...))) * R[k] for k in 1:5)
    start[1] = R[1]
    kw = (; loss = VgtkCore.CauchyShape(), loss_width = 1.5, max_iter = 1000, ftol = 1.0e-12, xtol = 1.0e-10)
    res = refine_rotations(CAM, start, [[k => x for (k, x) in t] for t in ts]; kw...)
    @test res.metadata.camera === CAM && res.metadata.calibration === nothing
    @test res.metadata.loss_width == 1.5 && res.retcode === :converged && 0 < res.iterations < 1000
    @test maximum(k -> rad2deg(norm(VgtkCore.log_so3(SMatrix{3, 3}(VgtkCore.estimate(res)[k])' * R[k]))), 1:5) < 0.05
end
