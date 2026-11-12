# The averaging's edges carry their rotations' covariances (the owner's ruling): each verified pair's
# `uncertain_rotation` (its scorer's noise_scale² × the implicit-function covariance of its pixel residuals), a pair with no determined covariance dropped and recorded (refused when the
# drop splits the frames), and the averaging's scale record ScaleUpstream (its edges' covariances absolute: no second scale).

@testitem "edge uncertainty: a verified pair's edge is its uncertain_rotation" setup = [SweepScene, BayerMaps] begin
    using Test, Random, LinearAlgebra, Statistics, StaticArrays
    using PanoramaStitching: PanoramaStitching, pair_bearings
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    import VisualGeometryToolkitFeatures as VgtkFeatures
    rng = MersenneTwister(5)
    bcam = VgtkCore.BoundedCamera(CAM, VgtkCore.Size2(1200, 1200))
    R = SMatrix{3, 3, Float64, 9}(VgtkCore.exp_so3(SVector(0.02, deg2rad(20.0), -0.01)))
    u1 = SVector{2, Float64}[]; u2 = SVector{2, Float64}[]
    while length(u1) < 600
        f = normalize(randn(rng, SVector{3, Float64})); acos(f[3]) < deg2rad(70) || continue
        g = R * f; acos(clamp(g[3], -1, 1)) < deg2rad(70) || continue
        push!(u1, VgtkCore.project(CAM, f) + 0.7 * randn(rng, SVector{2, Float64}))
        push!(u2, VgtkCore.project(CAM, g) + 0.7 * randn(rng, SVector{2, Float64}))
    end
    fa = described(u1); fb = described(u2)                         # plain keypoints (no keypoint covariance in this release)
    ms = [VgtkFeatures.NNMatch(i, i, 1.0f0, 0.5f0) for i in eachindex(u1)]
    cs, kept = pair_bearings(bcam, fa, fb, ms; frame_map = IDENTITY_MAP)
    # plain endpoints: no covariance columns (unit isotropic pixel noise)
    @test !VgtkCore.hasattr(cs, :Σ₁) && !VgtkCore.hasattr(cs, :Σ₂) && kept[1] == (1, 1)
    @test VgtkCore.hasattr(cs, :score) && cs.score[1] == 1.0                       # Features' csponds' score column
    res = VgtkEstimation.find_rotation(cs; max_trials = 2000, min_trials = 100, rng = MersenneTwister(1),
                                       scoring = VgtkCore.MsacScoring(VgtkEstimation.RotationProblem; threshold = 5.0))
    Σ = PanoramaStitching._verified_rotation_covariance(cs, res)
    @test Σ == SMatrix{3, 3, Float64, 9}(Statistics.cov(VgtkEstimation.uncertain_rotation(VgtkEstimation.RotationProblem(cs), res)))
    @test Σ ≈ Σ' && all(>(0), eigvals(Symmetric(Matrix(Σ))))
    # the edge: the pair's rotation with that covariance, weighted by its inliers
    verified = Dict((1, 2) => (rotation = SMatrix{3, 3, Float64, 9}(VgtkCore.estimate(res)), inliers = 100, covariance = Σ))
    e = PanoramaStitching._view_graph(2, verified).edges[(1, 2)]
    @test VgtkCore.weight_of(e) == 100.0 && Statistics.cov(VgtkCore.value_of(e)) == Σ
end

@testitem "edge uncertainty: a pair with no covariance is dropped and recorded; a dropped bridge is refused; scale 1, upstream" begin
    using Test, StaticArrays
    import Rotations
    using PanoramaStitching: PanoramaStitching
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    R = Dict(k => SMatrix{3, 3, Float64, 9}(Rotations.RotY(0.3 * (k - 1))) for k in 1:4)
    Σ = SMatrix{3, 3, Float64, 9}(1.0e-6, 0, 0, 0, 1.0e-6, 0, 0, 0, 1.0e-6)
    edge(i, j; cov = Σ) = (rotation = R[j]' * R[i], inliers = 100, covariance = cov)
    # a 4-cycle with one pair undetermined: it is no edge, the graph stays connected, the pair is listed
    verified = Dict((1, 2) => edge(1, 2), (2, 3) => edge(2, 3), (3, 4) => edge(3, 4), (1, 4) => edge(1, 4; cov = nothing))
    vg = PanoramaStitching._view_graph(4, verified)
    @test !haskey(vg.edges, (1, 4)) && length(vg.edges) == 3 && PanoramaStitching._no_covariance_pairs(verified) == [(1, 4)]
    avg = VgtkEstimation.fit_rotation_averaging(vg)
    @test VgtkCore.scale_source(avg) == VgtkCore.ScaleUpstream && VgtkCore.noise_scale(avg) == 1.0   # absolute edges: no second scale
    # the only edge to frame 4 undetermined: refused, naming it (no averaging of a partial graph)
    chain = Dict((1, 2) => edge(1, 2), (2, 3) => edge(2, 3), (3, 4) => edge(3, 4; cov = nothing))
    @test_throws r"pair\(s\) with no determined rotation covariance 2-3 disconnects" PanoramaStitching._view_graph(4, chain)   # 0-based
end
