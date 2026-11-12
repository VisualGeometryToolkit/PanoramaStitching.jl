# =============================================================================
# The bundle's scale: each verified pair's noise scale by SIMFIT++ (VisualGeometryToolkitEstimation's `simfitpp_loss_width`, after
# its RANSAC, on held-out splits), pooled over the kept pairs into one σ̂ (inlier-count weighted variance), the bundle's
# robust width the bundle strategy's efficiency constant × σ̂, fixed for its one LM run (its `scale = σ̂`).
# =============================================================================

@testitem "_pair_scale: SIMFIT++ recovers a rotation pair's pixel noise σ" setup = [SweepScene, BayerMaps] begin
    using Test, Random, LinearAlgebra, StaticArrays
    using PanoramaStitching: PanoramaStitching
    import VisualGeometryToolkitFeatures as VgtkFeatures
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    rng = MersenneTwister(5)
    bcam = VgtkCore.BoundedCamera(CAM, VgtkCore.Size2(1200, 1200))
    R = SMatrix{3, 3, Float64, 9}(VgtkCore.exp_so3(SVector(0.02, deg2rad(20.0), -0.01)))
    for σ in (0.5, 1.2)
        u1 = SVector{2, Float64}[]; u2 = SVector{2, Float64}[]
        while length(u1) < 1500
            f = normalize(randn(rng, SVector{3, Float64})); acos(f[3]) < deg2rad(70) || continue
            g = R * f; acos(clamp(g[3], -1, 1)) < deg2rad(70) || continue
            push!(u1, VgtkCore.project(CAM, f) + σ * randn(rng, SVector{2, Float64}))
            push!(u2, rand(rng) < 0.2 ? SVector(600 + 500 * (2rand(rng) - 1), 600 + 500 * (2rand(rng) - 1)) :
                                         VgtkCore.project(CAM, g) + σ * randn(rng, SVector{2, Float64}))
        end
        fa = described(u1); fb = described(u2)                     # unit isotropic covariances: whitened residuals in pixels
        cs, _ = PanoramaStitching.pair_bearings(bcam, fa, fb, [VgtkFeatures.NNMatch(i, i, 1.0f0, 0.5f0) for i in eachindex(u1)];
                                                frame_map = IDENTITY_MAP)
        lo = VgtkCore.Triggs{VgtkCore.WelschShape}(efficiency_constant = 2.5)
        res = VgtkEstimation.find_rotation(cs; max_trials = 2000, min_trials = 100, rng = MersenneTwister(1),
                                           scoring = VgtkCore.MsacScoring(VgtkEstimation.RotationProblem; threshold = 5.0), local_optimization_strategy = lo)
        s = PanoramaStitching._pair_scale(cs, res, lo, MersenneTwister(2))
        @info "SIMFIT++ on a synthetic pair: true σ $σ px, estimated $(round(s.sigma; digits = 3)) px, success $(s.success)"
        @test s.success
        @test isapprox(s.sigma, σ; rtol = 0.15)
    end
end

@testitem "register_rotations: the bundle's width is its strategy's efficiency constant × the pooled pair scale, one LM run" setup = [SweepScene, BayerMaps] begin
    using Test, Random, LinearAlgebra, StaticArrays
    using PanoramaStitching: PanoramaStitching, register_rotations, refine_rotations, chain_tracks, VerifiedPairs
    import VisualGeometryToolkitCore as VgtkCore
    rng = MersenneTwister(3)
    R = sweep_rotations(rng, 4)
    ts, _, _ = sweep_tracks(rng, R, 400; σ = 0.4)
    K = 4; bcam = VgtkCore.BoundedCamera(CAM, VgtkCore.Size2(1200, 1200))
    pixels = [SVector{2, Float64}[] for _ in 1:K]; index = [Dict{Int, Int}() for _ in 1:K]
    for (t, obs) in enumerate(ts), (k, u) in obs
        push!(pixels[k], u); index[k][t] = length(pixels[k])
    end
    feats = [described(pixels[k]) for k in 1:K]
    verified = Dict{Tuple{Int, Int}, NamedTuple{(:rotation, :inliers, :covariance), Tuple{SMatrix{3, 3, Float64, 9}, Int, Union{Nothing, SMatrix{3, 3, Float64, 9}}}}}()
    matches = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    stats = Dict{Tuple{Int, Int}, NamedTuple{(:matches, :inliers, :rms, :rotation_deg), Tuple{Int, Int, Float64, Float64}}}()
    for i in 1:K, j in (i + 1):K
        common = sort!([t for t in keys(index[i]) if haskey(index[j], t)]); length(common) >= 25 || continue
        verified[(i, j)] = (rotation = SMatrix{3, 3, Float64, 9}(R[j]' * R[i]), inliers = length(common), covariance = SMatrix{3, 3, Float64, 9}(1.0e-6, 0, 0, 0, 1.0e-6, 0, 0, 0, 1.0e-6))
        matches[(i, j)] = [(index[i][t], index[j][t]) for t in common]
        stats[(i, j)] = (matches = length(common), inliers = length(common), rms = 0.4, rotation_deg = 0.0)
    end
    ps = sort(collect(keys(verified)))
    # per pair a scale: 0.3 px, but 0.6 px on the first pair and a failed estimate on the last (not pooled)
    scales = Dict(p => (sigma = p == ps[1] ? 0.6 : 0.3, success = p != ps[end]) for p in ps)
    vp = VerifiedPairs(feats, verified, matches, stats, scales, (; descriptor = (;), matcher = (;)))
    pooled = (n = [verified[p].inliers for p in ps[1:(end - 1)]]; σ = [scales[p].sigma for p in ps[1:(end - 1)]];
              sqrt(sum(n .* σ .^ 2) / sum(n)))
    bkw = (; frame_map = IDENTITY_MAP, averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100), bundle_optimization_strategy = VgtkCore.Triggs{VgtkCore.WelschShape}(efficiency_constant = 2.5), 
           bundle_max_iter = 1000, bundle_ftol = 1.0e-12, bundle_xtol = 1.0e-6)
    reg = register_rotations(vp, bcam; bkw...)
    @test reg.bundle_sigma ≈ pooled
    @test reg.bundle.metadata.loss_width ≈ 2.5 * pooled
    # the scale record of the robust-estimation contract: the pooled scale, given
    @test VgtkCore.noise_scale(reg.bundle) == reg.bundle_sigma && VgtkCore.scale_source(reg.bundle) == VgtkCore.ScaleGiven
    @test VgtkCore.converged(reg.bundle) && reg.bundle.iterations < 1000
    # the same bundle with that width given explicitly: one LM run, the same result (its observations the keypoints with their
    # covariances, as register_rotations builds them)
    obs = [[k => PanoramaStitching._camera_keypoint(IDENTITY_MAP, feats[k][kp].detection) for (k, kp) in t] for t in reg.tracks]
    direct = refine_rotations(bcam, VgtkCore.estimate(reg.averaging), obs; optimization_strategy = VgtkCore.Triggs{VgtkCore.WelschShape}(), width = 2.5 * pooled,
                              max_iter = 1000, ftol = 1.0e-12, xtol = 1.0e-6)
    @test [VgtkCore.estimate(direct)[k] for k in 1:K] == reg.rotations
    @test direct.iterations == reg.bundle.iterations
    # no pair with a scale: refused; a width that is not positive: refused
    none = VerifiedPairs(feats, verified, matches, stats, Dict(p => (sigma = NaN, success = false) for p in ps), vp.loma)
    @test_throws ArgumentError register_rotations(none, bcam; bkw...)
    @test_throws ArgumentError refine_rotations(bcam, VgtkCore.estimate(reg.averaging), obs; optimization_strategy = VgtkCore.Triggs{VgtkCore.WelschShape}(), width = 0.0,
                                                max_iter = 10, ftol = 1.0e-12, xtol = 1.0e-6)
end

@testitem "the default bundle strategy is VisualGeometryToolkitEstimation's [refine_camera_and_poses]: Core's constant × scale" setup = [SweepScene] begin
    using Test, Random, StaticArrays
    using PanoramaStitching: PanoramaStitching, refine_rotations
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    import TOML
    table = TOML.parsefile(VgtkEstimation.default_config_path())["refine_camera_and_poses"]["optimization_strategy"]
    record = VgtkEstimation.optimization_strategy_from_toml(table; class = :refine_camera_and_poses)
    @test PanoramaStitching._DEFAULT_BUNDLE_STRATEGY == record
    @test VgtkCore.lo_width(PanoramaStitching._DEFAULT_BUNDLE_STRATEGY) === nothing   # no constant of its own: Core's at d = 2
    rng = MersenneTwister(4)
    R = sweep_rotations(rng, 3)
    ts, _, _ = sweep_tracks(rng, R, 200; σ = 0.4)
    bcam = VgtkCore.BoundedCamera(CAM, VgtkCore.Size2(1200, 1200))
    kw = (; scale = 0.4, max_iter = 200, ftol = 1.0e-12, xtol = 1.0e-9)
    dflt = refine_rotations(bcam, R, ts; kw...)
    c = VgtkCore.efficiency_constant(VgtkCore.WelschShape(), 2)
    stated = refine_rotations(bcam, R, ts; optimization_strategy = VgtkCore.Triggs{VgtkCore.WelschShape}(efficiency_constant = c), kw...)
    # one loss everywhere: the width is Estimation's bundle_loss of the same keywords (c × scale, deployed), the scale given
    bl = VgtkEstimation.bundle_loss(PanoramaStitching._DEFAULT_BUNDLE_STRATEGY; scale = 0.4)
    @test dflt.metadata.loss_width == sqrt(VgtkCore.sq_loss_width(bl.loss)) && dflt.metadata.loss_width ≈ c * 0.4
    @test VgtkCore.noise_scale(dflt) == 0.4 && VgtkCore.scale_source(dflt) == VgtkCore.ScaleGiven
    @test [VgtkCore.estimate(dflt)[k] for k in 1:3] == [VgtkCore.estimate(stated)[k] for k in 1:3]
    @test dflt.iterations == stated.iterations
end

@testitem "_pair_scale: SIMFIT++ on the default strategy runs no refine with a scorer (the rotation class's local optimization)" setup = [SweepScene, BayerMaps] begin
    using Test, Random, LinearAlgebra, StaticArrays
    using PanoramaStitching: PanoramaStitching
    import VisualGeometryToolkitFeatures as VgtkFeatures
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    rng = MersenneTwister(6)
    bcam = VgtkCore.BoundedCamera(CAM, VgtkCore.Size2(1200, 1200))
    R = SMatrix{3, 3, Float64, 9}(VgtkCore.exp_so3(SVector(0.01, deg2rad(15.0), 0.02)))
    σ = 0.8
    u1 = SVector{2, Float64}[]; u2 = SVector{2, Float64}[]
    while length(u1) < 1200
        f = normalize(randn(rng, SVector{3, Float64})); acos(f[3]) < deg2rad(70) || continue
        g = R * f; acos(clamp(g[3], -1, 1)) < deg2rad(70) || continue
        push!(u1, VgtkCore.project(CAM, f) + σ * randn(rng, SVector{2, Float64}))
        push!(u2, rand(rng) < 0.2 ? SVector(600 + 500 * (2rand(rng) - 1), 600 + 500 * (2rand(rng) - 1)) :
                                     VgtkCore.project(CAM, g) + σ * randn(rng, SVector{2, Float64}))
    end
    fa = described(u1); fb = described(u2)                         # unit isotropic covariances: whitened residuals in pixels
    cs, _ = PanoramaStitching.pair_bearings(bcam, fa, fb, [VgtkFeatures.NNMatch(i, i, 1.0f0, 0.5f0) for i in eachindex(u1)];
                                            frame_map = IDENTITY_MAP)
    res = VgtkEstimation.find_rotation(cs; max_trials = 2000, min_trials = 100, rng = MersenneTwister(1),
                                       scoring = VgtkCore.MsacScoring(VgtkEstimation.RotationProblem; threshold = 5.0))
    # the default (`nothing`) is the strategy find_rotation ran: Estimation's [pipelines.rotation]
    @test PanoramaStitching._ROTATION_LOCAL_OPTIMIZATION ==
          VgtkEstimation.optimization_strategy_from_toml(VgtkCore.toml_default(PanoramaStitching._ESTIMATION_DEFAULTS, "pipelines.rotation",
                                                                                "local_optimization_strategy"); class = :rotation)
    s = PanoramaStitching._pair_scale(cs, res, nothing, MersenneTwister(2))
    @test s.success
    @test isapprox(s.sigma, σ; rtol = 0.15)
    # a refine_* refuses a scorer: the SIMFIT++ refits are the seeded local optimization, not a refine
    @test_throws ArgumentError VgtkEstimation.refine_rotation(VgtkCore.estimate(res), cs;
                                                              scoring = VgtkCore.MsacScoring(VgtkEstimation.RotationProblem; threshold = 1.0))
end
