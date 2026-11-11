# =============================================================================
# The bundle's scale: each verified pair's noise scale by SIMFIT++ (VisualGeometryToolkitEstimation's `simfitpp_loss_width`, after
# its RANSAC, on held-out splits), pooled over the kept pairs into one σ̂ (inlier-count weighted variance), the bundle's
# robust width `bundle_efficiency_constant·σ̂` fixed for its one LM run.
# =============================================================================

@testitem "_pair_scale: SIMFIT++ recovers a rotation pair's pixel noise σ" setup = [SweepScene] begin
    using Test, Random, LinearAlgebra, StaticArrays
    using PanoramaStitching: PanoramaStitching, FrameFeatures
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
        fa = FrameFeatures(u1, fill(SMatrix{2, 2}(fill(NaN, 2, 2)), length(u1)), nothing, length(u1))
        fb = FrameFeatures(u2, fill(SMatrix{2, 2}(fill(NaN, 2, 2)), length(u2)), nothing, length(u2))
        cs, _ = PanoramaStitching.pair_bearings(bcam, fa, fb, [(first = i, second = i, score = 1.0) for i in eachindex(u1)])
        lo = VgtkEstimation.Triggs{VgtkCore.WelschShape}(efficiency_constant = 2.5)
        res = VgtkEstimation.find_rotation(cs; max_trials = 2000, min_trials = 100, rng = MersenneTwister(1),
                                           scoring = VgtkCore.MsacScoring(VgtkEstimation.RotationProblem; threshold = 5.0), local_optimization_strategy = lo)
        s = PanoramaStitching._pair_scale(cs, res, lo, MersenneTwister(2))
        @info "SIMFIT++ on a synthetic pair: true σ $σ px, estimated $(round(s.sigma; digits = 3)) px, success $(s.success)"
        @test s.success
        @test isapprox(s.sigma, σ; rtol = 0.15)
    end
end

@testitem "register_rotations: the bundle's width is bundle_efficiency_constant × the pooled pair scale, one LM run" setup = [SweepScene] begin
    using Test, Random, LinearAlgebra, StaticArrays
    using PanoramaStitching: register_rotations, refine_rotations, chain_tracks, FrameFeatures, VerifiedPairs
    import VisualGeometryToolkitCore as VgtkCore
    rng = MersenneTwister(3)
    R = sweep_rotations(rng, 4)
    ts, _, _ = sweep_tracks(rng, R, 400; σ = 0.4)
    K = 4; bcam = VgtkCore.BoundedCamera(CAM, VgtkCore.Size2(1200, 1200))
    pixels = [SVector{2, Float64}[] for _ in 1:K]; index = [Dict{Int, Int}() for _ in 1:K]
    for (t, obs) in enumerate(ts), (k, u) in obs
        push!(pixels[k], u); index[k][t] = length(pixels[k])
    end
    feats = [FrameFeatures(pixels[k], fill(SMatrix{2, 2}(fill(NaN, 2, 2)), length(pixels[k])), nothing, length(pixels[k])) for k in 1:K]
    verified = Dict{Tuple{Int, Int}, NamedTuple{(:rotation, :inliers), Tuple{SMatrix{3, 3, Float64, 9}, Int}}}()
    matches = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    stats = Dict{Tuple{Int, Int}, NamedTuple{(:matches, :inliers, :rms_px, :rotation_deg), Tuple{Int, Int, Float64, Float64}}}()
    for i in 1:K, j in (i + 1):K
        common = sort!([t for t in keys(index[i]) if haskey(index[j], t)]); length(common) >= 25 || continue
        verified[(i, j)] = (rotation = SMatrix{3, 3, Float64, 9}(R[j]' * R[i]), inliers = length(common))
        matches[(i, j)] = [(index[i][t], index[j][t]) for t in common]
        stats[(i, j)] = (matches = length(common), inliers = length(common), rms_px = 0.4, rotation_deg = 0.0)
    end
    ps = sort(collect(keys(verified)))
    # per pair a scale: 0.3 px, but 0.6 px on the first pair and a failed estimate on the last (not pooled)
    scales = Dict(p => (sigma = p == ps[1] ? 0.6 : 0.3, success = p != ps[end]) for p in ps)
    vp = VerifiedPairs(feats, verified, matches, stats, scales, (; descriptor = (;), matcher = (;)))
    pooled = (n = [verified[p].inliers for p in ps[1:(end - 1)]]; σ = [scales[p].sigma for p in ps[1:(end - 1)]];
              sqrt(sum(n .* σ .^ 2) / sum(n)))
    bkw = (; averaging_loss = VgtkCore.HuberShape(), bundle_loss = VgtkCore.WelschShape(), bundle_efficiency_constant = 2.5,
           bundle_max_iter = 1000, bundle_ftol = 1.0e-12, bundle_xtol = 1.0e-6)
    reg = register_rotations(vp, bcam; bkw...)
    @test reg.bundle_sigma ≈ pooled
    @test reg.bundle.metadata.loss_width ≈ 2.5 * pooled
    @test reg.bundle.retcode === :converged && reg.bundle.iterations < 1000
    # the same bundle with that width given explicitly: one LM run, the same result
    obs = [[k => feats[k].pixels[kp] for (k, kp) in t] for t in reg.tracks]
    direct = refine_rotations(bcam, VgtkCore.estimate(reg.averaging), obs; loss = VgtkCore.WelschShape(), loss_width = 2.5 * pooled,
                              max_iter = 1000, ftol = 1.0e-12, xtol = 1.0e-6)
    @test [VgtkCore.estimate(direct)[k] for k in 1:K] == reg.rotations
    @test direct.iterations == reg.bundle.iterations
    # no pair with a scale: refused; a width that is not positive: refused
    none = VerifiedPairs(feats, verified, matches, stats, Dict(p => (sigma = NaN, success = false) for p in ps), vp.loma)
    @test_throws ArgumentError register_rotations(none, bcam; bkw...)
    @test_throws ArgumentError refine_rotations(bcam, VgtkCore.estimate(reg.averaging), obs; loss = VgtkCore.WelschShape(), loss_width = 0.0,
                                                max_iter = 10, ftol = 1.0e-12, xtol = 1.0e-6)
end
