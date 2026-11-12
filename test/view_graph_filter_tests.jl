# =============================================================================
# register_rotations' view-graph filters (VisualGeometryToolkitEstimation's fit_filtered_rotation_averaging): a pair verified one
# period off, as a periodic scene verifies it, is removed by the cycle and the orientation filters, and its matches make no
# track; the tolerances' defaults (2° and 5°) and refusals.
# =============================================================================

@testitem "register_rotations: the view-graph filters remove a pair one period off, and its tracks" setup = [SweepScene, BayerMaps] begin
    using Test, Random, LinearAlgebra, StaticArrays
    using PanoramaStitching: register_rotations, VerifiedPairs
    using VisualGeometryToolkitCore: BoundedCamera, Size2, exp_so3
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    BCAM = BoundedCamera(CAM, Size2(1200, 1200))

    # The verified pairs of a 6-frame sweep (frame k at yaw 0.5(k − 1) rad): every pair sharing ≥ 25 tracks verified with its true
    # rotation R_ij = R_jᵀR_i and its true matches, but `alias` (i, j) verified 12° off about the vertical (one bay of a periodic
    # hall) with its matches one keypoint off (the aliased correspondences).
    function aliased_pairs(; alias = (2, 4), rng = MersenneTwister(3))
        R = sweep_rotations(rng, 6)
        ts, _, _ = sweep_tracks(rng, R, 800; σ = 0.3)
        K = length(R)
        pixels = [SVector{2, Float64}[] for _ in 1:K]
        index = [Dict{Int, Int}() for _ in 1:K]                     # track → keypoint, per frame
        for (t, obs) in enumerate(ts), (k, u) in obs
            push!(pixels[k], u); index[k][t] = length(pixels[k])
        end
        feats = [described(pixels[k]) for k in 1:K]                  # unit isotropic covariances: the pixel residuals
        verified = Dict{Tuple{Int, Int}, NamedTuple{(:rotation, :inliers, :covariance), Tuple{SMatrix{3, 3, Float64, 9}, Int, Union{Nothing, SMatrix{3, 3, Float64, 9}}}}}()
        matches = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
        stats = Dict{Tuple{Int, Int}, NamedTuple{(:matches, :inliers, :rms, :rotation_deg), Tuple{Int, Int, Float64, Float64}}}()
        for i in 1:K, j in (i + 1):K
            common = sort!([t for t in keys(index[i]) if haskey(index[j], t)])
            length(common) >= 25 || continue
            Rij = SMatrix{3, 3, Float64, 9}(R[j]' * R[i])
            m = [(index[i][t], index[j][t]) for t in common]
            if (i, j) == alias
                Rij = SMatrix{3, 3, Float64, 9}(exp_so3(SVector(0.0, deg2rad(12.0), 0.0))) * Rij
                m = [(m[q][1], m[mod1(q + 1, length(m))][2]) for q in eachindex(m)]
            end
            verified[(i, j)] = (rotation = Rij, inliers = length(m), covariance = SMatrix{3, 3, Float64, 9}(1.0e-6, 0, 0, 0, 1.0e-6, 0, 0, 0, 1.0e-6)); matches[(i, j)] = m
            stats[(i, j)] = (matches = length(m), inliers = length(m), rms = 0.3, rotation_deg = 0.0)
        end
        return VerifiedPairs(feats, verified, matches, stats, Dict(p => (sigma = 0.3, dof = 2 * verified[p].inliers - 3) for p in keys(verified)),
                             (; descriptor = (;), matcher = (;))), R
    end
    vp, R = aliased_pairs()
    @test haskey(vp.verified, (2, 4)) && length(vp.verified) > 8
    bkw = (; frame_map = IDENTITY_MAP, averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100), bundle_optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(efficiency_constant = 3.0), 
           bundle_max_iter = 1000, bundle_ftol = 1.0e-10, bundle_xtol = 1.0e-10)
    err(reg) = maximum(k -> rad2deg(norm(VgtkCore.log_so3(SMatrix{3, 3}(reg.rotations[k])' * R[k]))), 1:6)
    # the defaults: cycle_tol 2°, orientation_tol 5°
    reg = register_rotations(vp, BCAM; bkw...)
    @test reg.removed == [(2, 4)]
    @test !haskey(reg.graph, 2, 4) && VgtkEstimation.nedges(reg.graph) == length(vp.verified) - 1
    @test length(reg.pairs) == length(vp.verified)                    # every verified pair's numbers, the removed one's too
    # no track holds the aliased matches: every track is one scene point's
    @test reg.bundle.rms < 1.0 && err(reg) < 0.05
    @test register_rotations(vp, BCAM; bkw..., cycle_tol = deg2rad(2), orientation_tol = deg2rad(5)).rotations == reg.rotations
    # either filter alone removes it here
    @test register_rotations(vp, BCAM; bkw..., orientation_tol = nothing).removed == [(2, 4)]
    @test register_rotations(vp, BCAM; bkw..., cycle_tol = nothing).removed == [(2, 4)]
    # both skipped: the pair stays, its aliased matches in the tracks
    off = register_rotations(vp, BCAM; bkw..., cycle_tol = nothing, orientation_tol = nothing)
    @test isempty(off.removed) && haskey(off.graph, 2, 4)
    # the aliased matches chain true tracks of frames 2 and 4 into inconsistent ones (two keypoints of a frame), which chain_tracks
    # drops: with the filters none is
    @test reg.dropped == 0
    @test off.dropped > 0 && length(off.tracks) < length(reg.tracks)
    # a tolerance must be positive (radians) or nothing
    @test_throws ArgumentError register_rotations(vp, BCAM; bkw..., cycle_tol = 0.0)
    @test_throws ArgumentError register_rotations(vp, BCAM; bkw..., orientation_tol = -1.0)
end

@testitem "register_rotations: a filter tolerance is refused before any feature is detected" setup = [BayerMaps] begin
    using Test, StaticArrays
    using PanoramaStitching: register_rotations
    import VisualGeometryToolkitCore as VgtkCore
    cam = VgtkCore.BoundedCamera(VgtkCore.CameraModel(300.0VgtkCore.px, SA[64.0VgtkCore.px, 64.0VgtkCore.px]; lens = VgtkCore.KannalaBrandt(-0.02)),
                                 VgtkCore.Size2(128, 128))
    kw = (; frame_map = IDENTITY_MAP, tone_percentile = 99.0, detector_margin_px = 0.0, asift_tilts = :imas25,
          max_detections = 16, descriptor = :b, matcher = :b128, min_score = 0.0, pairs = :all, max_lag = 1, overlap_deg = 90.0,
          min_inliers = 10, seed = 1, max_trials = 10, min_trials = 1, averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100),
          bundle_optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(efficiency_constant = 3.0), bundle_max_iter = 1,
          bundle_ftol = 1.0e-8, bundle_xtol = 1.0e-8)
    frames = [rand(128, 128, 3) for _ in 1:2]
    @test_throws ArgumentError register_rotations(frames, cam; kw..., cycle_tol = 0.0)
    @test_throws ArgumentError register_rotations(frames, cam; kw..., orientation_tol = NaN)
end
