# =============================================================================
# register_rotations' view-graph filters (VisualGeometryToolkitEstimation's FilteredRotationAveraging): a pair verified one
# period off, as a periodic scene verifies it, is removed by the cycle and the orientation filters, and its matches make no
# track; the tolerances' defaults (2° and 5°) and refusals.
# =============================================================================

@testitem "register_rotations: the view-graph filters remove a pair one period off, and its tracks" setup = [SweepScene, BayerMaps] begin
    using Test, Random, LinearAlgebra, StaticArrays
    using PanoramaStitching: PanoramaStitching, register_rotations, VerifiedPairs
    using VisualGeometryToolkitCore: BoundedCamera, Size2, exp_so3
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    # BCAM: SweepScene's camera, its valid region an image circle (the always-on near-duplicate check reads it)

    # The verified pairs of a 6-frame sweep (frame k at yaw 0.5(k − 1) rad): every pair sharing ≥ 25 tracks verified with its true
    # rotation R_ij = R_jᵀR_i and its true matches, but `alias` (i, j) verified 12° off about the vertical (one bay of a periodic
    # hall) with its matches one detection off (the aliased correspondences).
    function aliased_pairs(; alias = (2, 4), rng = MersenneTwister(3))
        R = sweep_rotations(rng, 6)
        ts, _, _ = sweep_tracks(rng, R, 800; σ = 0.3)
        K = length(R)
        pixels = [SVector{2, Float64}[] for _ in 1:K]
        index = [Dict{Int, Int}() for _ in 1:K]                     # track → detection, per frame
        for (t, obs) in enumerate(ts), (k, u) in obs
            push!(pixels[k], u); index[k][t] = length(pixels[k])
        end
        feats = [described(pixels[k]) for k in 1:K]                  # plain detections: the pixel residuals
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
    averaging(; cycle_tol = deg2rad(2), orientation_tol = deg2rad(5)) =
        VgtkEstimation.FilteredRotationAveraging(; cycle_tol, orientation_tol, averaging = VgtkEstimation.RotationAveraging(; optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100)))
    bkw = (; frame_map = IDENTITY_MAP, averaging = averaging(),
           bundle = PanoramaStitching.TrackCheck(VgtkCore.LevenbergMarquardt(; optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(efficiency_constant = 3.0))),
           bundle_stops = VgtkCore.LmStops(; max_iter = 1000, ftol = 1.0e-10, xtol = 1.0e-10))
    err(reg) = maximum(k -> rad2deg(norm(VgtkCore.log_so3(SMatrix{3, 3}(reg.rotations[k])' * R[k]))), 1:6)
    # the defaults: cycle_tol 2°, orientation_tol 5°
    reg = register_rotations(vp, BCAM; bkw...)
    @test register_rotations(vp, BCAM; Base.structdiff(bkw, NamedTuple{(:averaging,)})...).removed == reg.removed   # register_rotations' default averaging
    @test isempty(reg.near_duplicates)          # the near-duplicate check ran: these frames share at most 0.88 of a footprint
    @test reg.removed == [(2, 4)]
    @test !haskey(reg.graph, 2, 4) && VgtkEstimation.nedges(reg.graph) == length(vp.verified) - 1
    @test length(reg.pairs) == length(vp.verified)                    # every verified pair's numbers, the removed one's too
    # no track holds the aliased matches: every track is one scene point's
    @test reg.bundle.rms < 1.0 && err(reg) < 0.05
    @test register_rotations(vp, BCAM; bkw..., averaging = averaging(; cycle_tol = deg2rad(2), orientation_tol = deg2rad(5))).rotations == reg.rotations
    # either filter alone removes it here
    @test register_rotations(vp, BCAM; bkw..., averaging = averaging(; orientation_tol = nothing)).removed == [(2, 4)]
    @test register_rotations(vp, BCAM; bkw..., averaging = averaging(; cycle_tol = nothing)).removed == [(2, 4)]
    # both skipped: the pair stays, its aliased matches in the tracks
    off = register_rotations(vp, BCAM; bkw..., averaging = averaging(; cycle_tol = nothing, orientation_tol = nothing))
    @test isempty(off.removed) && haskey(off.graph, 2, 4)
    # the aliased matches chain true tracks of frames 2 and 4 into inconsistent ones (two detections of a frame), which chain_tracks
    # drops: with the filters none is
    @test reg.dropped == 0
    @test off.dropped > 0 && length(off.tracks) < length(reg.tracks)
    # a verified pair with no determined covariance is never an edge: it is `no_covariance`, not one the filters `removed`
    p0 = (1, 2); e0 = vp.verified[p0]
    vn = VerifiedPairs(vp.features, merge(vp.verified, Dict(p0 => (rotation = e0.rotation, inliers = e0.inliers, covariance = nothing))),
                                         vp.inlier_matches, vp.stats, vp.scales, vp.loma)
    rn = register_rotations(vn, BCAM; bkw...)
    @test rn.no_covariance == [p0] && rn.removed == [(2, 4)]
    # a tolerance must be positive (radians) or nothing: refused where the averaging is built (Estimation's)
    @test_throws ArgumentError averaging(; cycle_tol = 0.0)
    @test_throws ArgumentError averaging(; orientation_tol = -1.0)
end

@testitem "register_rotations: a filter tolerance is refused where the averaging is built, before any feature is detected" begin
    # the view-graph filters are the `averaging` value's (Estimation's FilteredRotationAveraging): a tolerance that is not positive
    # is refused when it is built, so no registration with it starts
    using Test
    import VisualGeometryToolkitEstimation as VgtkEstimation
    @test_throws ArgumentError VgtkEstimation.FilteredRotationAveraging(; cycle_tol = 0.0)
    @test_throws ArgumentError VgtkEstimation.FilteredRotationAveraging(; orientation_tol = NaN)
end
