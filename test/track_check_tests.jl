# =============================================================================
# The registration's near-duplicate pairs and track check: a near-duplicate pair (optical axes within near_duplicate_deg under
# the averaged rotations, the overlap estimate of :lag_overlap) verifies and is averaged but seeds no track; after the bundle,
# the observations past 3.035 σ̂ are removed and the bundle re-solved until none is (COLMAP's filter_max_reproj_error loop).
# =============================================================================

@testitem "register_rotations: a near-duplicate pair chains no track; the track check removes a planted wrong chain past 3.035 σ̂, the rms drops" setup = [SweepScene, BayerMaps] begin
    using Test, Random, LinearAlgebra, StaticArrays
    import Rotations
    using PanoramaStitching: register_rotations, VerifiedPairs
    import VisualGeometryToolkitCore as VgtkCore
    # K sweep frames 0.5 rad apart and frame K + 1 a near-duplicate of frame `dup` (1° off); tracks of world directions (σ px), and
    # `nwrong` wrong matches planted on the pair `wrong_pair`: detection of track a in frame i ↔ detection of track b in frame j,
    # a and b true tracks each seen in only one of the two frames and in no frame together — each chains a and b into one wrong
    # track of two clusters.
    function planted_sweep(rng; K = 5, dup = 3, wrong_pair = (1, 2), nwrong = 40, σ = 0.4)
        R = Dict(k => (k == 1 ? SMatrix{3, 3}(1.0I) :
                       SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec(0.05randn(rng), 0.5 * (k - 1), 0.05randn(rng))))) for k in 1:K)
        R[K + 1] = R[dup] * SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec(0.0, deg2rad(1.0), 0.0)))
        ts, _, _ = sweep_tracks(rng, R, 600; σ)
        N = K + 1
        pixels = [SVector{2, Float64}[] for _ in 1:N]; index = [Dict{Int, Int}() for _ in 1:N]
        for (t, obs) in enumerate(ts), (k, u) in obs
            push!(pixels[k], u); index[k][t] = length(pixels[k])
        end
        feats = [described(pixels[k]) for k in 1:N]
        verified = Dict{Tuple{Int, Int}, NamedTuple{(:rotation, :inliers, :covariance), Tuple{SMatrix{3, 3, Float64, 9}, Int, Union{Nothing, SMatrix{3, 3, Float64, 9}}}}}()
        matches = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
        stats = Dict{Tuple{Int, Int}, NamedTuple{(:matches, :inliers, :rms, :rotation_deg), Tuple{Int, Int, Float64, Float64}}}()
        for i in 1:N, j in (i + 1):N
            common = sort!([t for t in keys(index[i]) if haskey(index[j], t)]); length(common) >= 25 || continue
            verified[(i, j)] = (rotation = SMatrix{3, 3, Float64, 9}(R[j]' * R[i]), inliers = length(common),
                                covariance = SMatrix{3, 3, Float64, 9}(1.0e-6, 0, 0, 0, 1.0e-6, 0, 0, 0, 1.0e-6))
            matches[(i, j)] = [(index[i][t], index[j][t]) for t in common]
            stats[(i, j)] = (matches = length(common), inliers = length(common), rms = σ, rotation_deg = 0.0)
        end
        # the planted wrong matches on `wrong_pair` and, the repeated structure of a near-duplicate view, on (dup, K + 1)
        planted = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
        frames(t) = Set(k for k in 1:N if haskey(index[k], t))
        for (i, j) in (wrong_pair, (dup, N))
            as = shuffle(rng, [t for t in keys(index[i]) if !haskey(index[j], t)])
            bs = shuffle(rng, [t for t in keys(index[j]) if !haskey(index[i], t)])
            used = Set{Int}(); planted[(i, j)] = Tuple{Int, Int}[]
            for a in as, b in bs                                              # a and b in no common frame: the chain survives
                length(planted[(i, j)]) < nwrong || break
                (a in used || b in used || !isempty(intersect(frames(a), frames(b)))) && continue
                push!(planted[(i, j)], (a, b)); push!(used, a, b)
            end
            append!(matches[(i, j)], [(index[i][a], index[j][b]) for (a, b) in planted[(i, j)]])
        end
        scales = Dict(p => (sigma = σ, dof = 2 * verified[p].inliers - 3) for p in keys(verified))
        vp = VerifiedPairs(feats, verified, matches, stats, scales, (; descriptor = (;), matcher = (;)))
        return (; vp, R, ts, index, planted, N, dup)
    end

    @testset "near-duplicates" begin
        s = planted_sweep(MersenneTwister(11); nwrong = 40)
        bcam = VgtkCore.BoundedCamera(CAM, VgtkCore.Size2(1200, 1200))
        kw = (; frame_map = IDENTITY_MAP, averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100),
              bundle_max_iter = 1000, bundle_ftol = 1.0e-12, bundle_xtol = 1.0e-6, check_tracks = false)
        # an observation is (frame, detection); a planted match joins detection a of frame i and b of frame j
        together(ts, (i, a), (j, b)) = any(t -> (i, a) in t && (j, b) in t, ts)
        joined(reg, p) = count(((a, b),) -> together(reg.tracks, (p[1], s.index[p[1]][a]), (p[2], s.index[p[2]][b])), s.planted[p])
        dp = (s.dup, s.N)
        plain = register_rotations(s.vp, bcam; kw...)
        @test isempty(plain.near_duplicates) && joined(plain, dp) > 0             # chained, the near-duplicate's wrong matches join tracks
        near = register_rotations(s.vp, bcam; kw..., near_duplicate_deg = 4.0)
        @test near.near_duplicates == [dp]                                       # 1° apart: the one near-duplicate pair
        @test joined(near, dp) == 0                                              # its matches seed no track
        @test haskey(near.graph, dp...)                                          # still an edge of the averaging
        @test all(maximum(abs, SMatrix{3, 3}(near.rotations[k]) - SMatrix{3, 3}(s.R[k])) < 1.0e-3 for k in 1:s.N)   # frame 1 = I: the common frame
        @test_throws ArgumentError register_rotations(s.vp, bcam; kw..., near_duplicate_deg = 0.0)
        @test_throws ArgumentError register_rotations(s.vp, bcam; kw..., near_duplicate_deg = NaN)
    end

    @testset "the track check" begin
        s = planted_sweep(MersenneTwister(12); nwrong = 60)
        bcam = VgtkCore.BoundedCamera(CAM, VgtkCore.Size2(1200, 1200))
        kw = (; frame_map = IDENTITY_MAP, averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100),
              bundle_max_iter = 1000, bundle_ftol = 1.0e-12, bundle_xtol = 1.0e-6, near_duplicate_deg = 4.0)
        off = register_rotations(s.vp, bcam; kw..., check_tracks = false)
        on = register_rotations(s.vp, bcam; kw..., check_tracks = true)
        @test isempty(off.track_check)
        h = on.track_check
        @test length(h) >= 2 && h[1].removed > 0 && h[end].removed == 0           # removed, re-solved, until none is removed
        @test h[1].rms == off.bundle.rms && h[1].inlier_rms == off.bundle.metadata.inlier_rms   # the first solve is the plain bundle
        @test off.bundle.rms > 5 * off.bundle.metadata.inlier_rms                 # the wrong chains' far cluster in the plain rms
        @test on.bundle.rms < off.bundle.rms / 5                                  # removed: the rms drops to the inliers'
        @test on.bundle.metadata.inlier_rms < 1.05 * off.bundle.metadata.inlier_rms   # the inliers barely move (the cut sits by the loss width)
        @test on.bundle.rms ≈ h[end].rms && on.bundle.metadata.inlier_rms ≈ h[end].inlier_rms
        # every residual of the final solve within the cut; the inlier rms is over those within the loss width
        cut = 3.035 * on.bundle_sigma
        @test all(r -> !(r > cut), reduce(vcat, on.bundle.metadata.residuals))
        flat = filter(isfinite, reduce(vcat, on.bundle.metadata.residuals))
        @test on.bundle.metadata.inlier_rms ≈ sqrt(sum(abs2, filter(≤(on.bundle.metadata.loss_width), flat)) / count(≤(on.bundle.metadata.loss_width), flat))
        # the planted wrong matches: no track keeps both of a planted pair's observations
        p = (1, 2)
        @test count(((a, b),) -> any(t -> (1, s.index[1][a]) in t && (2, s.index[2][b]) in t, on.tracks), s.planted[p]) == 0
        # the rotations: the true ones (frame 1 = I: the common frame), to the noise
        @test all(maximum(abs, SMatrix{3, 3}(on.rotations[k]) - SMatrix{3, 3}(s.R[k])) < 1.0e-3 for k in 1:s.N)
    end
end
