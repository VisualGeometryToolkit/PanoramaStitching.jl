# =============================================================================
# The registration's near-duplicate pairs and track check: a near-duplicate pair (its frames' shared view, VisualGeometryToolkitCore's
# view_overlap under the averaged rotations, at least near_duplicate_overlap of the smaller footprint) verifies and is averaged
# but seeds no track; after the bundle,
# the observations past 3.035 σ̂ are removed and the bundle re-solved until none is (COLMAP's filter_max_reproj_error loop).
# =============================================================================

@testitem "register_rotations: a near-duplicate pair chains no track; the track check removes a planted wrong chain past 3.035 σ̂, the rms drops" setup = [SweepScene, BayerMaps] begin
    using Test, Random, LinearAlgebra, StaticArrays
    import Rotations
    using PanoramaStitching: PanoramaStitching, register_rotations, VerifiedPairs
    import VisualGeometryToolkitCore as VgtkCore
    # K sweep frames 0.5 rad apart and frame K + 1 a near-duplicate of frame `dup` (1° off); tracks of world directions (σ px),
    # within 70° of a frame's axis: the bounded camera's lens coverage (BCAM70, a circle 70.6° off axis), so a frame's footprint
    # shares about 0.77 of itself with the next frame's (0.5 rad on) and about 0.99 with its near-duplicate's.
    # Planted: `nwrong` wrong chains — a true track a joined, by one wrong match on a verified pair, to a ghost track (a second
    # world direction's detections in two or more frames that do not see a: their own consistent cluster) — and, the repeated
    # structure of a near-duplicate view, wrong matches on (dup, K + 1) between true tracks seen in no frame together.
    function planted_sweep(rng; K = 5, dup = 3, nwrong = 40, σ = 0.4)
        R = Dict(k => (k == 1 ? SMatrix{3, 3}(1.0I) :
                       SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec(0.05randn(rng), 0.5 * (k - 1), 0.05randn(rng))))) for k in 1:K)
        R[K + 1] = R[dup] * SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec(0.0, deg2rad(1.0), 0.0)))
        ts, _, _ = sweep_tracks(rng, R, 600; σ, θmax = deg2rad(70))
        N = K + 1
        pixels = [SVector{2, Float64}[] for _ in 1:N]; index = [Dict{Int, Int}() for _ in 1:N]
        for (t, obs) in enumerate(ts), (k, u) in obs
            push!(pixels[k], u); index[k][t] = length(pixels[k])
        end
        frames(t) = Set(k for k in 1:N if haskey(index[k], t))
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
        link!(i, a, j, b) = i < j ? push!(matches[(i, j)], (a, b)) : push!(matches[(j, i)], (b, a))
        # the wrong chains: (frame of a, a's detection) ↔ (frame of the ghost, the ghost's detection)
        chains = Tuple{Tuple{Int, Int}, Tuple{Int, Int}}[]; chained = Set{Int}()
        while length(chains) < nwrong
            a = rand(rng, eachindex(ts)); a in chained && continue; Fa = frames(a)
            d = normalize(randn(rng, SVector{3, Float64}))
            G = [k for k in 1:N if k ∉ Fa && acos(clamp((R[k]' * d)[3], -1, 1)) < deg2rad(70)]
            links = [(i, j) for i in Fa for j in G if haskey(verified, minmax(i, j))]
            (length(G) >= 2 && !isempty(links)) || continue
            ghost = Dict(k => (push!(pixels[k], VgtkCore.project(CAM, R[k]' * d) + σ * randn(rng, SVector{2, Float64})); length(pixels[k])) for k in G)
            for k1 in G, k2 in G
                k1 < k2 && haskey(verified, (k1, k2)) && link!(k1, ghost[k1], k2, ghost[k2])
            end
            i, j = rand(rng, links)
            link!(i, index[i][a], j, ghost[j]); push!(chains, ((i, index[i][a]), (j, ghost[j]))); push!(chained, a)
        end
        # the near-duplicate's repeated structure: true tracks seen in no frame together, joined on (dup, N)
        planted = Tuple{Int, Int}[]; used = Set{Int}()
        for a in shuffle(rng, [t for t in keys(index[dup]) if !haskey(index[N], t)]), b in shuffle(rng, [t for t in keys(index[N]) if !haskey(index[dup], t)])
            length(planted) < nwrong || break
            (a in used || b in used || !isempty(intersect(frames(a), frames(b)))) && continue
            push!(planted, (a, b)); push!(used, a, b); link!(dup, index[dup][a], N, index[N][b])
        end
        feats = [described(pixels[k]) for k in 1:N]
        scales = Dict(p => (sigma = σ, dof = 2 * verified[p].inliers - 3) for p in keys(verified))
        vp = VerifiedPairs(feats, verified, matches, stats, scales, (; descriptor = (;), matcher = (;)))
        return (; vp, R, index, chains, planted, N, dup)
    end

    BCAM70 = VgtkCore.BoundedCamera(CAM, VgtkCore.Size2(1200, 1200); lens_coverage = VgtkCore.Circle(VgtkCore.Point2(600.0, 600.0), 360.0))

    @testset "near-duplicates" begin
        s = planted_sweep(MersenneTwister(11); nwrong = 40)
        bcam = BCAM70
        kw = (; frame_map = IDENTITY_MAP, averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100),
              bundle_max_iter = 1000, bundle_ftol = 1.0e-12, bundle_xtol = 1.0e-6, check_tracks = false)
        # an observation is (frame, detection); a planted match joins detection a of frame i and b of frame j
        together(ts, (i, a), (j, b)) = any(t -> (i, a) in t && (j, b) in t, ts)
        joined(reg) = count(((a, b),) -> together(reg.tracks, (s.dup, s.index[s.dup][a]), (s.N, s.index[s.N][b])), s.planted)
        dp = (s.dup, s.N)
        plain = register_rotations(s.vp, bcam; kw..., near_duplicate_overlap = 0.0)   # 0: the check off, every kept pair chained
        @test isempty(plain.near_duplicates) && joined(plain) > 0             # chained, the near-duplicate's wrong matches join tracks
        near = register_rotations(s.vp, bcam; kw...)                          # always on: near_duplicate_overlap = 0.90 by default
        @test near.near_duplicates == [dp]                                       # 1° apart: the one near-duplicate pair
        @test PanoramaStitching._NEAR_DUPLICATE_OVERLAP == 0.90                  # the owner's value (owner ruling 2026-11-14)
        @test register_rotations(s.vp, bcam; kw..., near_duplicate_overlap = 0.90).near_duplicates == [dp]
        @test joined(near) == 0                                              # its matches seed no track
        @test haskey(near.graph, dp...)                                          # still an edge of the averaging
        @test all(maximum(abs, SMatrix{3, 3}(near.rotations[k]) - SMatrix{3, 3}(s.R[k])) < 1.0e-3 for k in 1:s.N)   # frame 1 = I: the common frame
        # a fraction of the smaller footprint: 0 (off) to 1; another is refused, and so is the retired angle, naming its successor
        for bad in (-0.1, 1.5, NaN)
            @test_throws "near_duplicate_overlap" register_rotations(s.vp, bcam; kw..., near_duplicate_overlap = bad)
        end
        @test_throws r"`near_duplicate_deg` is retired: .*near_duplicate_overlap" register_rotations(s.vp, bcam; kw..., near_duplicate_deg = 4.0)
    end

    @testset "the track check" begin
        s = planted_sweep(MersenneTwister(12); nwrong = 60)
        bcam = BCAM70
        kw = (; frame_map = IDENTITY_MAP, averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100),
              bundle_max_iter = 1000, bundle_ftol = 1.0e-12, bundle_xtol = 1.0e-6)
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
        @test count(((o, g),) -> any(t -> o in t && g in t, off.tracks), s.chains) == length(s.chains)   # chained without the check
        @test count(((o, g),) -> any(t -> o in t && g in t, on.tracks), s.chains) == 0                   # each chain removed
        # the rotations: the true ones (frame 1 = I: the common frame), to the noise
        @test all(maximum(abs, SMatrix{3, 3}(on.rotations[k]) - SMatrix{3, 3}(s.R[k])) < 1.0e-3 for k in 1:s.N)
    end
end

@testitem "register_rotations: the near-duplicate rule is view_overlap's fraction of the smaller footprint" begin
    # frames of a pinhole 80° × 37° wide (±40° in yaw) at yaw 0°, 2°, 40° and 41.5° (frame 4 a near-duplicate of frame 3):
    # the pairs 1-2 and 3-4 share ≥ 0.90 of a footprint (VisualGeometryToolkitCore's view_overlap), 1-3 and 2-4 about half; the
    # threshold inclusive of the shared fraction, 0 the check off
    using Test, StaticArrays
    using PanoramaStitching: PanoramaStitching
    import VisualGeometryToolkitCore as VgtkCore
    cam = VgtkCore.BoundedCamera(VgtkCore.CameraModel((500.0 / tand(40.0))VgtkCore.px, SA[500.0VgtkCore.px, 200.0VgtkCore.px];
                                                      lens = VgtkCore.Pinhole()), VgtkCore.Size2(1000, 400))
    A = [SMatrix{3, 3, Float64, 9}(VgtkCore.exp_so3(SVector(0.0, deg2rad(y), 0.0))) for y in (0.0, 2.0, 40.0, 41.5)]
    pairs = [(1, 2), (1, 3), (2, 4), (3, 4)]
    @test PanoramaStitching._near_duplicate_pairs(cam, A, pairs, 0.90) == [(1, 2), (3, 4)]
    @test PanoramaStitching._near_duplicate_pairs(cam, A, pairs, 0.0) == Tuple{Int, Int}[]    # 0: no pair is a near-duplicate
    @test PanoramaStitching._near_duplicate_pairs(cam, A, pairs, 0.30) == pairs               # every pair shares ≥ 0.30
    rows = VgtkCore.view_overlap([PanoramaStitching._oriented_frame(cam, A[k]) for k in 1:4])
    @test all(r -> r.fraction_i ≈ r.fraction_j, rows)                                     # one camera: one footprint size
    @test only(r.fraction_i for r in rows if (r.i, r.j) == (1, 2)) > 0.90 > only(r.fraction_i for r in rows if (r.i, r.j) == (1, 3))
end
