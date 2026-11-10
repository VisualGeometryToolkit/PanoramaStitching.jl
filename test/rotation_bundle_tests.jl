# =============================================================================
# The registration structure (src/view_graph.jl, src/rotation_bundle.jl): the view graph refuses a disconnected sequence, the tracks
# chain matches and drop the inconsistent ones, and the rotation bundle recovers the rotations and the directions of a synthetic
# fisheye sweep past 90° with outliers; its Jacobian blocks against ForwardDiff; the eliminated and the joint solve agree.
# =============================================================================

@testsetup module SweepScene
using Random, LinearAlgebra, StaticArrays
import Rotations
using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, px, project
export CAM, sweep_rotations, sweep_tracks
const CAM = CameraModel(300.0px, SA[600.0px, 600.0px]; lens = KannalaBrandt(-0.02))
sweep_rotations(rng, K) = Dict(k => (k == 1 ? SMatrix{3, 3}(1.0I) :
    SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec(0.1randn(rng), 0.5 * (k - 1) + 0.05randn(rng), 0.1randn(rng))))) for k in 1:K)
# tracks of world directions seen within 100° of the axes of ≥ 2 frames: (frame => pixel) observations, σ px noise; a
# fraction `outl` of the observations moved to a random pixel
function sweep_tracks(rng, R, n; σ = 0.5, outl = 0.0, θmax = deg2rad(100))
    K = length(R); out = Vector{Vector{Pair{Int, SVector{2, Float64}}}}(); dirs = SVector{3, Float64}[]; past90 = 0
    while length(out) < n
        d = normalize(randn(rng, SVector{3, Float64}))
        t = Pair{Int, SVector{2, Float64}}[]
        for k in 1:K
            c = R[k]' * d
            acos(clamp(c[3], -1, 1)) < θmax || continue
            u = project(CAM, c); all(isfinite, u) || continue
            c[3] < 0 && (past90 += 1)
            x = rand(rng) < outl ? SVector(600 + 400 * (2rand(rng) - 1), 600 + 400 * (2rand(rng) - 1)) : u + σ * randn(rng, SVector{2, Float64})
            push!(t, k => x)
        end
        length(t) >= 2 || continue
        push!(out, t); push!(dirs, d)
    end
    return out, dirs, past90
end
end

@testitem "ViewGraph and chain_tracks" begin
    using Test, StaticArrays, LinearAlgebra
    using PanoramaStitching: ViewGraph, chain_tracks
    e(R) = (rotation = R, inliers = 10)
    I3 = SMatrix{3, 3}(1.0I)
    g = ViewGraph([1, 2, 3], Dict((1, 2) => e(I3), (2, 3) => e(I3)))
    @test g.inliers[(1, 2)] == 10
    @test_throws ArgumentError ViewGraph([1, 2, 3, 4], Dict((1, 2) => e(I3), (3, 4) => e(I3)))
    # 1:5 ↔ 2:7 ↔ 3:1 chains into one track; 1:6 ↔ 2:8 and 2:8 ↔ 3:2 and 3:2 ↔ 1:9 put two keypoints of frame 1 in one track: dropped
    ts, dropped = chain_tracks(Dict((1, 2) => [(5, 7), (6, 8)], (2, 3) => [(7, 1), (8, 2)], (1, 3) => [(9, 2)]))
    @test ts == [[(1, 5), (2, 7), (3, 1)]]
    @test dropped == 1
end

@testitem "refine_rotations: a fisheye sweep past 90°, outliers; Jacobian; eliminated = joint" setup = [SweepScene] begin
    using Test, Random, LinearAlgebra, StaticArrays, Statistics
    import Rotations, ForwardDiff, NLLSsolver
    using PanoramaStitching: refine_rotations
    import PanoramaStitching
    import VisualGeometryToolkitCore as VgtkCore

    rng = Xoshiro(31)
    K = 5
    R = sweep_rotations(rng, K)
    ts, dirs, past90 = sweep_tracks(rng, R, 400; outl = 0.05)
    @test past90 > 0
    start = Dict(k => SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec((0.02 .* randn(rng, 3))...))) * R[k] for k in 1:K)
    start[1] = R[1]
    err(Rs) = maximum(k -> rad2deg(Rotations.rotation_angle(Rotations.RotMatrix{3}(SMatrix{3, 3}(Rs[k])' * R[k]))), 1:K)
    @test err(start) > 0.5
    kw = (; loss = VgtkCore.CauchyShape(), scale_multiple = 3.0, iterations = 2, max_iter = 50, ftol = 1.0e-12, xtol = 1.0e-10)
    res = refine_rotations(CAM, start, [[k => x for (k, x) in t] for t in ts]; kw...)
    @test err(VgtkCore.estimate(res)) < 0.05                       # the noise floor: 0.5 px at f = 300 px (0.19°/px), ~400 tracks
    @test VgtkCore.estimate(res)[1] ≈ R[1]
    good = [maximum(norm.(r)) < 5 for r in res.metadata.residuals]
    @test count(good) > 300
    derr = [rad2deg(acos(clamp(dot(res.metadata.directions[j], dirs[j]), -1, 1))) for j in findall(good)]
    @test median(derr) < 0.1 && maximum(derr) < 0.5                  # a track of 2–5 observations at 0.5 px
    # the joint solve (no elimination) reaches the same optimum
    res2 = refine_rotations(CAM, start, [[k => x for (k, x) in t] for t in ts]; kw..., eliminate = false)
    @test maximum(k -> norm(SMatrix{3, 3}(VgtkCore.estimate(res)[k]) - SMatrix{3, 3}(VgtkCore.estimate(res2)[k])), 1:K) < 1.0e-8
    # the closed-form Jacobian blocks against ForwardDiff through the variables' updates
    ch = VgtkCore.SO3Manifold(SMatrix{3, 3, Float64, 9}(R[2])); dv = VgtkCore.Eliminable{false}(VgtkCore.SphereManifold(dirs[1]))
    x = VgtkCore.project(CAM, SMatrix{3, 3}(R[2])' * dirs[1]) + SVector(0.3, -0.2)
    fwd = PanoramaStitching._forward_map(CAM)
    bc = PanoramaStitching._BundleCamera(CAM)
    obs = PanoramaStitching._Observation{false, typeof(bc), typeof(fwd), Nothing, Nothing}(bc, fwd, x, SVector(1, 2), nothing, nothing)
    _, J = NLLSsolver.computeresjac(NLLSsolver.static(3), obs, ch, dv)
    f(δ) = NLLSsolver.computeresidual(obs, NLLSsolver.update(ch, δ[1:3]), NLLSsolver.update(dv, δ[4:5]))
    @test J ≈ ForwardDiff.jacobian(f, zeros(5)) rtol = 1.0e-7
end

# Observations whose start prediction is not imaged (off the grid lens's lattice: no projection, no back-projection Jacobian)
# are dropped and counted, and the bundle solves the rest: 5 tracks each gain an observation in a frame where the track's
# direction images past the sensor's right edge (recorded at the edge: a detection at the border of a frame the direction
# leaves). The bundle drops exactly those 5, converges, and its rotations are those of the same bundle without them to 1e-9;
# when every track is left with fewer than two frames it is an ArgumentError.
@testitem "refine_rotations drops and counts the observations whose start prediction is not imaged" begin
    using Test, Random, LinearAlgebra, StaticArrays
    import PanoramaStitching
    using PanoramaStitching: refine_rotations
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    rot(v) = SMatrix{3, 3, Float64, 9}(VgtkCore.exp_so3(SVector{3, Float64}(v)))
    W, H = 1000, 800
    pinhole = VgtkCore.CameraModel(800.0VgtkCore.px, SA[500.0VgtkCore.px, 400.0VgtkCore.px])
    bounded = VgtkCore.BoundedCamera(pinhole, VgtkCore.Size2(W, H))
    grid = VgtkCore.with_model(bounded, VgtkCore.estimate(VgtkEstimation.fit_camera(VgtkCore.SplineLens, bounded, VgtkCore.imaged_region(bounded);
                                                                                    tolerance = 0.01, grid_size = (20, 16))))
    rng = Xoshiro(11); K = 4
    R = Dict(k => rot(SVector(0.02randn(rng), 0.25 * (k - 1), 0.02randn(rng))) for k in 1:K)
    inside(x) = 20 < x[1] < W - 20 && 20 < x[2] < H - 20
    gaining = Vector{Vector{Pair{Int, SVector{2, Float64}}}}(); extra = Pair{Int, SVector{2, Float64}}[]
    others = Vector{Vector{Pair{Int, SVector{2, Float64}}}}()
    while length(others) < 295 || length(extra) < 5
        d = normalize(randn(rng, SVector{3, Float64})); t = Pair{Int, SVector{2, Float64}}[]; past = nothing
        for k in 1:K
            c = R[k]' * d; c[3] > 0 || continue
            u = VgtkCore.project(pinhole, c)
            if inside(u)
                push!(t, k => u + 0.3randn(rng, SVector{2, Float64}))
            elseif W + 30 < u[1] < W + 60 && 20 < u[2] < H - 20
                past = k => SVector(W - 2.0, u[2])          # imaged 30–60 px past the right edge, recorded at the edge
            end
        end
        length(t) >= 2 || continue
        if past !== nothing && length(extra) < 5
            push!(extra, past); push!(gaining, t)
        elseif length(others) < 295
            push!(others, t)
        end
    end
    tracks = [gaining; others]
    with_extra = [j <= 5 ? [tracks[j]; extra[j]] : tracks[j] for j in eachindex(tracks)]   # the first 5 tracks gain one each
    start = Dict(k => (k == 1 ? R[1] : rot(0.003 .* randn(rng, 3)) * R[k]) for k in 1:K)
    kw = (; loss = VgtkCore.CauchyShape(), scale_multiple = 3.0, iterations = 3, max_iter = 100, ftol = 1.0e-10, xtol = 1.0e-10)
    res = refine_rotations(grid, start, with_extra; kw...)
    ref = refine_rotations(grid, start, tracks; kw...)
    @test res.metadata.outcome === :converged && res.metadata.n_dropped_start == 5
    @test ref.metadata.outcome === :converged && ref.metadata.n_dropped_start == 0
    @test all(j -> isnan(res.metadata.residuals[j][end]), 1:5)                         # a dropped observation has no residual
    @test maximum(k -> norm(SMatrix{3, 3}(VgtkCore.estimate(res)[k]) - SMatrix{3, 3}(VgtkCore.estimate(ref)[k])), 1:K) <= 1.0e-9
    @test maximum(k -> rad2deg(norm(VgtkCore.log_so3(SMatrix{3, 3}(VgtkCore.estimate(res)[k])' * R[k]))), 1:K) < 0.05
    # every track a pair whose second observation is not imaged at the start: nothing is left to solve
    lone = [[first(t), extra[j]] for (j, t) in enumerate(tracks[1:5]) if first(first(t)) != first(extra[j])]
    @test !isempty(lone)
    @test_throws ArgumentError refine_rotations(grid, start, lone; kw...)
end

# The bundle's outcome: the Tamron 20 mm sequence's mechanism on a synthetic sequence — a rectilinear grid lens calibrated over
# the sensor (a SplineLens camera, its forward spline over the same region), observations up to the frame's border, a start
# ~0.6° off (an averaging's rotations). A start prediction near the border falls outside the grid's domain, where the grid
# has no back-projection Jacobian; until plan 2 that made the whole start non-finite and NLLSsolver stopped at once
# ("nonfinite_start", the rotations the start's). Now those observations are dropped and counted, and the bundle converges.
# Controls: the same data and start through the parametric camera converge with none dropped, as does the grid from the true
# rotations, and no iteration is "not_run".
@testitem "refine_rotations' outcome: a grid lens's border observations from an imperfect start converge (their misses dropped)" begin
    using Test, Random, LinearAlgebra, StaticArrays
    import NLLSsolver, PanoramaStitching
    using PanoramaStitching: refine_rotations
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    rot(v) = SMatrix{3, 3, Float64, 9}(VgtkCore.exp_so3(SVector{3, Float64}(v)))
    W, H = 1000, 800
    pinhole = VgtkCore.CameraModel(800.0VgtkCore.px, SA[500.0VgtkCore.px, 400.0VgtkCore.px])
    # the grid: a 20 × 16 SplineLens fitted to the pinhole over the sensor, with its forward spline; bounded by the sensor
    bounded = VgtkCore.BoundedCamera(pinhole, VgtkCore.Size2(W, H))
    grid = VgtkCore.with_model(bounded, VgtkCore.estimate(VgtkEstimation.fit_camera(VgtkCore.SplineLens, bounded, VgtkCore.imaged_region(bounded);
                                                                                    tolerance = 0.01, grid_size = (20, 16))))
    sp = PanoramaStitching._forward_map(grid)
    rng = Xoshiro(7); K = 5
    R = Dict(k => rot(SVector(0.02randn(rng), 0.3 * (k - 1), 0.02randn(rng))) for k in 1:K)
    tracks = Vector{Vector{Pair{Int, SVector{2, Float64}}}}()
    while length(tracks) < 400
        d = normalize(randn(rng, SVector{3, Float64})); t = Pair{Int, SVector{2, Float64}}[]
        for k in 1:K
            c = R[k]' * d; c[3] > 0 || continue
            u = sp(c); all(isfinite, u) || continue
            x = SVector(u[1], u[2]) + 0.5randn(rng, SVector{2, Float64})
            (2 < x[1] < W - 2 && 2 < x[2] < H - 2) && push!(t, k => x)
        end
        length(t) >= 2 && push!(tracks, t)
    end
    start = Dict(k => (k == 1 ? R[1] : rot(0.01 .* randn(rng, 3)) * R[k]) for k in 1:K)
    kw = (; loss = VgtkCore.CauchyShape(), scale_multiple = 3.0, iterations = 3, max_iter = 100, ftol = 1.0e-10, xtol = 1.0e-10)
    err(Rs) = maximum(k -> rad2deg(norm(VgtkCore.log_so3(SMatrix{3, 3}(Rs[k])' * R[k]))), 1:K)

    res = refine_rotations(grid, start, tracks; kw...)
    @info "the grid from an imperfect start: $(res.metadata.n_dropped_start) of $(res.metadata.n_observations) observations dropped"
    @test res.metadata.outcome === :converged && res.metadata.n_dropped_start > 0 && err(VgtkCore.estimate(res)) < 0.05
    resp = refine_rotations(pinhole, start, tracks; kw...)
    @test resp.metadata.outcome === :converged && resp.metadata.n_dropped_start == 0 && err(VgtkCore.estimate(resp)) < 0.05
    # The grid from the TRUE rotations (a finite start) converges. Until 2026-11-04 it never returned: the grid's forward map
    # (then a separate spline) extrapolated past the grid's imaged region, so an iterate there had a finite residual but a NaN Jacobian, NLLSsolver
    # accepted it and its Levenberg–Marquardt trial loop spun on the NaN system. The residual is now NaN off the imaged region
    # too (`_project`): such a step is a rejected one.
    rest = refine_rotations(grid, R, tracks; kw...)
    @test rest.metadata.outcome === :converged && rest.metadata.n_dropped_start == 0 && err(VgtkCore.estimate(rest)) < 0.05
    # the residual and its Jacobian are defined on the same set: off the imaged region both are NaN
    fwd = sp
    bc = PanoramaStitching._BundleCamera(grid)
    o = PanoramaStitching._Observation{false, typeof(bc), typeof(fwd), Nothing, Nothing}(bc, fwd, SVector(500.0, 400.0), SVector(1, 2), nothing, nothing)
    ch = VgtkCore.SO3Manifold(SMatrix{3, 3, Float64, 9}(I)); far = VgtkCore.Eliminable{false}(VgtkCore.SphereManifold(SVector(0.9, 0.0, 0.3)))
    e, J = NLLSsolver.computeresjac(NLLSsolver.static(3), o, ch, far)
    @test all(isnan, e) && all(isnan, J) && all(isnan, NLLSsolver.computeresidual(o, ch, far))
    @test all(isnan, sp(SVector(0.9, 0.0, 0.3)))                      # (the forward spline is NaN off its fitted chart too)
    @test refine_rotations(pinhole, start, tracks; kw..., iterations = 0).metadata.outcome === :not_run
end
