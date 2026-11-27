# The pair equation on the suite's synthetic sweep (VignettingSequence: I_k = L(R_k·f(u))·V(u) at every pixel centre through a
# Kannala–Brandt fisheye, six frames in yaw and pitch): a V constant on rings recovered exactly by its ring-pair medians, a smooth
# radial V to the ring's spread, an additive level found and removed, and the symmetric saturation mask dropping a clipped
# direction from every frame.

@testitem "RingPairMedians: a ring-constant V recovered, a smooth V to its ring spread, the gauge ln V₁ = 0" setup = [VignettingSequence, BayerMaps] begin
    using Test, Statistics, StaticArrays
    using PanoramaStitching: ring_correspondences, RingVignetting, RingPairMedians, fibonacci_directions
    import VisualGeometryToolkitCore as VgtkCore
    medians(pc, p = VgtkCore.NullParameters()) = VgtkCore.solve(VgtkCore.EstimationProblem(RingVignetting, pc, p), RingPairMedians())
    c = SVector(400.0, 400.0)
    radius(u) = hypot(u[1] - c[1], u[2] - c[2])
    edges = collect(0.0:40.0:360.0)                                          # rings of the radius (px) about the centre
    ring_logV(u) = (q = clamp(searchsortedlast(edges, radius(u)), 1, length(edges) - 1); -0.04 * (q - 1)^1.5)
    q = vignetting_sequence(ring_logV)
    D = fibonacci_directions(q.R, 40000; fov_deg = 120.0)
    kw = (; saturated = q.saturated, saturation_margin_px = 2.0, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2, field_angle = radius, edges)
    pc = ring_correspondences(q.camera, q.frames, q.R, D; kw...)
    @test length(pc) > 10000 && pc.n_frames == 6 && pc.n_dropped == 0
    @test all(pc.rings[1, :] .< pc.rings[2, :])
    r = medians(pc)
    x = r.u; truth = [-0.04 * (j - 1)^1.5 for j in 1:(length(edges) - 1)]
    ok = isfinite.(x)
    @test (r.retcode, r.stop_rule) == (VgtkCore.ReturnCode.Success, VgtkCore.ClosedForm) && x[1] == 0.0 && count(ok) >= 7
    @test maximum(abs, x[ok] .- truth[ok]) < 2e-3                           # the medians are interior values: exact to the frames' Float32
    # a smooth radial V: each ring's value lies within the ring's own range of the truth
    smooth(u) = -0.8 * (radius(u) / 400)^2
    q2 = vignetting_sequence(smooth)
    r2 = medians(ring_correspondences(q2.camera, q2.frames, q2.R, D; kw..., saturated = q2.saturated))
    x2 = r2.u
    ref(j) = -0.8 * ((edges[j] + edges[j + 1]) / 2 / 400)^2 + 0.8 * (20 / 400)^2   # the ring's mid value relative to the first ring's
    for j in findall(isfinite, x2)
        @test abs(x2[j] - ref(j)) < 0.8 * ((edges[j + 1] / 400)^2 - (edges[j] / 400)^2) / 2 + 0.01
    end
    # frame weights: a frame weighted twice changes nothing on exact data; all-zero weights leave nothing
    w = ones(6); w[3] = 2.0
    @test maximum(abs, filter(isfinite, medians(pc, (; frame_weights = w)).u .- x)) < 2e-3
    @test (rz = medians(pc, (; frame_weights = zeros(6))); (rz.retcode, rz.stop_rule) == (VgtkCore.ReturnCode.InitialFailure, VgtkCore.NoRule))
    @test_throws DimensionMismatch medians(pc, (; frame_weights = ones(5)))
end

@testitem "LevelBisection: the additive level found and removed; the symmetric saturation mask" setup = [VignettingSequence, BayerMaps] begin
    using Test, Statistics, StaticArrays
    using PanoramaStitching: ring_correspondences, RingVignetting, RingPairMedians, AdditiveLevel, LevelBisection, additive_level_slope, fibonacci_directions
    import VisualGeometryToolkitCore as VgtkCore
    medians(pc, p = VgtkCore.NullParameters()) = VgtkCore.solve(VgtkCore.EstimationProblem(RingVignetting, pc, p), RingPairMedians())
    c = SVector(400.0, 400.0)
    radius(u) = hypot(u[1] - c[1], u[2] - c[2])
    edges = collect(0.0:40.0:360.0)
    smooth(u) = -0.8 * (radius(u) / 400)^2
    q = vignetting_sequence(smooth)
    D = fibonacci_directions(q.R, 40000; fov_deg = 120.0)
    kw = (; saturation_margin_px = 2.0, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2, field_angle = radius, edges)
    level = 0.004                                                            # ≈ 10 % of the scene's median value
    lifted = [F .+ Float32(level) for F in q.frames]
    x0 = medians(ring_correspondences(q.camera, q.frames, q.R, D; kw..., saturated = q.saturated)).u
    pc = ring_correspondences(q.camera, lifted, q.R, D; kw..., saturated = q.saturated)
    flat = medians(pc).u
    ok = findall(isfinite, x0)
    @test flat[ok[end]] > x0[ok[end]] + 0.02                                # the level flattens V at the rim
    @test additive_level_slope(pc, 0.0) > 0 && additive_level_slope(pc, 2level) < 0   # positive below the level, negative above
    f = VgtkCore.solve(VgtkCore.EstimationProblem(AdditiveLevel, pc), LevelBisection(; bounds = (-0.01, 0.009), tolerance = 1e-5))
    @test (f.retcode, f.stop_rule) == (VgtkCore.ReturnCode.Success, VgtkCore.ToleranceMet)
    @test abs(f.u - level) < 3e-4
    xr = f.metadata.vignetting.u
    @test maximum(abs, xr[ok] .- x0[ok]) < 5e-3
    @test_throws r"LevelBisection: no correspondence has both values above" VgtkCore.solve(VgtkCore.EstimationProblem(AdditiveLevel, pc), LevelBisection(; bounds = (0.0, 1.0e3), tolerance = 1e-5))   # no correspondence above the bound
    # saturation: one frame clipped over a patch; symmetric drops those directions from every frame, per-frame only from it
    sat = [falses(size(F, 1), size(F, 2)) for F in q.frames]; sat[2][150:250, 150:250] .= true
    ps = ring_correspondences(q.camera, q.frames, q.R, D; kw..., saturated = sat)
    pf = PanoramaStitching._ring_correspondences(q.camera, q.frames, q.R, D, false; kw..., saturated = sat)   # the per-frame rule (≤ 0.8)
    p0 = ring_correspondences(q.camera, q.frames, q.R, D; kw..., saturated = q.saturated)
    @test ps.n_dropped > 0 && pf.n_dropped == 0
    @test length(ps) < length(pf) < length(p0)
end

@testitem "ring_correspondences, RingPairMedians: the device agrees with the host" setup = [VignettingSequence, BayerMaps] begin
    using Test, StaticArrays, CUDA
    using PanoramaStitching: ring_correspondences, RingVignetting, RingPairMedians, fibonacci_directions
    import VisualGeometryToolkitCore as VgtkCore
    medians(pc, p = VgtkCore.NullParameters()) = VgtkCore.solve(VgtkCore.EstimationProblem(RingVignetting, pc, p), RingPairMedians())
    struct Radius
        c::SVector{2, Float64}
    end
    (r::Radius)(u) = hypot(u[1] - r.c[1], u[2] - r.c[2])
    if !CUDA.functional()
        @test_skip "no CUDA device"
    else
        radius = Radius(SVector(400.0, 400.0))                               # a kernel-lawful field angle (an isbits callable)
        edges = collect(0.0:40.0:360.0)
        q = vignetting_sequence(u -> -0.8 * (radius(u) / 400)^2)
        sat = [falses(size(F, 1), size(F, 2)) for F in q.frames]; sat[2][150:250, 150:250] .= true
        D = fibonacci_directions(q.R, 40000; fov_deg = 120.0)
        kw = (; saturation_margin_px = 2.0, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2, field_angle = radius, edges)
        h = ring_correspondences(q.camera, q.frames, q.R, D; saturated = sat, kw...)
        d = ring_correspondences(q.camera, CuArray.(q.frames), q.R, CuArray(D); saturated = CuArray.(sat), kw...)
        @test d.n_dropped == h.n_dropped && abs(length(d) - length(h)) <= length(h) ÷ 1000   # a tap on a ring's edge may round apart
        xh = medians(h).u; xd = medians(d).u
        @test isequal(isfinite.(xh), isfinite.(xd)) && maximum(abs, filter(isfinite, xh .- xd)) < 1e-4
    end
end

@testitem "ring_correspondences: one mask per frame; a clipped pixel never blended into a kept value. RingPairMedians: the gauge's ring unconnected" setup = [VignettingSequence, BayerMaps] begin
    using Test, StaticArrays
    using PanoramaStitching: ring_correspondences, RingVignetting, RingPairMedians, fibonacci_directions, PairCorrespondences
    import VisualGeometryToolkitCore as VgtkCore
    medians(pc, p = VgtkCore.NullParameters()) = VgtkCore.solve(VgtkCore.EstimationProblem(RingVignetting, pc, p), RingPairMedians())
    c = SVector(400.0, 400.0)
    radius(u) = hypot(u[1] - c[1], u[2] - c[2])
    edges = collect(0.0:40.0:360.0)
    q = vignetting_sequence(u -> -0.8 * (radius(u) / 400)^2)
    D = fibonacci_directions(q.R, 40000; fov_deg = 120.0)
    kw = (; frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2, field_angle = radius, edges)
    # one saturation mask per frame, on either path (too few: a frame's clipping silently untested; too many: a frame index past
    # the rotations in the kernel)
    for symmetric in (true, false), n in (5, 7)
        sat = [falses(size(q.frames[1], 1), size(q.frames[1], 2)) for _ in 1:n]
        @test_throws DimensionMismatch PanoramaStitching._ring_correspondences(q.camera, q.frames, q.R, D, symmetric; kw..., saturated = sat, saturation_margin_px = 2.0)
    end
    # frame 2's scattered clipped pixels (1 in 50) read 1000, the scene ≤ 0.2: with no dilation (0 px) a sample whose nearest
    # pixel is clear but whose bilinear taps reach a clipped one is dropped by the symmetric test as by the frame's own mask
    sat = [falses(size(F, 1), size(F, 2)) for F in q.frames]
    sat[2] .= [(7i + 13j) % 50 == 0 for i in axes(sat[2], 1), j in axes(sat[2], 2)]
    frames = [copy(F) for F in q.frames]
    for p in 1:3
        view(frames[2], :, :, p)[sat[2]] .= 1.0f3
    end
    for symmetric in (true, false)
        pc = PanoramaStitching._ring_correspondences(q.camera, frames, q.R, D, symmetric; kw..., saturated = sat, saturation_margin_px = 0.0)
        @test length(pc) > 1000 && maximum(pc.values) < 1.0
    end
    # the public signature has no per-frame rule
    @test_throws MethodError ring_correspondences(q.camera, q.frames, q.R, D; kw..., saturated = q.saturated, saturation_margin_px = 2.0, symmetric = false)
    # three rings, the only ring pair (2, 3): nothing is connected to the gauge's ring, so nothing is identified — not a success
    pc1 = PairCorrespondences([0.0, 1.0, 2.0, 3.0], 3.0, Int32[1 1; 2 2], Int32[2 2; 3 3], [1.0 2.0; 0.5 1.0], [1.5 1.5; 2.5 2.5], 2, 2, 0)
    r1 = medians(pc1)
    @test (r1.retcode, r1.stop_rule) == (VgtkCore.ReturnCode.InitialFailure, VgtkCore.NoRule) && all(isnan, r1.u[2:end])
end
