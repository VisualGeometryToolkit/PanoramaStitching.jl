# The pair equation on the suite's synthetic sweep (VignettingSequence: I_k = L(R_k·f(u))·V(u) at every pixel centre through a
# Kannala–Brandt fisheye, six frames in yaw and pitch): a V constant on rings recovered exactly by its ring-pair medians, a smooth
# radial V to the ring's spread, an additive level found and removed, and the symmetric saturation mask dropping a clipped
# direction from every frame.

@testitem "ring_pair_vignetting: a ring-constant V recovered, a smooth V to its ring spread, the gauge ln V₁ = 0" setup = [VignettingSequence, BayerMaps] begin
    using Test, Statistics, StaticArrays
    using PanoramaStitching: ring_correspondences, ring_pair_vignetting, fibonacci_directions
    import VisualGeometryToolkitCore as VgtkCore
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
    r = ring_pair_vignetting(pc)
    x = VgtkCore.estimate(r); truth = [-0.04 * (j - 1)^1.5 for j in 1:(length(edges) - 1)]
    ok = isfinite.(x)
    @test VgtkCore.retcode_symbol(r) === :solved && x[1] == 0.0 && count(ok) >= 7
    @test maximum(abs, x[ok] .- truth[ok]) < 2e-3                           # the medians are interior values: exact to the frames' Float32
    # a smooth radial V: each ring's value lies within the ring's own range of the truth
    smooth(u) = -0.8 * (radius(u) / 400)^2
    q2 = vignetting_sequence(smooth)
    r2 = ring_pair_vignetting(ring_correspondences(q2.camera, q2.frames, q2.R, D; kw..., saturated = q2.saturated))
    x2 = VgtkCore.estimate(r2)
    ref(j) = -0.8 * ((edges[j] + edges[j + 1]) / 2 / 400)^2 + 0.8 * (20 / 400)^2   # the ring's mid value relative to the first ring's
    for j in findall(isfinite, x2)
        @test abs(x2[j] - ref(j)) < 0.8 * ((edges[j + 1] / 400)^2 - (edges[j] / 400)^2) / 2 + 0.01
    end
    # frame weights: a frame weighted twice changes nothing on exact data; all-zero weights leave nothing
    w = ones(6); w[3] = 2.0
    @test maximum(abs, filter(isfinite, VgtkCore.estimate(ring_pair_vignetting(pc; frame_weights = w)) .- x)) < 2e-3
    @test VgtkCore.retcode_symbol(ring_pair_vignetting(pc; frame_weights = zeros(6))) === :degenerate
    @test_throws DimensionMismatch ring_pair_vignetting(pc; frame_weights = ones(5))
end

@testitem "fit_additive_level: the level found and removed; the symmetric saturation mask" setup = [VignettingSequence, BayerMaps] begin
    using Test, Statistics, StaticArrays
    using PanoramaStitching: ring_correspondences, ring_pair_vignetting, fit_additive_level, additive_level_slope, fibonacci_directions
    import VisualGeometryToolkitCore as VgtkCore
    c = SVector(400.0, 400.0)
    radius(u) = hypot(u[1] - c[1], u[2] - c[2])
    edges = collect(0.0:40.0:360.0)
    smooth(u) = -0.8 * (radius(u) / 400)^2
    q = vignetting_sequence(smooth)
    D = fibonacci_directions(q.R, 40000; fov_deg = 120.0)
    kw = (; saturation_margin_px = 2.0, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2, field_angle = radius, edges)
    level = 0.004                                                            # ≈ 10 % of the scene's median value
    lifted = [F .+ Float32(level) for F in q.frames]
    x0 = VgtkCore.estimate(ring_pair_vignetting(ring_correspondences(q.camera, q.frames, q.R, D; kw..., saturated = q.saturated)))
    pc = ring_correspondences(q.camera, lifted, q.R, D; kw..., saturated = q.saturated)
    flat = VgtkCore.estimate(ring_pair_vignetting(pc))
    ok = findall(isfinite, x0)
    @test flat[ok[end]] > x0[ok[end]] + 0.02                                # the level flattens V at the rim
    @test additive_level_slope(pc, 0.0) > 0 && additive_level_slope(pc, 2level) < 0   # positive below the level, negative above
    f = fit_additive_level(pc; bounds = (-0.01, 0.009), tolerance = 1e-5)
    @test VgtkCore.retcode_symbol(f) === :converged
    @test abs(VgtkCore.estimate(f) - level) < 3e-4
    xr = VgtkCore.estimate(VgtkCore.metadata(f).vignetting)
    @test maximum(abs, xr[ok] .- x0[ok]) < 5e-3
    @test_throws ArgumentError fit_additive_level(pc; bounds = (0.0, 1.0e3), tolerance = 1e-5)   # no correspondence above the bound
    # saturation: one frame clipped over a patch; symmetric drops those directions from every frame, per-frame only from it
    sat = [falses(size(F, 1), size(F, 2)) for F in q.frames]; sat[2][150:250, 150:250] .= true
    ps = ring_correspondences(q.camera, q.frames, q.R, D; kw..., saturated = sat)
    pf = ring_correspondences(q.camera, q.frames, q.R, D; kw..., saturated = sat, symmetric = false)
    p0 = ring_correspondences(q.camera, q.frames, q.R, D; kw..., saturated = q.saturated)
    @test ps.n_dropped > 0 && pf.n_dropped == 0
    @test length(ps) < length(pf) < length(p0)
end

@testitem "ring_correspondences, ring_pair_vignetting: the device agrees with the host" setup = [VignettingSequence, BayerMaps] begin
    using Test, StaticArrays, CUDA
    using PanoramaStitching: ring_correspondences, ring_pair_vignetting, fibonacci_directions
    import VisualGeometryToolkitCore as VgtkCore
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
        xh = VgtkCore.estimate(ring_pair_vignetting(h)); xd = VgtkCore.estimate(ring_pair_vignetting(d))
        @test isequal(isfinite.(xh), isfinite.(xd)) && maximum(abs, filter(isfinite, xh .- xd)) < 1e-4
    end
end
