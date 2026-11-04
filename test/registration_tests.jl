# register_rotations, sample_frames and render_panorama on a synthetic pure-rotation sequence rendered through a known fisheye
# (a Kannala–Brandt lens as a generic grid model imaging past 90°): the rotations, the frames sampled at directions (every frame
# seeing a direction sees one radiance once its gain and vignetting are divided out) and the panorama (it is the scene).

@testsetup module SyntheticSequence
using LinearAlgebra: I
using StaticArrays: SVector, SMatrix, SA
using StableRNGs: StableRNG
using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, px, CentralGenericModel, backproject, bilinear_sample, Equirectangular,
    project, exp_so3, angle_between, Circle, Point2, Rect, estimate
import VisualGeometryToolkitEstimation as VgtkEstimation
using VisualGeometryToolkitCore: SplineProjection
export sequence

# the scene: random Gaussian blobs on an equirectangular texture; the camera: a KB fisheye (f = 300 px) imaging to 100° off axis;
# K frames rotating 35° about the vertical (`yaw_deg[k]` for k ≥ 2; frame 1 is the common frame, at the identity), 2×2 binned; a vignetting V(θ) = exp(−θ²/2·1.4²) and per-frame gains
function sequence(; K = 4, spline = false, yaw_deg = 35.0 .* (0:(K - 1)))
    rng = StableRNG(7)
    TW, TH = 2048, 1024
    tex = fill(0.15f0, TH, TW, 3)
    for _ in 1:20000
        λ = 2π * rand(rng) - π; φ = asin(2rand(rng) - 1)
        x = (λ + π) / 2π * TW; y = (π / 2 - φ) / π * TH
        s = 1.0 + 3rand(rng); a = 0.6 * rand(rng); c = 0.3 .+ 0.7 .* rand(rng, 3)
        sx = s / max(cos(φ), 0.05)
        for j in floor(Int, x - 3sx):ceil(Int, x + 3sx), i in max(1, floor(Int, y - 3s)):min(TH, ceil(Int, y + 3s))
            w = a * exp(-((j - x)^2 / sx^2 + (i - y)^2 / s^2) / 2)
            for ch in 1:3
                tex[i, mod1(j, TW), ch] += w * c[ch]
            end
        end
    end
    texture(d) = (p = project(Equirectangular(), d); bilinear_sample(tex, Val(3), SVector((p[1] + π) / 2π, (π / 2 - p[2]) / π)))
    V(θ) = exp(-0.5 * (θ / 1.4)^2)
    W = H = 1000; s = 2; Wf, Hf = W ÷ s, H ÷ s
    model = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))   # the parametric lens the grid resamples
    cam = CentralGenericModel(model, (W, H); grid_size = (40, 40))
    circle = Circle(Point2(500.0, 500.0), 480.0)
    axis = backproject(cam, SVector(500.0, 500.0))
    R = [k == 1 ? SMatrix{3, 3}(1.0I) : SMatrix{3, 3}(exp_so3(SVector(0.05randn(rng), deg2rad(yaw_deg[k]), 0.05randn(rng)))) for k in 1:K]
    gains = [1.0; exp.(0.1 .* randn(rng, K - 1))]
    frames = map(1:K) do k
        img = zeros(Float32, Hf, Wf, 3)
        for j in 1:Wf, i in 1:Hf
            u = SVector((j - 0.5) * s, (i - 0.5) * s)
            hypot(u[1] - 500.0, u[2] - 500.0) < 480.0 || continue
            f = backproject(cam, u)
            f === nothing && continue
            img[i, j, :] .= gains[k] * V(angle_between(f, axis)) .* texture(R[k] * f)
        end
        img
    end
    # the grid's forward map: a SplineProjection fitted to it over the image circle (a grid model has no closed-form projection)
    projection = spline ? estimate(VgtkEstimation.fit_camera(SplineProjection, cam, Rect(0, 0, W, H) ∩ circle; tolerance = 1.0e-2)) : nothing
    return (; frames, cam, model, projection, circle, raw_scale = s, R, gains, V, texture, axis)
end
end

@testitem "sample_frames: a parametric camera and a grid's SplineProjection agree" setup = [SyntheticSequence] begin
    # the sequence's Kannala–Brandt lens as the parametric CameraModel itself and the 40 × 40 grid resampled from it, through the
    # spline fitted to the grid: the frames sampled at the same directions agree
    using Test, StaticArrays
    using PanoramaStitching: sample_frames, fibonacci_directions
    q = sequence(; K = 2, spline = true)
    Hf, Wf = size(q.frames[1], 1), size(q.frames[1], 2)
    D = fibonacci_directions(q.R, 4000; fov_deg = 89.0)
    @test 0 < length(D) < 4000
    samples(cam) = sample_frames(cam, q.frames, q.R, D; saturated = [falses(Hf, Wf) for _ in 1:2], circle = q.circle,
                                 raw_scale = q.raw_scale, rim_margin_px = 12.0, fov_deg = 89.0, vignetting = (u, f) -> 1.0)
    p, g = samples(q.model), samples(q.projection)
    both = p.seen .& g.seen
    @test count(p.seen) > 0 && count(both) >= 0.99 * count(g.seen)
    @test maximum(i -> maximum(abs.(p.pixels[i] - g.pixels[i])), findall(both)) < 0.05                 # px: the spline's tolerance
    @test maximum(abs.(p.values[both, 2] .- view(g.values, :, :, 2)[both])) < 0.02
    @test all(isnan, view(p.values, :, :, 2)[.!p.seen]) && all(isnan, p.vignetting[.!p.seen])
    # a direction no frame sees: all of its row is unseen
    back = sample_frames(q.model, q.frames, q.R, [SVector(0.0, 0.0, -1.0)]; saturated = [falses(Hf, Wf) for _ in 1:2], circle = q.circle,
                         raw_scale = q.raw_scale, rim_margin_px = 12.0, fov_deg = 89.0, vignetting = (u, f) -> 1.0)
    @test !any(back.seen)
end

@testitem "sample_frames: a saturated pixel that any plane's sample blends masks the direction" begin
    # the planes are read at their own sample positions (a 2 × 2 Bayer binning's R and B at ±¼ px): a direction is a
    # measurement only when none of the pixels ANY plane's bilinear value blends is saturated, not only the centre's four
    using Test, LinearAlgebra, StaticArrays
    using PanoramaStitching: sample_frames
    using VisualGeometryToolkitCore: CameraModel, Lens, Perspective, px, WholePlane, backproject
    cam = CameraModel(100.0px, SA[50.0px, 50.0px]; lens = Lens(Perspective()))
    Hf = Wf = 50                                                                     # raw_scale 2: a 100 × 100 sensor
    frame = ones(Float32, Hf, Wf, 3); sat = falses(Hf, Wf)
    frame[25, 25, :] .= 1.0f6; sat[25, 25] = true                                    # clipped, its centre at (24.5, 24.5)
    offsets = (SVector(0.25, 0.25), SVector(0.0, 0.0), SVector(-0.25, -0.25))
    pts = vec([SVector(x, y) for x in 22.5:0.05:26.5, y in 22.5:0.05:26.5])          # frame COLMAP points around it
    D = [normalize(SVector{3, Float64}(backproject(cam, 2 .* p))) for p in pts]
    samples(o) = sample_frames(cam, [frame], [SMatrix{3, 3}(1.0I)], D; saturated = [sat], circle = WholePlane(), raw_scale = 2,
                               rim_margin_px = 0.0, fov_deg = 89.0, vignetting = (u, f) -> 1.0, plane_offsets = o)
    for o in (nothing, offsets)
        S = samples(o); seen = S.seen[:, 1]
        @test count(seen) > length(pts) ÷ 4
        @test all(c -> maximum(S.values[seen, 1, c]) < 1.001, 1:3)                  # no seen value blends the clipped pixel
    end
    # the centre's taps alone leave the points within 1 px of the clipped centre (per axis) unseen; the planes' ±¼ px reach more
    S0, S1 = samples(nothing), samples(offsets)
    @test all(S1.seen[:, 1] .<= S0.seen[:, 1]) && count(S1.seen) < count(S0.seen)
end

@testitem "register_rotations, sample_frames: a synthetic fisheye sweep" setup = [SyntheticSequence] begin
    using Test, LinearAlgebra, Statistics, StaticArrays
    using PanoramaStitching: register_rotations, sample_frames, fibonacci_directions
    import VisualGeometryToolkitCore as VgtkCore
    q = sequence(; spline = true)
    K = length(q.frames)
    reg = register_rotations(q.frames, q.cam; projection = q.projection, raw_scale = q.raw_scale, circle = q.circle, tone_percentile = 99.5,
                             detector_margin_px = 20.0, asift_tilts = [(1.0, 0.0)], max_detections = 1024, descriptor = :b, matcher = :b128,
                             min_score = 0.0, pairs = :all, max_lag = 1, overlap_deg = 120.0, min_inliers = 40, seed = 1, max_trials = 10000,
                             min_trials = 100, averaging_loss = VgtkCore.HuberShape(), bundle_loss = VgtkCore.CauchyShape(),
                             bundle_scale_multiple = 3.0, bundle_iterations = 3, bundle_max_iter = 100, bundle_ftol = 1.0e-10, bundle_xtol = 1.0e-10)
    errs = [rad2deg(norm(VgtkCore.log_so3(SMatrix{3, 3}(reg.rotations[k])' * q.R[k]))) for k in 1:K]
    @info "synthetic register: rotation errors (deg) $(round.(errs; digits = 4)), bundle rms $(round(reg.bundle.rms; digits = 3)) px"
    @test maximum(errs) < 0.02                                  # 1 px ≈ 0.19° at f = 300 px; thousands of tracks in the bundle
    @test length(reg.pairs) == K * (K - 1) ÷ 2 && all(p -> p.inliers >= 100, values(reg.pairs))
    @test reg.bundle.rms < 0.5 && reg.dropped == 0
    # a grid camera without its forward map is refused, before any work
    @test_throws ArgumentError register_rotations(q.frames, q.cam; raw_scale = q.raw_scale, circle = q.circle, tone_percentile = 99.5,
                             detector_margin_px = 20.0, asift_tilts = nothing, max_detections = 1, descriptor = :b, matcher = :b128,
                             min_score = 0.0, pairs = :all, max_lag = 1, overlap_deg = 120.0, min_inliers = 40, seed = 1, max_trials = 1,
                             min_trials = 1, averaging_loss = VgtkCore.HuberShape(), bundle_loss = VgtkCore.CauchyShape(),
                             bundle_scale_multiple = 3.0, bundle_iterations = 1, bundle_max_iter = 1, bundle_ftol = 1.0e-10, bundle_xtol = 1.0e-10)
    # the frames sampled at directions spread uniformly over the covered sphere, the vignetting read on the ray: every frame
    # seeing a direction sees one radiance once its gain and the TRUE V are divided out; a wrong V leaves a spread
    Hf, Wf = size(q.frames[1], 1), size(q.frames[1], 2)
    D = fibonacci_directions(reg.rotations, 20000; fov_deg = 89.0)
    spread(V) = begin
        S = sample_frames(q.projection, q.frames, reg.rotations, D; saturated = [falses(Hf, Wf) for _ in 1:K], circle = q.circle,
                          raw_scale = q.raw_scale, rim_margin_px = 12.0, fov_deg = 89.0,
                          vignetting = (u, f) -> V(VgtkCore.angle_between(f, q.axis)))
        out = Float64[]
        for i in axes(S.seen, 1)
            ks = findall(view(S.seen, i, :)); length(ks) >= 2 || continue
            L = [S.values[i, k, 2] / (q.gains[k] * S.vignetting[i, k]) for k in ks]
            push!(out, maximum(L) / minimum(L) - 1)
        end
        out
    end
    good = spread(q.V); bad = spread(θ -> exp(-0.5 * (θ / 1.0)^2))
    @info "sample_frames: $(length(good)) directions seen twice or more, median radiance spread $(round(median(good); digits = 4)) " *
          "(true V), $(round(median(bad); digits = 4)) (a wrong V)"
    @test length(good) > 1000
    @test median(good) < 0.03 && median(bad) > 3 * median(good)
end

@testitem "register_rotations: a dropped lag pair, the overlap pairs connect the frames" setup = [SyntheticSequence] begin
    # frames at yaw 0°, 180° and 90° in capture order, keypoints inside the circle less 60 px (r ≤ 420 px: ~84° off axis): the
    # lag pair 0-1 shares no keypoint's scene and is dropped, which leaves frame 0 a part of its own; the pair 0-2, between the
    # two parts, is verified and joins them (with the 20 px margin of the sweep above, the ~13° rim of 0-1 is verified)
    using Test, LinearAlgebra, StaticArrays
    using PanoramaStitching: register_rotations
    import VisualGeometryToolkitCore as VgtkCore
    q = sequence(; K = 3, spline = true, yaw_deg = [0.0, 180.0, 90.0])
    reg = register_rotations(q.frames, q.cam; projection = q.projection, raw_scale = q.raw_scale, circle = q.circle, tone_percentile = 99.5,
                             detector_margin_px = 60.0, asift_tilts = [(1.0, 0.0)], max_detections = 1024, descriptor = :b, matcher = :b128,
                             min_score = 0.0, pairs = :lag_overlap, max_lag = 1, overlap_deg = 60.0, min_inliers = 40, seed = 1,
                             max_trials = 10000, min_trials = 100, averaging_loss = VgtkCore.HuberShape(), bundle_loss = VgtkCore.CauchyShape(),
                             bundle_scale_multiple = 3.0, bundle_iterations = 3, bundle_max_iter = 100, bundle_ftol = 1.0e-10, bundle_xtol = 1.0e-10)
    @test sort(collect(keys(reg.pairs))) == [(1, 3), (2, 3)]
    errs = [rad2deg(norm(VgtkCore.log_so3(SMatrix{3, 3}(reg.rotations[k])' * q.R[k]))) for k in 1:3]
    @info "a dropped lag pair: rotation errors (deg) $(round.(errs; digits = 4))"
    @test maximum(errs) < 0.05
end

@testitem "register_rotations: the overlap pairs of :lag_overlap, and the pairs' samplers" begin
    # frames 30° apart in yaw (overlap_deg 70°: the pairs two apart overlap), a pair verified with its true rotation (f_j = R_ij·f_i,
    # R_ij = R_jᵀR_i) unless it is in `dropped`; the order in which the pairs past the lag are tried, after the lag pairs
    using Test, StaticArrays
    using PanoramaStitching: PanoramaStitching, ViewGraph
    import VisualGeometryToolkitCore as VgtkCore
    function tried_pairs(K, dropped)
        R = [SMatrix{3, 3, Float64, 9}(VgtkCore.exp_so3(SVector(0.0, deg2rad(30.0 * (k - 1)), 0.0))) for k in 1:K]
        verified = Dict{Tuple{Int, Int}, NamedTuple{(:rotation, :inliers), Tuple{SMatrix{3, 3, Float64, 9}, Int}}}()
        tried = Tuple{Int, Int}[]
        verify(i, j) = ((i, j) in dropped || (verified[(i, j)] = (rotation = R[j]' * R[i], inliers = 100)); nothing)
        foreach(k -> verify(k, k + 1), 1:(K - 1))
        PanoramaStitching._verify_overlap_pairs!((i, j) -> (push!(tried, (i, j)); verify(i, j)), verified, K, 1, 70.0, VgtkCore.HuberShape())
        return tried, verified
    end
    @test first(tried_pairs(6, [])) == [(1, 3), (2, 4), (3, 5), (4, 6)]                # connected: the aimed pairs only
    # the lag pair 3-4 dropped: the nearest pair between {1, 2, 3} and {4, 5, 6} joins them, then the aimed pairs
    tried, verified = tried_pairs(6, [(3, 4)])
    @test tried == [(2, 4), (1, 3), (3, 5), (4, 6)]
    @test length(ViewGraph(collect(1:6), verified).frames) == 6
    @test first(tried_pairs(6, [(3, 4), (2, 4)])) == [(2, 4), (3, 5), (1, 3), (4, 6)]  # the nearest dropped too: the next
    @test first(tried_pairs(6, [(1, 2)])) == [(1, 3), (2, 4), (3, 5), (4, 6)]          # frame 1 a part of its own
    # nothing joins {1, 2} and {3, 4}: every pair between them tried, the graph refused
    tried, verified = tried_pairs(4, [(2, 3), (1, 3), (2, 4), (1, 4)])
    @test tried == [(1, 3), (2, 4), (1, 4)]
    @test_throws ArgumentError ViewGraph(collect(1:4), verified)
    # the samplers: one stream per (seed, i, j), reproducible
    draw(s, i, j) = rand(PanoramaStitching._pair_rng(s, i, j), UInt64, 4)
    @test draw(1, 1, 2) == draw(1, 1, 2) && draw(1, 1, 2) != draw(1, 1, 3)
end

@testitem "render_panorama: the warp through any camera reproduces the scene, in any view" setup = [SyntheticSequence] begin
    # every panorama pixel's ray, rotated into each frame and projected through the camera, sampled and divided by the vignetting
    # looked up at the sensor point or the ray: with the true rotations, constant exposure (the sequence's gains divided out of the
    # frames up front) and the true V, the panorama IS the scene's texture, equirectangular and cylindrical; the parametric
    # Kannala–Brandt model and the grid's spline give the same panorama, and a V read on the sensor the same as one read on the ray
    using Test, Statistics, LinearAlgebra, StaticArrays
    using PanoramaStitching: render_panorama, render_cube_map, panorama_view, world_direction, CUBE_FACES, sample_frames
    import VisualGeometryToolkitCore as VgtkCore
    q = sequence(; K = 3, spline = true)
    frames = [f ./ Float32(g) for (f, g) in zip(q.frames, q.gains)]                 # constant exposure
    Hf, Wf = size(frames[1], 1), size(frames[1], 2)
    on_ray(u, f) = q.V(VgtkCore.angle_between(f, q.axis))                           # V defined on the ray's direction
    on_sensor(u, f) = q.V(VgtkCore.angle_between(VgtkCore.backproject(q.cam, u), q.axis))   # V defined on the sensor
    kw = (; circle = q.circle, raw_scale = q.raw_scale, rim_margin_px = 12.0, fov_deg = 89.0)
    saturated = [falses(Hf, Wf) for _ in 1:3]                                       # the sampling's measurement masks
    eq = panorama_view(VgtkCore.Equirectangular(), (-80.0, 150.0, -60.0, 60.0), 0.5)
    pano(cam, view = eq; vignetting = on_ray) = render_panorama(cam, frames, q.R, view; vignetting, kw...)
    g = pano(q.projection); p = pano(q.model)
    @test size(g.image) == (240, 460, 3) && all(g.extent .≈ (-80.0, 150.0, -60.0, 60.0))
    seen = findall(>(0), g.owner)
    @test length(seen) > 0.4 * length(g.owner)
    @test all(i -> 1 <= g.owner[i] <= 3 && g.coverage[i] >= 1, seen) && all(i -> isnan(g.image[i, 2]), findall(==(0), g.owner))
    # the scene at each pixel's ray (row i: latitude 60 − 0.5(i − ½); column j: longitude −80 + 0.5(j − ½))
    truth(i, j) = q.texture(world_direction(60 - 0.5 * (i - 0.5), -80 + 0.5 * (j - 0.5)))
    rel = [abs(g.image[I, 2] - truth(I[1], I[2])[2]) / truth(I[1], I[2])[2] for I in seen]
    @info "render_panorama: median relative error to the scene $(round(median(rel); digits = 4)), 90 % $(round(quantile(rel, 0.9); digits = 4))"
    @test median(rel) < 0.02 && quantile(rel, 0.9) < 0.08                          # bilinear sampling of 2 × 2-binned frames
    both = findall((g.owner .> 0) .& (p.owner .> 0))
    @test length(both) >= 0.99 * length(seen) && maximum(abs.(g.image[both, 2] .- p.image[both, 2])) < 0.01
    s = pano(q.projection; vignetting = on_sensor)                                  # the same V, read on the sensor
    @test s.owner == g.owner && maximum(abs.(s.image[seen, 2] .- g.image[seen, 2])) < 1.0e-3
    # the render and sample_frames are one pipeline: the frames sampled at the rays of the rendered pixels give, for the pixel's
    # owner, its value — with the planes' sample offsets too (R read ¼ px off, B the other way, G at the centre)
    offsets = (SVector(0.25, 0.25), SVector(0.0, 0.0), SVector(-0.25, -0.25))
    o = render_panorama(q.model, frames, q.R, eq; vignetting = on_ray, plane_offsets = offsets, kw...)
    @test o.owner == p.owner && view(o.image, :, :, 2) == view(p.image, :, :, 2) && view(o.image, :, :, 1) != view(p.image, :, :, 1)
    pick = seen[1:97:end]
    D = [eq.rays(SVector(I[2] - 0.5, I[1] - 0.5)) for I in pick]
    S = sample_frames(q.model, frames, q.R, D; vignetting = on_ray, plane_offsets = offsets, saturated, kw...)
    @test all(i -> S.seen[i, o.owner[pick[i]]], eachindex(pick))
    @test maximum(i -> maximum(abs(S.values[i, o.owner[pick[i]], c] / S.vignetting[i, o.owner[pick[i]]] - o.image[pick[i], c]) for c in 1:3),
                  eachindex(pick)) < 1.0e-5
    # cylindrical: rows uniform in h = tan(latitude), from tan 50° down; the scene at each pixel's ray
    cyl = panorama_view(VgtkCore.Cylindrical(), (-80.0, 150.0, -50.0, 50.0), 0.5)
    c = pano(q.model, cyl)
    δ = deg2rad(0.5); h1 = tand(50.0)
    @test size(c.image) == (round(Int, 2h1 / δ), 460, 3) && c.extent[4] == 50.0
    @test c.extent[3] ≈ -50.0 atol = 0.5
    ctruth(i, j) = q.texture(world_direction(atand(h1 - δ * (i - 0.5)), -80 + 0.5 * (j - 0.5)))
    cseen = findall(>(0), c.owner)
    crel = [abs(c.image[I, 2] - ctruth(I[1], I[2])[2]) / ctruth(I[1], I[2])[2] for I in cseen]
    @info "render_panorama (cylindrical): median relative error to the scene $(round(median(crel); digits = 4))"
    @test length(cseen) > 0.4 * length(c.owner) && median(crel) < 0.02 && quantile(crel, 0.9) < 0.08
    # a cube map: the front face is the rectilinear view along +z, the faces tile the cross
    cube = render_cube_map(q.model, frames, q.R, 64; vignetting = on_ray, kw...)
    @test keys(cube.faces) == keys(CUBE_FACES) && size(cube.cross) == (192, 256, 3)
    @test isequal(cube.cross[65:128, 65:128, :], cube.faces.front.image)
    f = cube.faces.front
    i, j = 20, 40                                                                     # the pixel's ray through the face camera
    d = normalize(SVector(j - 0.5 - 32, i - 0.5 - 32, 32.0))
    @test f.owner[i, j] > 0 && abs(f.image[i, j, 2] - q.texture(d)[2]) / q.texture(d)[2] < 0.05
end

@testitem "render_panorama: a grid lens's SplineProjection renders on the device exactly as on the host" setup = [SyntheticSequence] begin
    # the warp's map wraps the projection in a BoundedMap and a ComposedTransformation chain, which reach image_warp!'s kernel by
    # Core's Adapt rules (the spline's control grid moved to the device); the planes' offsets give each plane its own warp
    using Test, StaticArrays, CUDA
    using PanoramaStitching: render_panorama, panorama_view
    import VisualGeometryToolkitCore as VgtkCore
    if !CUDA.functional()
        @test_skip "no CUDA device"
    else
        q = sequence(; K = 3, spline = true)
        @test q.projection isa VgtkCore.SplineProjection
        eq = panorama_view(VgtkCore.Equirectangular(), (-80.0, 150.0, -60.0, 60.0), 0.5)
        kw = (; vignetting = (u, f) -> 1.0, circle = q.circle, raw_scale = q.raw_scale, rim_margin_px = 12.0, fov_deg = 89.0,
              plane_offsets = (SVector(0.25, 0.25), SVector(0.0, 0.0), SVector(-0.25, -0.25)))
        h = render_panorama(q.projection, q.frames, q.R, eq; kw...)
        d = render_panorama(q.projection, CuArray.(q.frames), q.R, eq; kw...)
        @test count(>(0), h.owner) > 0.4 * length(h.owner)
        @test Array(d.owner) == h.owner && Array(d.coverage) == h.coverage
        di = Array(d.image); seen = .!isnan.(h.image)
        @test isnan.(di) == .!seen && maximum(abs.(di[seen] .- h.image[seen])) == 0
    end
end
