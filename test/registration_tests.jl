# register_rotations, sample_frames and render_panorama on a synthetic pure-rotation sequence rendered through a known fisheye
# (a Kannala–Brandt lens as a SplineLens grid camera imaging past 90°, its lens coverage an image circle): the rotations, the
# frames sampled at directions (every frame seeing a direction sees one radiance once its gain and vignetting are divided out),
# the panorama (it is the scene), and the camera's `imaged_region` bounding the features and the render.

@testsetup module SyntheticSequence
using LinearAlgebra: I
using StaticArrays: SVector, SMatrix, SA
using StableRNGs: StableRNG
using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, px, SplineLens, BoundedCamera, Size2, backproject, bilinear_sample,
    Equirectangular, project, exp_so3, angle_between, Circle, Point2, imaged_region, with_model, estimate
import VisualGeometryToolkitEstimation as VgtkEstimation
export sequence

# the scene: random Gaussian blobs on an equirectangular texture; the camera: a KB fisheye (f = 300 px) imaging to 100° off axis;
# K frames rotating 35° about the vertical (`yaw_deg[k]` for k ≥ 2; frame 1 is the common frame, at the identity), 2×2 binned; a vignetting V(θ) = exp(−θ²/2·1.4²) and per-frame gains.
# `cam` the grid camera, `model` the parametric one: both bounded by the sensor, their lens coverage the image circle.
# `lit = :sensor` renders the whole sensor through the parametric lens instead (frames lit to the sensor's edge).
function sequence(; K = 4, yaw_deg = 35.0 .* (0:(K - 1)), lit = :circle)
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
    kb = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))   # the parametric lens the grid resamples
    circle = Circle(Point2(500.0, 500.0), 480.0)
    model = BoundedCamera(kb, Size2(W, H); lens_coverage = circle)
    # the grid: a 40 × 40 SplineLens fitted to the lens over its imaged region, with its forward spline (a grid's projection)
    grid = estimate(VgtkEstimation.fit_camera(SplineLens, model, imaged_region(model); tolerance = 1.0e-2, grid_size = (40, 40)))
    cam = with_model(model, grid)                     # the same sensor and lens coverage, the grid's model
    axis = backproject(cam, SVector(500.0, 500.0))
    R = [k == 1 ? SMatrix{3, 3}(1.0I) : SMatrix{3, 3}(exp_so3(SVector(0.05randn(rng), deg2rad(yaw_deg[k]), 0.05randn(rng)))) for k in 1:K]
    gains = [1.0; exp.(0.1 .* randn(rng, K - 1))]
    frames = map(1:K) do k
        img = zeros(Float32, Hf, Wf, 3)
        for j in 1:Wf, i in 1:Hf
            u = SVector((j - 0.5) * s, (i - 0.5) * s)
            lit === :sensor || hypot(u[1] - 500.0, u[2] - 500.0) < 480.0 || continue
            f = backproject(lit === :sensor ? model : cam, u)
            all(isfinite, f) || continue                    # a grid model's backproject is NaN where it images nothing
            img[i, j, :] .= gains[k] * V(angle_between(f, axis)) .* texture(R[k] * f)
        end
        img
    end
    return (; frames, cam, model, circle, R, gains, V, texture, axis)
end
end

@testitem "sample_frames: a parametric camera and a grid camera agree" setup = [SyntheticSequence, BayerMaps] begin
    # the sequence's Kannala–Brandt lens as the parametric CameraModel itself and the 40 × 40 SplineLens grid fitted to it,
    # through the grid's forward spline: the frames sampled at the same directions agree
    using Test, StaticArrays
    using PanoramaStitching: sample_frames, fibonacci_directions
    q = sequence(; K = 2)
    Hf, Wf = size(q.frames[1], 1), size(q.frames[1], 2)
    D = fibonacci_directions(q.R, 4000; fov_deg = 89.0)
    @test 0 < length(D) < 4000
    samples(cam) = sample_frames(cam, q.frames, q.R, D; saturated = [falses(Hf, Wf) for _ in 1:2],
                                 frame_map = FRAME_MAP, rim_margin_px = 12.0, fov_deg = 89.0, vignetting = (u, f) -> 1.0)
    p, g = samples(q.model), samples(q.cam)
    both = p.seen .& g.seen
    @test count(p.seen) > 0 && count(both) >= 0.99 * count(g.seen)
    @test maximum(i -> maximum(abs.(p.pixels[i] - g.pixels[i])), findall(both)) < 0.05                 # px: the spline's tolerance
    @test maximum(abs.(p.values[both, 2] .- view(g.values, :, :, 2)[both])) < 0.02
    @test all(isnan, view(p.values, :, :, 2)[.!p.seen]) && all(isnan, p.vignetting[.!p.seen])
    # a direction no frame sees: all of its row is unseen
    back = sample_frames(q.model, q.frames, q.R, [SVector(0.0, 0.0, -1.0)]; saturated = [falses(Hf, Wf) for _ in 1:2],
                         frame_map = FRAME_MAP, rim_margin_px = 12.0, fov_deg = 89.0, vignetting = (u, f) -> 1.0)
    @test !any(back.seen)
end

@testitem "sample_frames: a saturated pixel that any plane's sample blends masks the direction" setup = [BayerMaps] begin
    # the planes are read at their own sample positions (a 2 × 2 Bayer binning's R and B at ±¼ px): a direction is a
    # measurement only when none of the pixels ANY plane's bilinear value blends is saturated, not only the centre's four
    using Test, LinearAlgebra, StaticArrays
    using PanoramaStitching: sample_frames
    using VisualGeometryToolkitCore: CameraModel, BoundedCamera, Size2, Lens, Perspective, px, backproject
    cam = BoundedCamera(CameraModel(100.0px, SA[50.0px, 50.0px]; lens = Lens(Perspective())), Size2(100, 100))   # the whole sensor lit
    Hf = Wf = 50                                                                     # 2 × 2 binned: a 100 × 100 sensor
    frame = ones(Float32, Hf, Wf, 3); sat = falses(Hf, Wf)
    frame[25, 25, :] .= 1.0f6; sat[25, 25] = true                                    # clipped, its centre at (24.5, 24.5)
    pts = vec([SVector(x, y) for x in 22.5:0.05:26.5, y in 22.5:0.05:26.5])          # frame COLMAP points around it
    D = [normalize(SVector{3, Float64}(backproject(cam, 2 .* p))) for p in pts]
    centred = ntuple(_ -> FRAME_MAP, 3)                                            # every plane at the cell centre
    samples(P) = sample_frames(cam, [frame], [SMatrix{3, 3}(1.0I)], D; saturated = [sat], frame_map = FRAME_MAP,
                               rim_margin_px = 0.0, fov_deg = 89.0, vignetting = (u, f) -> 1.0, plane_maps = P)
    for P in (centred, PLANE_MAPS)
        S = samples(P); seen = S.seen[:, 1]
        @test count(seen) > length(pts) ÷ 4
        @test all(c -> maximum(S.values[seen, 1, c]) < 1.001, 1:3)                  # no seen value blends the clipped pixel
    end
    # the centre's taps alone leave the points within 1 px of the clipped centre (per axis) unseen; the planes' ±¼ px reach more
    S0, S1 = samples(centred), samples(PLANE_MAPS)
    @test samples(centred).seen == sample_frames(cam, [frame], [SMatrix{3, 3}(1.0I)], D; saturated = [sat], frame_map = FRAME_MAP,
                                                 rim_margin_px = 0.0, fov_deg = 89.0, vignetting = (u, f) -> 1.0).seen   # the default
    @test all(S1.seen[:, 1] .<= S0.seen[:, 1]) && count(S1.seen) < count(S0.seen)
end

@testitem "register_rotations, sample_frames: a synthetic fisheye sweep" setup = [SyntheticSequence, BayerMaps] begin
    using Test, LinearAlgebra, Statistics, StaticArrays
    using PanoramaStitching: PanoramaStitching, register_rotations, sample_frames, fibonacci_directions
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    q = sequence()
    K = length(q.frames)
    reg = register_rotations(q.frames, q.cam; frame_map = FRAME_MAP, tone_percentile = 99.5,
                             detector_margin_px = 20.0, asift_tilts = [(1.0, 0.0)], max_detections = 1024, descriptor = :b, matcher = :b128,
                             min_score = 0.0, pairs = :all, max_lag = 1, overlap_deg = 120.0, min_inliers = 40, seed = 1, max_trials = 10000,
                             min_trials = 100, averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100), bundle_optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(efficiency_constant = 3.0),
                              bundle_max_iter = 100, bundle_ftol = 1.0e-10, bundle_xtol = 1.0e-10)
    errs = [rad2deg(norm(VgtkCore.log_so3(SMatrix{3, 3}(reg.rotations[k])' * q.R[k]))) for k in 1:K]
    @info "synthetic register: rotation errors (deg) $(round.(errs; digits = 4)), bundle rms $(round(reg.bundle.rms; digits = 3)) px"
    @test maximum(errs) < 0.02                                  # 1 px ≈ 0.19° at f = 300 px; thousands of tracks in the bundle
    @test length(reg.pairs) == K * (K - 1) ÷ 2 && all(p -> p.inliers >= 100, values(reg.pairs))
    # The bundle's residuals in the pairs' units: every observation is its keypoint with the detector's covariance Σ (camera
    # pixels² for unit noise of the detection image), so a pixel residual over √(tr Σ / 2) is in sigmas of that unit noise, the
    # unit of the pairs' pooled scale σ̂ (σ₀: the noise σ₀²Σ)
    whitened = filter(isfinite, [reg.bundle.metadata.residuals[j][i] /
                                 sqrt(tr(Statistics.cov(PanoramaStitching._camera_keypoint(FRAME_MAP, reg.features[k][kp].detection))) / 2)
                                 for (j, t) in enumerate(reg.tracks) for (i, (k, kp)) in enumerate(t)])
    @test sqrt(mean(abs2, whitened)) < 3 * reg.bundle_sigma && reg.dropped == 0
    @test median(filter(isfinite, reduce(vcat, reg.bundle.metadata.residuals))) < 0.15   # px
    # the view graph is VisualGeometryToolkitEstimation's, each edge the pair's rotation weighted by its inlier count
    @test reg.graph isa VgtkEstimation.ViewGraph && VgtkEstimation.nedges(reg.graph) == length(reg.pairs)
    @test all(p -> VgtkCore.weight_of(reg.graph[p...]) == reg.pairs[p].inliers, keys(reg.pairs))
    # the averaging's gauge is the first frame: its rotation is the identity
    @test VgtkCore.estimate(reg.averaging)[1] ≈ one(SMatrix{3, 3, Float64, 9}) atol = 1e-12
    # every kept feature lies in the camera's imaged region eroded by the detector margin (the image circle less 20 px)
    px(f) = [FRAME_MAP(SVector{2, Float64}(d.center...)) for d in f]
    @test all(f -> all(p -> hypot(p[1] - 500.0, p[2] - 500.0) <= 480.0 - 20.0, px(f)), reg.features)
    @test all(f -> !isempty(f), reg.features)
    # the frames sampled at directions spread uniformly over the covered sphere, the vignetting read on the ray: every frame
    # seeing a direction sees one radiance once its gain and the TRUE V are divided out; a wrong V leaves a spread
    Hf, Wf = size(q.frames[1], 1), size(q.frames[1], 2)
    D = fibonacci_directions(reg.rotations, 20000; fov_deg = 89.0)
    spread(V) = begin
        S = sample_frames(q.cam, q.frames, reg.rotations, D; saturated = [falses(Hf, Wf) for _ in 1:K],
                          frame_map = FRAME_MAP, rim_margin_px = 12.0, fov_deg = 89.0,
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

@testitem "register_rotations: a dropped lag pair, the overlap pairs connect the frames" setup = [SyntheticSequence, BayerMaps] begin
    # frames at yaw 0°, 180° and 90° in capture order, keypoints inside the circle less 60 px (r ≤ 420 px: ~84° off axis): the
    # lag pair 0-1 shares no keypoint's scene and is dropped, which leaves frame 0 a part of its own; the pair 0-2, between the
    # two parts, is verified and joins them (with the 20 px margin of the sweep above, the ~13° rim of 0-1 is verified)
    using Test, LinearAlgebra, StaticArrays
    using PanoramaStitching: register_rotations, verify_pairs, VerifiedPairs
    import VisualGeometryToolkitCore as VgtkCore
    q = sequence(; K = 3, yaw_deg = [0.0, 180.0, 90.0])
    reg = register_rotations(q.frames, q.cam; frame_map = FRAME_MAP, tone_percentile = 99.5,
                             detector_margin_px = 60.0, asift_tilts = [(1.0, 0.0)], max_detections = 1024, descriptor = :b, matcher = :b128,
                             min_score = 0.0, pairs = :lag_overlap, max_lag = 1, overlap_deg = 60.0, min_inliers = 40, seed = 1,
                             max_trials = 10000, min_trials = 100, averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100), bundle_optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(efficiency_constant = 3.0),
                              bundle_max_iter = 100, bundle_ftol = 1.0e-10, bundle_xtol = 1.0e-10)
    @test sort(collect(keys(reg.pairs))) == [(1, 3), (2, 3)]
    errs = [rad2deg(norm(VgtkCore.log_so3(SMatrix{3, 3}(reg.rotations[k])' * q.R[k]))) for k in 1:3]
    @info "a dropped lag pair: rotation errors (deg) $(round.(errs; digits = 4))"
    @test maximum(errs) < 0.05
    # the verified pairs kept apart: verify_pairs, then the rest from them, is the registration exactly
    vp = verify_pairs(q.frames, q.cam; frame_map = FRAME_MAP, tone_percentile = 99.5, detector_margin_px = 60.0, asift_tilts = [(1.0, 0.0)],
                      max_detections = 1024, descriptor = :b, matcher = :b128, min_score = 0.0, pairs = :lag_overlap, max_lag = 1,
                      overlap_deg = 60.0, min_inliers = 40, seed = 1, max_trials = 10000, min_trials = 100, averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100))
    @test vp isa VerifiedPairs && vp.stats == reg.pairs && length(vp.features) == 3
    @test all(p -> vp.inlier_matches[p] isa Vector{Tuple{Int, Int}} && length(vp.inlier_matches[p]) == vp.verified[p].inliers, keys(vp.verified))
    bkw = (; frame_map = FRAME_MAP, bundle_optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(efficiency_constant = 3.0), bundle_max_iter = 100,
           bundle_ftol = 1.0e-10, bundle_xtol = 1.0e-10)
    reg2 = register_rotations(vp, q.cam; averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100), bkw...)
    @test reg2.rotations == reg.rotations && reg2.tracks == reg.tracks && reg2.pairs == reg.pairs
    # another averaging loss from the same pairs: no feature detected again, the rotations as good
    reg3 = register_rotations(vp, q.cam; averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.CauchyShape}(max_iter = 100), bkw...)
    @test maximum(k -> rad2deg(norm(VgtkCore.log_so3(SMatrix{3, 3}(reg3.rotations[k])' * q.R[k]))), 1:3) < 0.05
    # the bundle always whitens by the keypoints' covariances: the retired whitening switch is refused by name
    @test_throws Regex("bundle_" * "whitening. is retired") register_rotations(vp, q.cam; averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100), bkw..., Symbol("bundle_" * "whitening") => :none)
end

@testitem "register_rotations: the overlap pairs of :lag_overlap, and the pairs' samplers" begin
    # frames 30° apart in yaw (overlap_deg 70°: the pairs two apart overlap), a pair verified with its true rotation (f_j = R_ij·f_i,
    # R_ij = R_jᵀR_i) unless it is in `dropped`; the order in which the pairs past the lag are tried, after the lag pairs
    using Test, StaticArrays
    using PanoramaStitching: PanoramaStitching
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    function tried_pairs(K, dropped)
        R = [SMatrix{3, 3, Float64, 9}(VgtkCore.exp_so3(SVector(0.0, deg2rad(30.0 * (k - 1)), 0.0))) for k in 1:K]
        verified = Dict{Tuple{Int, Int}, NamedTuple{(:rotation, :inliers, :covariance), Tuple{SMatrix{3, 3, Float64, 9}, Int, Union{Nothing, SMatrix{3, 3, Float64, 9}}}}}()
        tried = Tuple{Int, Int}[]
        verify(i, j) = ((i, j) in dropped || (verified[(i, j)] = (rotation = R[j]' * R[i], inliers = 100,
                                                                         covariance = SMatrix{3, 3, Float64, 9}(1.0e-6, 0, 0, 0, 1.0e-6, 0, 0, 0, 1.0e-6))); nothing)
        foreach(k -> verify(k, k + 1), 1:(K - 1))
        PanoramaStitching._verify_overlap_pairs!((i, j) -> (push!(tried, (i, j)); verify(i, j)), verified, K, 1, 70.0,
                                                 VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100))
        return tried, verified
    end
    @test first(tried_pairs(6, [])) == [(1, 3), (2, 4), (3, 5), (4, 6)]                # connected: the aimed pairs only
    # the lag pair 3-4 dropped: the nearest pair between {1, 2, 3} and {4, 5, 6} joins them, then the aimed pairs
    tried, verified = tried_pairs(6, [(3, 4)])
    @test tried == [(2, 4), (1, 3), (3, 5), (4, 6)]
    @test VgtkEstimation.components(PanoramaStitching._view_graph(6, verified)) == [collect(1:6)]
    @test first(tried_pairs(6, [(3, 4), (2, 4)])) == [(2, 4), (3, 5), (1, 3), (4, 6)]  # the nearest dropped too: the next
    @test first(tried_pairs(6, [(1, 2)])) == [(1, 3), (2, 4), (3, 5), (4, 6)]          # frame 1 a part of its own
    # nothing joins {1, 2} and {3, 4}: every pair between them tried, the graph refused
    tried, verified = tried_pairs(4, [(2, 3), (1, 3), (2, 4), (1, 4)])
    @test tried == [(1, 3), (2, 4), (1, 4)]
    @test VgtkEstimation.components(PanoramaStitching._view_graph(4, verified)) == [[1, 2], [3, 4]]
    @test_throws ArgumentError VgtkEstimation.fit_rotation_averaging(PanoramaStitching._view_graph(4, verified);
                                                                 optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}())
    # the samplers: one stream per (seed, i, j), reproducible
    draw(s, i, j) = rand(PanoramaStitching._pair_rng(s, i, j), UInt64, 4)
    @test draw(1, 1, 2) == draw(1, 1, 2) && draw(1, 1, 2) != draw(1, 1, 3)
end

@testitem "render_panorama: the warp through any camera reproduces the scene, in any view" setup = [SyntheticSequence, BayerMaps] begin
    # every panorama pixel's ray, rotated into each frame and projected through the camera, sampled and divided by the vignetting
    # looked up at the sensor point or the ray: with the true rotations, constant exposure (the sequence's gains divided out of the
    # frames up front) and the true V, the panorama IS the scene's texture, equirectangular and cylindrical; the parametric
    # Kannala–Brandt model and the grid camera give the same panorama, and a V read on the sensor the same as one read on the ray
    using Test, Statistics, LinearAlgebra, StaticArrays
    using PanoramaStitching: render_panorama, render_cube_map, panorama_view, world_direction, CUBE_FACES, sample_frames
    import VisualGeometryToolkitCore as VgtkCore
    q = sequence(; K = 3)
    frames = [f ./ Float32(g) for (f, g) in zip(q.frames, q.gains)]                 # constant exposure
    Hf, Wf = size(frames[1], 1), size(frames[1], 2)
    on_ray(u, f) = q.V(VgtkCore.angle_between(f, q.axis))                           # V defined on the ray's direction
    on_sensor(u, f) = q.V(VgtkCore.angle_between(VgtkCore.backproject(q.cam, u), q.axis))   # V defined on the sensor
    kw = (; frame_map = FRAME_MAP, rim_margin_px = 12.0, fov_deg = 89.0)
    saturated = [falses(Hf, Wf) for _ in 1:3]                                       # the sampling's measurement masks
    eq = panorama_view(VgtkCore.Equirectangular(), (-80.0, 150.0, -60.0, 60.0), 0.5)
    pano(cam, view = eq; vignetting = on_ray) = render_panorama(cam, frames, q.R, view; vignetting, kw...)
    g = pano(q.cam); p = pano(q.model)
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
    s = pano(q.cam; vignetting = on_sensor)                                  # the same V, read on the sensor
    @test s.owner == g.owner && maximum(abs.(s.image[seen, 2] .- g.image[seen, 2])) < 1.0e-3
    # the render and sample_frames are one pipeline: the frames sampled at the rays of the rendered pixels give, for the pixel's
    # owner, its value — through the planes' own maps too (R read ¼ px off, B the other way, G at the centre: a BGGR binning)
    o = render_panorama(q.model, frames, q.R, eq; vignetting = on_ray, plane_maps = PLANE_MAPS, kw...)
    @test o.owner == p.owner && view(o.image, :, :, 2) == view(p.image, :, :, 2) && view(o.image, :, :, 1) != view(p.image, :, :, 1)
    pick = seen[1:97:end]
    D = [eq.rays(SVector(I[2] - 0.5, I[1] - 0.5)) for I in pick]
    S = sample_frames(q.model, frames, q.R, D; vignetting = on_ray, plane_maps = PLANE_MAPS, saturated, kw...)
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

@testitem "a camera with no stored lens coverage registers as in 0.5.1, its frames lit to the sensor's edge" setup = [SyntheticSequence, BayerMaps] begin
    # Review focus 4. A parametric camera bounded by its sensor alone (coverage the whole sensor, 0.5.1's `circle =
    # WholePlane()`), its frames lit to the sensor's edge: the margins erode only a lens coverage's edge, so the features reach
    # the sensor's edge whatever `detector_margin_px` (the same keypoints, in the same order: detect_features' order is
    # deterministic); the bundle converges, and its rotations are those of the bundle through the bare CameraModel — 0.5.1's
    # bundle, whose residual nothing masks — to 1e-9 (the lens coverage does not enter the bundle).
    using Test, LinearAlgebra, StaticArrays
    using PanoramaStitching: PanoramaStitching, register_rotations, refine_rotations
    import VisualGeometryToolkitCore as VgtkCore
    q = sequence(; K = 3, lit = :sensor)
    bare = VgtkCore.BoundedCamera(VgtkCore.model(q.model), VgtkCore.Size2(1000, 1000))
    @test PanoramaStitching._imaged_less_rim(bare, 30.0) == VgtkCore.imaged_region(bare)
    region = PanoramaStitching.frame_region(bare, 1000, 1000, 12.0)
    @test SVector(2.5, 2.5) ∈ region && SVector(997.5, 500.0) ∈ region && SVector(1.5, 500.0) ∉ region
    bkw = (; optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(efficiency_constant = 3.0), max_iter = 1000, ftol = 1.0e-10,
           xtol = 1.0e-10)
    reg(margin) = register_rotations(q.frames, bare; frame_map = FRAME_MAP, tone_percentile = 99.5, detector_margin_px = margin,
                                     asift_tilts = [(1.0, 0.0)], max_detections = 1024, descriptor = :b, matcher = :b128, min_score = 0.0,
                                     pairs = :all, max_lag = 1, overlap_deg = 120.0, min_inliers = 40, seed = 1, max_trials = 10000,
                                     min_trials = 100, averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100),
                                     bundle_optimization_strategy = bkw.optimization_strategy,
                                     bundle_max_iter = bkw.max_iter, bundle_ftol = bkw.ftol, bundle_xtol = bkw.xtol)
    a, b = reg(0.0), reg(30.0)
    px(f) = [FRAME_MAP(SVector{2, Float64}(d.center...)) for d in f]
    @test [px(f) for f in a.features] == [px(f) for f in b.features]
    edge(p) = min(p[1], p[2], 1000 - p[1], 1000 - p[2])
    @test any(f -> any(p -> edge(p) < 30.0, px(f)), b.features)                 # kept within the margin of the sensor's edge
    @test VgtkCore.converged(b.bundle) && b.bundle.metadata.n_dropped_start == 0
    @test maximum(k -> rad2deg(norm(VgtkCore.log_so3(SMatrix{3, 3}(b.rotations[k])' * q.R[k]))), 1:3) < 0.05
    # 0.5.1's bundle: the same tracks and start through the bare model (this lens's projection is defined everywhere, so
    # nothing masks its residual, as in 0.5.1)
    @test VgtkCore.valid_domain(VgtkCore.model(bare)) isa VgtkCore.WholePlane
    obs = [[k => PanoramaStitching._camera_keypoint(FRAME_MAP, b.features[k][kp].detection) for (k, kp) in t] for t in b.tracks]
    ref = refine_rotations(VgtkCore.model(bare), VgtkCore.estimate(b.averaging), obs; merge(bkw, (; optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}()))..., width = b.bundle.metadata.loss_width)
    @test VgtkCore.converged(ref)
    @test maximum(k -> norm(SMatrix{3, 3}(VgtkCore.estimate(ref)[k]) - SMatrix{3, 3}(b.rotations[k])), 1:3) <= 1.0e-9
end

@testitem "the camera is the projection: no `projection` keyword" begin
    # a grid lens is a SplineLens camera that projects through its own forward spline, so the registration and the bundle take
    # the camera alone
    using Test
    using PanoramaStitching: register_rotations, refine_rotations
    for f in (register_rotations, refine_rotations)
        @test all(m -> :projection ∉ Base.kwarg_decl(m), methods(f))
    end
end

@testitem "Review focus 1: a whole-sensor grid camera renders no taller than its parametric lens" setup = [BayerMaps] begin
    # The Tamron case: a rectilinear lens whose distortion folds inside the sensor (its view-angle limit 64°, the fold's circle
    # cutting the sensor's corners), as a SplineLens grid calibrated over the whole sensor with no stored coverage. Its
    # cylindrical panorama is no taller than the parametric lens's + 10 %. Neither `fov_deg` (the same rows at 120° and 179°)
    # nor the view (±85°) is the cap: the lens is. A projection that images past the lens's limit fails this (an equidistant
    # map of the same focal length over the same sensor: 353 rows against 253, measured 2026-11-10).
    using Test, StaticArrays
    using PanoramaStitching: render_panorama, panorama_view
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    W, H, s = 1000, 800, 2
    lens = VgtkCore.Lens(VgtkCore.Perspective(), VgtkCore.Radial((-0.08,)))
    rect = VgtkCore.CameraModel(400.0VgtkCore.px, SA[500.0VgtkCore.px, 400.0VgtkCore.px]; lens)
    bare = VgtkCore.BoundedCamera(rect, VgtkCore.Size2(W, H))
    @test VgtkCore.valid_domain(rect) isa VgtkCore.Ellipse                          # the fold cuts the sensor
    scene(d) = 0.1 + 0.05 * sin(5d[1]) * cos(3d[2])
    R = [SMatrix{3, 3}(VgtkCore.exp_so3(SVector(0.0, deg2rad(35.0 * k), 0.0))) for k in 0:2]
    frames = map(R) do Rk
        img = zeros(Float32, H ÷ s, W ÷ s, 3)
        for j in 1:(W ÷ s), i in 1:(H ÷ s)
            f = VgtkCore.backproject(rect, SVector((j - 0.5) * s, (i - 0.5) * s))
            all(isfinite, f) && (img[i, j, :] .= Float32(scene(Rk * f)))
        end
        img
    end
    grid = VgtkCore.with_model(bare, VgtkCore.estimate(VgtkEstimation.fit_camera(VgtkCore.SplineLens, bare, VgtkCore.imaged_region(bare);
                                                                                 tolerance = 1.0e-2, grid_size = (40, 32))))
    cyl = panorama_view(VgtkCore.Cylindrical(), (-100.0, 170.0, -85.0, 85.0), 0.5)
    rows(r) = count(i -> any(>(0), view(r.owner, i, :)), axes(r.owner, 1))
    kw(fov) = (; vignetting = (u, f) -> 1.0, frame_map = FRAME_MAP, rim_margin_px = 0.0, fov_deg = fov)
    p = rows(render_panorama(bare, frames, R, cyl; kw(120.0)...))
    g = rows(render_panorama(grid, frames, R, cyl; kw(120.0)...))
    @info "Review focus 1: covered rows, parametric $p, grid $g (of $(cyl.size[1]))"
    @test 0 < p < cyl.size[1] && g <= 1.1 * p
    @test rows(render_panorama(grid, frames, R, cyl; kw(179.0)...)) == g      # the field-angle cap does not bound it
end

@testitem "frame_region: the margin erodes the lens coverage's edge only" begin
    # a crop coverage loses the margin on its sides inside the sensor, not on the side the sensor's edge cuts; a margin that
    # erodes a coverage away leaves an empty region (nothing imaged), not an error
    using Test, StaticArrays
    using PanoramaStitching: PanoramaStitching
    import VisualGeometryToolkitCore as VgtkCore
    m = VgtkCore.CameraModel(300.0VgtkCore.px, SA[500.0VgtkCore.px, 400.0VgtkCore.px]; lens = VgtkCore.KannalaBrandt(-0.02))
    crop = VgtkCore.BoundedCamera(m, VgtkCore.Size2(1000, 800); lens_coverage = VgtkCore.Rect2{Float64}(0, 100, 900, 700))
    r = PanoramaStitching.frame_region(crop, 1000, 800, 20.0)
    @test SVector(3.0, 400.0) ∈ r                                   # the left side is the sensor's edge: 2 px inset only
    @test SVector(500.0, 115.0) ∉ r && SVector(500.0, 125.0) ∈ r     # the top side lies inside the sensor: eroded by 20 px
    @test SVector(885.0, 400.0) ∉ r && SVector(875.0, 400.0) ∈ r     # the right side too
    @test SVector(500.0, 797.0) ∈ r                                 # the bottom is the sensor's edge
    small = VgtkCore.BoundedCamera(m, VgtkCore.Size2(1000, 800); lens_coverage = VgtkCore.Circle(VgtkCore.Point2(500.0, 400.0), 15.0))
    e = PanoramaStitching.frame_region(small, 1000, 800, 20.0)
    @test SVector(500.0, 400.0) ∉ e && !any(VgtkCore.pixel_mask(e, 100, 80))
end

@testitem "render_panorama, sample_frames: nothing outside the camera's imaged region" setup = [SyntheticSequence, BayerMaps] begin
    # the camera's lens coverage bounds every mask: with a smaller, off-centre image circle every pixel the render shows and every
    # direction the sampling sees lies at a sensor point inside that circle less the rim margin, and fewer pixels are seen. The
    # circle is a lit region, not a limit of the lens: the forward map is finite and on the sensor well beyond it (so only the
    # coverage term of the masks keeps those pixels out; without it this item fails, checked 2026-11-10)
    using Test, StaticArrays
    using PanoramaStitching: render_panorama, sample_frames, panorama_view
    import VisualGeometryToolkitCore as VgtkCore
    q = sequence(; K = 2)
    small = VgtkCore.with_lens_coverage(q.model, VgtkCore.Circle(VgtkCore.Point2(560.0, 470.0), 300.0))
    inside(u) = hypot(u[1] - 560.0, u[2] - 470.0) <= 300.0 - 12.0 + 1.0e-9
    beyond = VgtkCore.project(VgtkCore.model(q.model), VgtkCore.backproject(VgtkCore.model(q.model), SVector(940.0, 470.0)))
    @test all(isfinite, beyond) && !inside(beyond) && VgtkCore.valid_domain(VgtkCore.model(q.model)) isa VgtkCore.WholePlane
    eq = panorama_view(VgtkCore.Equirectangular(), (-80.0, 120.0, -60.0, 60.0), 0.5)
    kw = (; vignetting = (u, f) -> 1.0, frame_map = FRAME_MAP, rim_margin_px = 12.0, fov_deg = 89.0)
    r = render_panorama(small, q.frames, q.R, eq; kw...); full = render_panorama(q.model, q.frames, q.R, eq; kw...)
    seen = findall(>(0), r.owner)
    @test !isempty(seen) && length(seen) < count(>(0), full.owner)
    project(k, I) = VgtkCore.project(VgtkCore.model(q.model), SMatrix{3, 3}(q.R[k])' * eq.rays(SVector(I[2] - 0.5, I[1] - 0.5)))
    @test all(I -> inside(project(r.owner[I], I)), seen)
    D = [eq.rays(SVector(I[2] - 0.5, I[1] - 0.5)) for I in CartesianIndices(r.owner)[1:7:end]]
    S = sample_frames(small, q.frames, q.R, D; saturated = [falses(size(f, 1), size(f, 2)) for f in q.frames], kw...)
    @test count(S.seen) > 0 && all(i -> inside(S.pixels[i]), findall(S.seen))
end

@testitem "render_panorama: a grid lens's SplineLens camera renders on the device exactly as on the host" setup = [SyntheticSequence, BayerMaps] begin
    # the warp's map wraps the projection in a BoundedMap and a ComposedTransformation chain, which reach image_warp!'s kernel by
    # Core's Adapt rules (the spline's control grids moved to the device); the planes' maps give each plane its own warp
    using Test, StaticArrays, CUDA
    using PanoramaStitching: render_panorama, panorama_view
    import VisualGeometryToolkitCore as VgtkCore
    if !CUDA.functional()
        @test_skip "no CUDA device"
    else
        q = sequence(; K = 3)
        @test VgtkCore.lens(VgtkCore.model(q.cam)) isa VgtkCore.SplineLens
        eq = panorama_view(VgtkCore.Equirectangular(), (-80.0, 150.0, -60.0, 60.0), 0.5)
        kw = (; vignetting = (u, f) -> 1.0, frame_map = FRAME_MAP, rim_margin_px = 12.0, fov_deg = 89.0,
              plane_maps = PLANE_MAPS)
        h = render_panorama(q.cam, q.frames, q.R, eq; kw...)
        d = render_panorama(q.cam, CuArray.(q.frames), q.R, eq; kw...)
        # Review focus 5: the image within 1e-4 relative (the plan's bound; a device's contracted FMAs may differ from the host in
        # the last bits), owner and coverage on all but 1e-4 of the pixels (a near-tie of a mask can flip)
        @test count(>(0), h.owner) > 0.4 * length(h.owner)
        @test count(Array(d.owner) .!= h.owner) <= 1.0e-4 * length(h.owner)
        @test count(Array(d.coverage) .!= h.coverage) <= 1.0e-4 * length(h.coverage)
        same = repeat(Array(d.owner) .== h.owner, 1, 1, 3)                 # a pixel whose owner flipped holds another frame's value
        di = Array(d.image); both = same .& .!isnan.(h.image) .& .!isnan.(di)
        @test count(isnan.(di) .!= isnan.(h.image)) <= 3.0e-4 * length(h.image)
        δ = maximum(abs.(di[both] .- h.image[both]))
        @info "Review focus 5: GPU against CPU render, max |difference| $δ (image max $(maximum(h.image[both])))"
        @test δ <= 1.0e-4 * maximum(abs, h.image[both])
    end
end

# The Student-t keyword rule (owner ruling 2026-11-11): each scorer takes its own keyword, the Student-t + uniform scorer its
# `outlier_halfwidth` and never a `threshold`, a threshold scorer its `threshold` and never an `outlier_halfwidth`.
# PanoramaStitching has no Student-t path: a pair's rotation is scored by MSAC, a threshold scorer, at the example's
# `[registration] scoring_threshold` (sigmas of the keypoints' unit noise). That value reaches the scorer exactly, and neither
# scorer takes the other's keyword.
@testitem "the rotation's scorer: MSAC at its threshold, no outlier half-width" begin
    using TOML
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using PanoramaStitching: PanoramaStitching
    dir = joinpath(pkgdir(PanoramaStitching), "examples", "panorama_vignetting")
    for f in ("config_fx3_14mm.toml", "config_fx3_8mm_rooftop.toml")
        reg = TOML.parsefile(joinpath(dir, f))["registration"]
        @test !any(k -> occursin("halfwidth", k), keys(reg))
        t = Float64(reg["scoring_threshold"])
        @test VgtkCore.MsacScoring(VgtkEstimation.RotationProblem; threshold = t).threshold === t
    end
    @test_throws MethodError VgtkCore.MsacScoring(VgtkEstimation.RotationProblem; outlier_halfwidth = 5.0)
    @test_throws MethodError VgtkCore.StudentTUniformScoring(VgtkEstimation.RotationProblem, 5.0; Symbol("thres" * "hold") => 5.0,
        scale_prior = VgtkCore.JeffreysPrior(), label_prior = VgtkCore.FlatLabelPrior())
end
