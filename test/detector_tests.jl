# =============================================================================
# detect_features' variants: the selection (:spread, VisualGeometryToolkitFeatures' spread_strongest through the descriptor,
# the one selection) and the detection image (:tone; :vst, the variance-stabilized green cells of the frame's
# LinearRaw), the kept detections the described set (Features' described features, plain: no detection covariance), and the
# refusals.
# =============================================================================

@testitem "detect_features: :spread covers a dim half the global top N leaves empty; the kept detections are the described set, plain" setup = [SyntheticSequence, BayerMaps] begin
    using Test, StaticArrays, Statistics, LinearAlgebra
    using PanoramaStitching: PanoramaStitching, detect_features
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitFeatures as VgtkFeatures
    q = sequence(; K = 1)
    frame = copy(q.frames[1]); Hf, Wf = size(frame, 1), size(frame, 2)
    frame[:, 1:(Wf ÷ 2), :] .*= 0.05f0                                  # a dim left half
    N = 300
    desc = VgtkFeatures.LoMaDescriptor(:b; max_detections = N, execution_provider = :cpu)
    kw = (; frame_map = FRAME_MAP, tone_percentile = 99.5, detector_margin_px = 20.0, asift_tilts = [(1.0, 0.0)], max_detections = N)
    px = detection_pixels                                                            # the detections' centres in the camera's pixels
    sp = detect_features(q.cam, frame, desc; kw..., selection = :spread)
    @test VgtkFeatures.descriptor_matrix(sp) isa AbstractMatrix{Float32}             # Features' StructArray{DescribedFeature}, as the descriptor returns it
    @test px(detect_features(q.cam, frame, desc; kw...)) == px(sp)                  # :spread and :tone the defaults
    # detection covariances are off in this release: every detection is the detector's plain detection
    @test !any(f -> f.detection isa VgtkCore.Attributed{<:VgtkCore.Uncertain}, sp)
    @test all(p -> hypot(p[1] - 500.0, p[2] - 500.0) <= 480.0 - 20.0, px(sp))
    @test round(Int, 0.9N) <= length(sp) <= round(Int, 1.1N)
    # the oracle: the global top N by |response| of the region's detections (ASIFT on the documented tone) leaves the dim half
    # nearly empty, the spread selection covers it
    p = quantile(vec(view(frame, :, :, 2))[vec(VgtkCore.pixel_mask(PanoramaStitching._in_frame(VgtkCore.imaged_region(q.cam), FRAME_MAP), Wf, Hf))], 0.995)
    dets = VgtkFeatures.asift_detect(Float32.(sqrt.(clamp.(Float64.(view(frame, :, :, 2)) ./ p, 0, 1))); tilts = [(1.0, 0.0)])
    kept = PanoramaStitching._imaged_less_rim(q.cam, 20.0)
    inside = filter(d -> VgtkCore.Point2(FRAME_MAP(SVector{2, Float64}(d.center...))...) ∈ kept, dets)
    top = inside[sortperm(abs.(getproperty.(inside, :response)); rev = true)[1:N]]
    dim(px) = count(u -> u[1] < VgtkCore.scale(FRAME_MAP) * Wf / 2, px) / length(px)
    dim_top = dim([FRAME_MAP(SVector{2, Float64}(d.center...)) for d in top])
    @info "detect_features: the dim half holds $(round(dim(px(sp)); digits = 3)) of :spread's detections, $(round(dim_top; digits = 3)) of the global top N"
    @test dim(px(sp)) > 0.15
    @test dim_top < dim(px(sp)) / 3
    @test_throws ArgumentError detect_features(q.cam, frame, desc; kw..., selection = :global)   # :spread is the one selection
end

@testitem "detect_features: :vst on the frame's LinearRaw; the refusals" setup = [SyntheticSequence, BayerMaps] begin
    using Test, StaticArrays
    using PanoramaStitching: PanoramaStitching, detect_features, verify_pairs, register_rotations
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitFeatures as VgtkFeatures
    q = sequence(; K = 1)
    frame = 0.5f0 .* q.frames[1]; Hf, Wf = size(frame, 1), size(frame, 2)
    # the frame as an RGGB mosaic (black 512, white 16383): each photosite its plane's value, so its :bin2 LinearRaw is the frame
    black = 512.0; white = 16383.0
    code(v) = UInt16(round(black + clamp(v, 0, 1) * (white - black)))
    raw = Matrix{UInt16}(undef, 2Hf, 2Wf)
    for j in 1:Wf, i in 1:Hf
        raw[2i - 1, 2j - 1] = code(frame[i, j, 1]); raw[2i - 1, 2j] = code(frame[i, j, 2])
        raw[2i, 2j - 1] = code(frame[i, j, 2]); raw[2i, 2j] = code(frame[i, j, 3])
    end
    L = VgtkCore.LinearRaw(VgtkCore.RawMosaic(raw, UInt8[0 1; 3 2], (black, black, black, black), white, (0, 0), (2Hf, 2Wf),
                                              (; model = "synthetic")); mode = :bin2)
    N = 300
    desc = VgtkFeatures.LoMaDescriptor(:b; max_detections = N, execution_provider = :cpu)
    kw = (; frame_map = FRAME_MAP, tone_percentile = 99.5, detector_margin_px = 20.0, asift_tilts = [(1.0, 0.0)], max_detections = N)
    noise = (; electrons_per_dn = 5.63, read_noise = 0.82, noise_sigmas = 3.0)
    px = detection_pixels
    # :tone reads a LinearRaw's image as it reads the array
    @test px(detect_features(q.cam, L, desc; kw...)) == px(detect_features(q.cam, L.image, desc; kw...))
    v = detect_features(q.cam, L, desc; kw..., detection_image = :vst, detection_noise = noise)
    @test round(Int, 0.9N) <= length(v) <= round(Int, 1.1N)
    @test all(p -> hypot(p[1] - 500.0, p[2] - 500.0) <= 480.0 - 20.0, px(v))
    @test px(v) != px(detect_features(q.cam, L, desc; kw...))
    # the refusals
    @test_throws ArgumentError detect_features(q.cam, L.image, desc; kw..., detection_image = :vst, detection_noise = noise)   # no LinearRaw
    @test_throws ArgumentError detect_features(q.cam, L, desc; kw..., detection_image = :vst)                                  # no noise
    @test_throws ArgumentError detect_features(q.cam, L, desc; kw..., detection_noise = noise)                                 # inert with :tone
    @test_throws ArgumentError detect_features(q.cam, L, desc; kw..., detection_image = :log)
    @test_throws ArgumentError detect_features(q.cam, L, desc; kw..., selection = :ssc)
    # verify_pairs and register_rotations refuse before any feature is detected
    vkw = (; kw..., descriptor = :b, matcher = :b128, min_score = 0.0, pairs = :all, max_lag = 1, overlap_deg = 90.0, min_inliers = 10,
           seed = 1, max_trials = 10, min_trials = 1, averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100), execution_provider = :cpu)
    @test_throws ArgumentError verify_pairs([L.image, L.image], q.cam; vkw..., detection_image = :vst, detection_noise = noise)
    @test_throws ArgumentError verify_pairs([L, L], q.cam; vkw..., selection = :ssc)
    @test_throws ArgumentError register_rotations([L, L], q.cam; vkw..., detection_image = :vst, bundle_optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(efficiency_constant = 3.0),
                                                   bundle_max_iter = 1,
                                                  bundle_ftol = 1.0e-8, bundle_xtol = 1.0e-8)
end

@testitem "detect_features: no detection reads a clipped pixel; the tilt schedule by name" setup = [SyntheticSequence, BayerMaps] begin
    # Features' detection mask (`asift_detect(…; exclusion_mask)`): a detection within three of its scales of a clipped pixel is dropped.
    # `saturated` is the frame's clip mask (a LinearRaw's own saturation by default); `asift_tilts` takes Features'
    # `tilt_schedule` names (`:untilted` is the identity warp alone).
    using Test, StaticArrays
    using PanoramaStitching: PanoramaStitching, detect_features
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitFeatures as VgtkFeatures
    q = sequence(; K = 1)
    frame = copy(q.frames[1]); Hf, Wf = size(frame, 1), size(frame, 2)
    clip = falses(Hf, Wf); clip[150:350, 150:350] .= true                         # a clipped window in the imaged region
    N = 300
    desc = VgtkFeatures.LoMaDescriptor(:b; max_detections = N, execution_provider = :cpu)
    kw = (; frame_map = FRAME_MAP, tone_percentile = 99.5, detector_margin_px = 20.0, asift_tilts = :untilted, max_detections = N)
    px = detection_pixels
    free = detect_features(q.cam, frame, desc; kw...)
    masked = detect_features(q.cam, frame, desc; kw..., saturated = clip)
    in_window(p) = all(c -> 2 * (150 - 1) <= c <= 2 * 350, p)                   # camera pixels of the clipped window
    @test count(in_window, px(free)) > 20
    @test !any(in_window, px(masked))
    # the tilt schedule by name: :untilted is the identity tilt
    @test px(free) == px(detect_features(q.cam, frame, desc; kw..., asift_tilts = [(1.0, 0.0)]))
    @test_throws ArgumentError detect_features(q.cam, frame, desc; kw..., asift_tilts = :imas26)
    @test_throws DimensionMismatch detect_features(q.cam, frame, desc; kw..., saturated = falses(Hf + 1, Wf))
end
