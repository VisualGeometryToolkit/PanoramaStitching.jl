# =============================================================================
# register_rotations' execution_provider: LoMa's ONNX Runtime provider for the descriptor and the matcher (one session each per
# registration), resolved by VisualGeometryToolkitFeatures' resolve_execution_provider. The keyword's refusal here;
# :auto's fallback and LoMa on CUDA against the CPU are VisualGeometryToolkitFeatures' tests (onnx_runtime_tests.jl).
# =============================================================================

@testitem "register_rotations: execution_provider other than :auto, :cpu or :cuda is refused" setup = [BayerMaps] begin
    using Test, StaticArrays
    using PanoramaStitching: PanoramaStitching, register_rotations
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    cam = VgtkCore.BoundedCamera(VgtkCore.CameraModel(300.0VgtkCore.px, SA[64.0VgtkCore.px, 64.0VgtkCore.px]; lens = VgtkCore.KannalaBrandt(-0.02)),
                                 VgtkCore.Size2(128, 128))
    kw = (; frame_map = IDENTITY_MAP, tone_percentile = 99.0, detector_margin_px = 0.0, asift_tilts = :imas25,
          max_detections = 16, descriptor = :b, matcher = :b128, min_score = 0.0, pairs = PanoramaStitching.AllPairs(),
          min_inliers = 10, seed = 1, ransac = VgtkEstimation.Ransac(; config = VgtkEstimation.RansacConfig(; max_trials = 10, min_trials = 1)),
          bundle_stops = VgtkCore.LmStops(; max_iter = 1, ftol = 1.0e-8, xtol = 1.0e-8))
    frames = [rand(128, 128, 3) for _ in 1:2]
    @test_throws ArgumentError register_rotations(frames, cam; kw..., execution_provider = :tensorrt)
    @test_throws ArgumentError register_rotations(frames, cam; kw..., Symbol("bundle_" * "whitening") => :diagonal)
    vkw = Base.structdiff(kw, NamedTuple{(:bundle_stops,)})
    @test_throws "is not :auto, :cpu or :cuda" PanoramaStitching.verify_pairs(frames, cam; vkw..., execution_provider = :tensorrt)
    @test_throws TypeError PanoramaStitching.verify_pairs(frames, cam; vkw..., pairs = :every)       # a pair selection is a type
    # PanoramaStitching 0.7's optical-axis angles, retired (0.8): refused by name, naming the view_overlap rules that replace them
    lag = PanoramaStitching.LagOverlap(; max_lag = 1)
    @test_throws r"`overlap_deg` is retired: .*any view.*min_inliers" PanoramaStitching.verify_pairs(frames, cam; vkw..., pairs = lag, overlap_deg = 60.0)
    @test_throws r"`overlap_deg` is retired" register_rotations(frames, cam; kw..., pairs = lag, overlap_deg = 60.0)
    @test_throws r"`near_duplicate_deg` is retired: .*near_duplicate_overlap" register_rotations(frames, cam; kw..., near_duplicate_deg = 4.0)
    # the settings of 0.11's flat keywords are the algorithms' now: an old keyword is unknown, refused
    @test_throws "unknown keyword `max_trials`" PanoramaStitching.verify_pairs(frames, cam; vkw..., max_trials = 10)
    @test_throws "unknown keyword `check_tracks`" register_rotations(frames, cam; kw..., check_tracks = false)
end
