# =============================================================================
# register_rotations' execution_provider: LoMa's ONNX Runtime provider for the descriptor and the matcher (one session each per
# registration), resolved by VisualGeometryToolkitFeatures' resolve_execution_provider. The keyword's refusal here;
# :auto's fallback and LoMa on CUDA against the CPU are VisualGeometryToolkitFeatures' tests (onnx_runtime_tests.jl).
# =============================================================================

@testitem "register_rotations: execution_provider other than :auto, :cpu or :cuda is refused" begin
    using Test, StaticArrays
    using PanoramaStitching: PanoramaStitching, register_rotations
    import VisualGeometryToolkitCore as VgtkCore
    cam = VgtkCore.BoundedCamera(VgtkCore.CameraModel(300.0VgtkCore.px, SA[64.0VgtkCore.px, 64.0VgtkCore.px]; lens = VgtkCore.KannalaBrandt(-0.02)),
                                 VgtkCore.Size2(128, 128))
    kw = (; raw_scale = 1, tone_percentile = 99.0, detector_margin_px = 0.0, asift_tilts = nothing,
          max_detections = 16, descriptor = :b, matcher = :b128, min_score = 0.0, pairs = :all, max_lag = 1, overlap_deg = 90.0,
          min_inliers = 10, seed = 1, max_trials = 10, min_trials = 1, averaging_loss = VgtkCore.HuberShape(),
          bundle_loss = VgtkCore.CauchyShape(), bundle_scale_multiple = 3.0, bundle_iterations = 1, bundle_max_iter = 1,
          bundle_ftol = 1.0e-8, bundle_xtol = 1.0e-8)
    frames = [rand(128, 128, 3) for _ in 1:2]
    @test_throws ArgumentError register_rotations(frames, cam; kw..., execution_provider = :tensorrt)
    @test_throws ArgumentError register_rotations(frames, cam; kw..., bundle_whitening = :diagonal)
    vkw = Base.structdiff(kw, NamedTuple{(:bundle_loss, :bundle_scale_multiple, :bundle_iterations, :bundle_max_iter, :bundle_ftol, :bundle_xtol)})
    @test_throws ArgumentError PanoramaStitching.verify_pairs(frames, cam; vkw..., execution_provider = :tensorrt)
    @test_throws ArgumentError PanoramaStitching.verify_pairs(frames, cam; vkw..., pairs = :every)
end
