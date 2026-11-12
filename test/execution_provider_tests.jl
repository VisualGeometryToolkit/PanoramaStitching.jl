# =============================================================================
# register_rotations' execution_provider: LoMa's ONNX Runtime provider for the descriptor and the matcher (one session each per
# registration), resolved by VisualGeometryToolkitFeatures' resolve_execution_provider. The keyword's refusal here;
# :auto's fallback and LoMa on CUDA against the CPU are VisualGeometryToolkitFeatures' tests (onnx_runtime_tests.jl).
# =============================================================================

@testitem "register_rotations: execution_provider other than :auto, :cpu or :cuda is refused" setup = [BayerMaps] begin
    using Test, StaticArrays
    using PanoramaStitching: PanoramaStitching, register_rotations
    import VisualGeometryToolkitCore as VgtkCore
    cam = VgtkCore.BoundedCamera(VgtkCore.CameraModel(300.0VgtkCore.px, SA[64.0VgtkCore.px, 64.0VgtkCore.px]; lens = VgtkCore.KannalaBrandt(-0.02)),
                                 VgtkCore.Size2(128, 128))
    kw = (; frame_map = IDENTITY_MAP, tone_percentile = 99.0, detector_margin_px = 0.0, asift_tilts = :imas25,
          max_detections = 16, descriptor = :b, matcher = :b128, min_score = 0.0, pairs = :all, max_lag = 1, overlap_deg = 90.0,
          min_inliers = 10, seed = 1, max_trials = 10, min_trials = 1, averaging_optimization_strategy = VgtkCore.Irls{VgtkCore.HuberShape}(max_iter = 100),
          bundle_optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(efficiency_constant = 3.0),  bundle_max_iter = 1,
          bundle_ftol = 1.0e-8, bundle_xtol = 1.0e-8)
    frames = [rand(128, 128, 3) for _ in 1:2]
    @test_throws ArgumentError register_rotations(frames, cam; kw..., execution_provider = :tensorrt)
    @test_throws ArgumentError register_rotations(frames, cam; kw..., bundle_whitening = :diagonal)
    vkw = Base.structdiff(kw, NamedTuple{(:bundle_optimization_strategy, :bundle_max_iter, :bundle_ftol, :bundle_xtol)})
    @test_throws ArgumentError PanoramaStitching.verify_pairs(frames, cam; vkw..., execution_provider = :tensorrt)
    @test_throws ArgumentError PanoramaStitching.verify_pairs(frames, cam; vkw..., pairs = :every)
end
