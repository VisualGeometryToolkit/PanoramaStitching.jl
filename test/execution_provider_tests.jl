# =============================================================================
# register_rotations' execution_provider: LoMa's ONNX Runtime provider for the descriptor and the matcher (one session each per
# registration). The keyword's refusal; on a CUDA device (skipped without one) the CUDA sessions describe and match a synthetic
# frame pair as the CPU sessions do, to ONNX's float differences.
# =============================================================================

@testitem "register_rotations: execution_provider other than :cpu or :cuda is refused" begin
    using Test, StaticArrays
    using PanoramaStitching: register_rotations
    import VisualGeometryToolkitCore as VgtkCore
    cam = VgtkCore.CameraModel(300.0VgtkCore.px, SA[64.0VgtkCore.px, 64.0VgtkCore.px]; lens = VgtkCore.KannalaBrandt(-0.02))
    kw = (; raw_scale = 1, circle = VgtkCore.WholePlane(), tone_percentile = 99.0, detector_margin_px = 0.0, asift_tilts = nothing,
          max_detections = 16, descriptor = :b, matcher = :b128, min_score = 0.0, pairs = :all, max_lag = 1, overlap_deg = 90.0,
          min_inliers = 10, seed = 1, max_trials = 10, min_trials = 1, averaging_loss = VgtkCore.HuberShape(),
          bundle_loss = VgtkCore.CauchyShape(), bundle_scale_multiple = 3.0, bundle_iterations = 1, bundle_max_iter = 1,
          bundle_ftol = 1.0e-8, bundle_xtol = 1.0e-8)
    frames = [rand(128, 128, 3) for _ in 1:2]
    @test_throws ArgumentError register_rotations(frames, cam; kw..., execution_provider = :tensorrt)
    @test_throws ArgumentError register_rotations(frames, cam; kw..., bundle_whitening = :diagonal)
end

# cuDNN is not a test dependency (cuDNN 1.4, the last for CUDA.jl 5, does not resolve beside the workspace's
# CUDA_Runtime_Discovery 2): the item runs where the environment holds it, with a device, on CUDA's runtime 12.x (ONNX Runtime's
# CUDA provider: CUDA.set_runtime_version!(v"12.9")), and is skipped otherwise.
@testitem "LoMa on CUDA describes and matches as on the CPU (skipped without a device)" begin
    using Test, Random
    import CUDA
    import PanoramaStitching
    import VisualGeometryToolkitFeatures as VgtkFeatures
    cudnn = Base.find_package("cuDNN") === nothing ? nothing :
            Base.require(Base.PkgId(Base.UUID("02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"), "cuDNN"))
    if cudnn === nothing || !CUDA.functional() || !Base.invokelatest(cudnn.has_cudnn) || CUDA.runtime_version() >= v"13"
        @test_skip "no cuDNN, functional CUDA device or CUDA 12 runtime"
    else
        rng = Xoshiro(3)
        # a textured frame and the same frame shifted by 9 pixels, ASIFT-detected; LoMa's input is the frame itself (784²)
        base = rand(rng, Float32, 800, 800); for _ in 1:3; base .= (base .+ circshift(base, (1, 0)) .+ circshift(base, (0, 1))) ./ 3; end
        a = base[1:784, 1:784]; b = base[10:793, 1:784]
        img(x) = permutedims(repeat(x, 1, 1, 3), (3, 1, 2))
        dets(x) = VgtkFeatures.asift_detect(x; tilts = [(1.0, 0.0)])
        da, db = dets(a), dets(b)
        sz = (3, PanoramaStitching.LOMA_SIZE, PanoramaStitching.LOMA_SIZE)
        run(ep) = (d = VgtkFeatures.LoMaDescriptor(:b; max_detections = 512, execution_provider = ep);
                   m = VgtkFeatures.LoMaMatcher(:b128; execution_provider = ep);
                   fa = d(img(a), da); fb = d(img(b), db); (fa, fb, m(fa, fb, sz, sz; min_score = 0.0)))
        fa_c, fb_c, mc = run(:cpu); fa_g, fb_g, mg = run(:cuda)
        Dc = reduce(hcat, fa_c.descriptor); Dg = reduce(hcat, fa_g.descriptor)
        @test maximum(abs, Dc .- Dg) < 1.0e-2 * maximum(abs, Dc)
        pc = Set((x.first, x.second) for x in mc); pg = Set((x.first, x.second) for x in mg)
        @test length(intersect(pc, pg)) >= 0.95 * max(length(pc), length(pg))
    end
end
