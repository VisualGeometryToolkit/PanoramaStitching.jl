# A decentred lens on the device: Core 0.10.10 adapts its PolygonRegion domain (the ring a device array), and
# PanoramaStitching's masks carry the region to the kernel. Its setups are decentred_lens_tests.jl's and the suite's.

@testitem "a decentred lens: sample_frames on the device as on the host (its polygon domain in the kernel)" setup = [DecentredLens, SyntheticSequence, BayerMaps] begin
    using Test, StaticArrays, CUDA
    using PanoramaStitching: sample_frames, fibonacci_directions
    import VisualGeometryToolkitCore as VgtkCore
    if !CUDA.functional()
        @test_skip "no CUDA device"
    else
        q = sequence(; K = 2)
        cam = VgtkCore.BoundedCamera(DECENTRED, VgtkCore.Size2(1000, 1000))
        @test VgtkCore.imaged_region(cam).b isa VgtkCore.PolygonRegion
        Hf, Wf = size(q.frames[1], 1), size(q.frames[1], 2)
        D = fibonacci_directions(q.R, 4000; fov_deg = 60.0)
        kw = (; frame_map = FRAME_MAP, rim_margin_px = 12.0, fov_deg = 60.0, vignetting = (u, f) -> 1.0)
        h = sample_frames(cam, q.frames, q.R, D; saturated = [falses(Hf, Wf) for _ in 1:2], kw...)
        d = sample_frames(cam, CuArray.(q.frames), q.R, CuArray(D); saturated = [CuArray(falses(Hf, Wf)) for _ in 1:2], kw...)
        @test count(h.seen) > 0
        @test count(Array(d.seen) .!= h.seen) <= 1.0e-4 * length(h.seen)          # a near-tie of the domain test can flip
        both = Array(d.seen) .& h.seen
        @test maximum(i -> maximum(abs.(Array(d.pixels)[i] - h.pixels[i])), findall(both)) < 1.0e-6
    end
end
