# The example's results of record written end to end — a CHECK (exit 1 on failure), run by run_checks.jl in this environment:
#
#   julia --project=examples/panorama_vignetting examples/panorama_vignetting/write_results_check.jl
#
# `write_results` on a tiny fixture (a fisheye camera, three frames' rotations as the registration hands them over —
# `SMatrix{3, 3}` —, the bundle's termination record, a radial V and gains): the vgio/panorama file reads back with the
# frames' rotations, zero translations, V, the gains and the record, and V's raster reads back over the frame's pixels. 0.6.4 failed here on every run (a pose built from a plain Vector).
include(joinpath(@__DIR__, "run.jl"))
using Test

function write_results_check()
    model = VgtkCore.CameraModel(260.0VgtkCore.px, SA[400.0VgtkCore.px, 400.0VgtkCore.px]; lens = VgtkCore.KannalaBrandt(-0.02))
    lens = VgtkCore.BoundedCamera(model, VgtkCore.Size2(800, 800); lens_coverage = VgtkCore.Circle(VgtkCore.Point2(400.0, 400.0), 390.0))
    rotations = [SMatrix{3, 3, Float64, 9}(VgtkCore.exp_so3(SVector(0.0, deg2rad(y), 0.0))) for y in (0.0, 30.0, 60.0)]
    v = VgtkCore.LogVignetting(VgtkCore.with_control_points(VgtkCore.RadialVignetting(4, SVector(400.0, 400.0), 420.0),
                                                            [0.0, -0.1, -0.2, -0.3]))
    G = [1.0 1.0 1.0; 1.1 1.05 0.95; 0.9 0.92 1.02]
    res = VgtkCore.FitResult(v, VgtkCore.ReturnCode.Success, VgtkCore.IterationCap, 1, 0.0, 0, (; gains = G))
    C = (; run = mktempdir(), cfg = Dict{String, Any}("vignetting" => Dict{String, Any}("gain_correction" => "channel")))
    frame_map = u -> 2.0 .* u                                         # frame pixels → camera pixels (2×2 bins)
    names = ["a.ARW", "b.ARW", "c.ARW"]
    bundle_fit = VgtkCore.FitResult(nothing, VgtkCore.ReturnCode.Success, VgtkCore.IterationCap, 7, 0.4, 120, nothing)
    # an outermost @testset that records a failure or an error throws Test.TestSetException when it finishes: the check's false
    ok = try
    @testset "write_results: the results of record end to end" begin
        @test write_results(C, "panorama", lens, names, rotations, bundle_fit, res, frame_map, (400, 400)) === nothing
        b = VgtkIO.load_panorama(joinpath(C.run, "panorama.json"))
        @test all(maximum(abs, SMatrix{3, 3, Float64, 9}(b.rotations[k].R) - rotations[k]) < 1.0e-15 for k in 1:3)
        @test all(iszero, b.rotations[k].t for k in 1:3)
        @test b.rotations.images == Dict(enumerate(names))
        @test VgtkCore.control_points(b.vignetting) == VgtkCore.control_points(v)
        @test b.gains == G
        @test (VgtkCore.retcode(b.fit), VgtkCore.stop_rule(b.fit), b.fit.iterations, b.fit.rms, b.fit.n) ==   # the fixture's record, read back
              (VgtkCore.retcode(bundle_fit), VgtkCore.stop_rule(bundle_fit), bundle_fit.iterations, bundle_fit.rms, bundle_fit.n)
        r = VgtkIO.load_raster(joinpath(C.run, "panorama_map.json"))
        @test size(r.values) == (400, 400) && eltype(r.values) == Float32
        @test r.values[17, 33] == Float32(v(frame_map(SVector(33 - 0.5, 17 - 0.5))))
    end
    true
    catch e
        e isa Test.TestSetException || rethrow()
        false
    end
    return ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_results_check() || exit(1)
end
