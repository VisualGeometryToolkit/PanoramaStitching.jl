# The example's stage cache — a CHECK (exit 1 on failure), run by run_checks.jl in this environment (it needs
# VisualGeometryToolkitEval, which is not a dependency of PanoramaStitching's own tests):
#
#   julia --project=examples/panorama_vignetting examples/panorama_vignetting/stage_cache_check.jl
#
# The verified pairs are a stage product (Eval's Product, produced or loaded by Eval's produce): REUSED when only the averaging or
# the bundle changes, produced again when a matching or detector setting, LoMa's provider or a RAW file changes; the registration
# is keyed by every [registration] setting and by the pairs' output (Eval's Merkle chain: pairs produced again with other outputs
# re-key it, and a registration produced from the earlier pairs is no longer produced); the decode's sensor record and the lens key
# the pairs by content; the record reads back exactly. Every file
# goes into temporary data and results roots; the products record the code they ran, so the check runs on committed code (a
# fresh clone, as run_checks.jl's stage does), as every run of record.
include(joinpath(@__DIR__, "run.jl"))
using Test

function stage_cache_check()
    # a frame's features as the detector gives them: Features' described features (detections of uncertain discs, their
    # covariances over (x, y, log σ), and their descriptions)
    Σ = VgtkCore.CovMat(SMatrix{3, 3, Float64, 9}(2.0, 0.1, 0, 0.1, 3.0, 0, 0, 0, 1.0))
    F(px) = VgtkFeatures.described_features([VgtkFeatures.Detection(VgtkCore.Uncertain(VgtkCore.Disc(VgtkCore.Point2(p...), 2.0), Σ),
                                                                    (; response = 1.0)) for p in px], Float32[1 2; 3 4])
    R = SMatrix{3, 3, Float64, 9}(cosd(10), 0.0, -sind(10), 0.0, 1.0, 0.0, sind(10), 0.0, cosd(10))
    vp = VerifiedPairs([F([SA[10.25, 20.5], SA[30.0, 40.125]]), F([SA[1.0 / 3, 2.0], SA[5.0, 6.0]])],
                       Dict((1, 2) => (rotation = R, inliers = 2, covariance = SMatrix{3, 3, Float64, 9}(1.0e-6, 0, 0, 0, 1.0e-6, 0, 0, 0, 1.0e-6))),
                       Dict((1, 2) => [(1, 2), (2, 1)]),
                       Dict((1, 2) => (matches = 3, inliers = 2, rms = 0.4, rotation_deg = 10.0)),
                       Dict((1, 2) => (sigma = 0.35, dof = 4)),
                       (; descriptor = (; model = "loma_b", execution_provider = :cuda, threads = 8),
                          matcher = (; model = "loma_b128", execution_provider = :cuda, threads = 8)))
    root = mktempdir(); mkpath(joinpath(root, "raw")); results = mktempdir()
    foreach(n -> write(joinpath(root, "raw", n), n), ("a.ARW", "b.ARW"))
    write(joinpath(root, "lens.json"), "the lens")
    strat(t, l, k, v) = Dict{String, Any}("type" => t, "loss" => l, k => v)
    cfg(; kw...) = Dict{String, Any}("inputs" => Dict{String, Any}("raw_dir" => "raw", "raw_extension" => ".ARW", "lens" => "lens.json"),
                                     "decode" => Dict{String, Any}("mode" => "bin2", "gain" => 1.0, "cache" => false, "sensor_characterisation" => "none"),
                                     "registration" => merge(Dict{String, Any}("pairs" => "all", "min_inliers" => 25, "seed" => 1, "sensor_characterisation" => "none",
                                                                               "averaging_optimization_strategy" => strat("irls", "huber", "max_iter", 100),
                                                                               "bundle_optimization_strategy" => strat("triggs", "cauchy", "efficiency_constant", 3.0)),
                                                             Dict{String, Any}(string(k) => v for (k, v) in kw)))
    inp = (; names = ["a.ARW", "b.ARW"])
    # an outermost @testset that records a failure or an error throws Test.TestSetException when it finishes (every Julia):
    # that is the check's false (Test.anynonpass, the earlier test, is not in Julia 1.12's Test)
    ok = try
    @testset "the example's stage cache" begin
        withenv("VGTK_DATA_ROOT" => root, "VGTK_RESULTS_ROOT" => results) do
            E = VgtkEval.Experiments
            code = E.current_code_trees()
            C(c) = (; cfg = c, experiment = "panorama_vignetting/stage_cache_check", code, path = @__FILE__)
            matched = Ref(0)                                                  # how often the matching ran
            compute(seconds = 1.0) = (matched[] += 1; merge!(pairs_record(vp), Dict{String, Any}("seconds" => seconds)))
            pairs(c; provider = :cuda) = pairs_product(C(c), inp, provider)
            stage(c) = stage_record(produce_stage(compute, pairs(c), C(c); log = devnull, what = "verified pairs"))
            rec = stage(cfg())
            @test matched[] == 1 && isfile(E.product_output(pairs(cfg()))) && E.produced(pairs(cfg()))
            @test stage(cfg()) == rec && matched[] == 1                       # a second run with the same matching settings skips it
            back = pairs_from_record(rec)                                     # the record reads back exactly
            @test back isa VerifiedPairs && back.verified == vp.verified && back.inlier_matches == vp.inlier_matches &&
                  back.stats == vp.stats && back.scales == vp.scales
            prov(d, k) = string(haskey(d, k) ? d[k] : d[Symbol(k)])                  # LoMa's provenance, as recorded
            @test prov(back.loma.descriptor, "execution_provider") == "cuda" && prov(back.loma.matcher, "model") == "loma_b128"
            @test all(k -> back.features[k] == vp.features[k], 1:2)          # the described features as they were
            reg(c; provider = :cuda) = stage_product(C(c), "registration", registration_settings(C(c), inp, provider);
                                                     upstream = ("verified_pairs" => pairs(c; provider),))
            pkey(c; provider = :cuda) = pairs(c; provider).key
            rkey(c; provider = :cuda) = reg(c; provider).key
            # the provider LoMa runs on keys both stages: a CPU record is not a CUDA run's
            @test pkey(cfg(); provider = :cpu) != pkey(cfg()) &&
                  rkey(cfg(); provider = :cpu) != rkey(cfg())
            # a changed averaging or bundle setting: a new registration, the SAME pairs (reused: not matched again)
            for c in (cfg(; averaging_optimization_strategy = strat("irls", "cauchy", "max_iter", 100)),
                      cfg(; bundle_optimization_strategy = strat("triggs", "cauchy", "efficiency_constant", 2.5)),
                      cfg(; bundle_optimization_strategy = strat("triggs", "huber", "efficiency_constant", 3.0)),
                      cfg(; cycle_tol_deg = 1.0), cfg(; orientation_tol_deg = 3.0))
                @test rkey(c) != rkey(cfg()) &&
                      pkey(c) == pkey(cfg())
                stage(c)
            end
            @test matched[] == 1
            # the Merkle chain: a registration produced from the pairs is reused while they stand; the pairs produced again with
            # other outputs re-key it, and the registration produced from the earlier pairs is no longer produced
            r0 = reg(cfg())
            produce_stage(() -> Dict{String, Any}("rotations" => 1), r0, C(cfg()); log = devnull, what = "registration")
            @test E.produced(r0) && reg(cfg()).key == r0.key
            produce_stage(() -> compute(2.0), pairs(cfg()), C(cfg()); log = devnull, what = "verified pairs", force = true)
            @test matched[] == 2 && reg(cfg()).key != r0.key && !E.produced(r0)
            # a changed matching setting, or a changed RAW file: NOT reused, matched again
            stage(cfg(; min_inliers = 30)); @test matched[] == 3
            write(joinpath(root, "raw", "b.ARW"), "b, rewritten"); stage(cfg(; min_inliers = 30)); @test matched[] == 4
            # the detector's keys key the pairs
            for c in (cfg(; selection = "spread"), cfg(; detection_image = "vst"), cfg(; max_detections = 4096))
                @test pkey(c) != pkey(cfg())
            end
            # the files the frames and the camera are read from key the pairs by content: the decode's sensor record, the lens
            write(joinpath(root, "sensor.json"), "a record")
            recorded() = (c = cfg(); c["decode"]["sensor_characterisation"] = c["registration"]["sensor_characterisation"] = "sensor.json"; c)
            # "vst": the detector decodes again under its own record, its content keyed by itself ([decode]'s here "none")
            vst() = (c = cfg(; detection_image = "vst", noise_sigmas = 3); c["registration"]["sensor_characterisation"] = "sensor.json"; c)
            k, kv = pkey(recorded()), pkey(vst()); @test k != pkey(cfg())
            write(joinpath(root, "sensor.json"), "the record, measured again"); @test pkey(recorded()) != k && pkey(vst()) != kv
            k = pkey(cfg()); write(joinpath(root, "lens.json"), "the lens, calibrated again"); @test pkey(cfg()) != k
            # pairs = "lag_overlap" aims the pairs past the lag by the rotation averaging: its strategy keys the pairs there
            @test pkey(cfg(; pairs = "lag_overlap", averaging_optimization_strategy = strat("irls", "cauchy", "max_iter", 100))) !=
                  pkey(cfg(; pairs = "lag_overlap"))
        end
    end
    true
    catch e
        e isa Test.TestSetException || rethrow()
        false
    end
    return ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    stage_cache_check() || exit(1)
end
