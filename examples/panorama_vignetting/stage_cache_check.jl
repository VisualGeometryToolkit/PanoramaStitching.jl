# The example's stage cache — a CHECK (exit 1 on failure), run by run_checks.jl in this environment (it needs
# VisualGeometryToolkitEval, which is not a dependency of PanoramaStitching's own tests):
#
#   julia --project=examples/panorama_vignetting examples/panorama_vignetting/stage_cache_check.jl
#
# The verified pairs are a cached stage (Eval's cached_result, keyed by Eval's run_key of the stage's settings): REUSED when only
# the averaging or the bundle changes, recomputed when a matching or detector setting, LoMa's provider or a RAW file changes;
# the registration is keyed by every [registration] setting; the record reads back exactly. Every file goes into temporary data
# and results roots.
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
    strat(t, l, k, v) = Dict{String, Any}("type" => t, "loss" => l, k => v)
    cfg(; kw...) = Dict{String, Any}("inputs" => Dict{String, Any}("raw_dir" => "raw", "raw_extension" => ".ARW"),
                                     "decode" => Dict{String, Any}("mode" => "bin2", "gain" => 1.0, "cache" => false),
                                     "registration" => merge(Dict{String, Any}("pairs" => "all", "min_inliers" => 25, "seed" => 1,
                                                                               "averaging_optimization_strategy" => strat("irls", "huber", "max_iter", 100),
                                                                               "bundle_optimization_strategy" => strat("triggs", "cauchy", "efficiency_constant", 3.0)),
                                                             Dict{String, Any}(string(k) => v for (k, v) in kw)))
    inp = (; names = ["a.ARW", "b.ARW"])
    # an outermost @testset that records a failure or an error throws Test.TestSetException when it finishes (every Julia):
    # that is the check's false (Test.anynonpass, the earlier test, is not in Julia 1.12's Test)
    ok = try
    @testset "the example's stage cache" begin
        withenv("VGTK_DATA_ROOT" => root, "VGTK_RESULTS_ROOT" => results) do
            out = mktempdir(results)
            matched = Ref(0)                                                  # how often the matching ran
            compute() = (matched[] += 1; merge!(pairs_record(vp), Dict{String, Any}("seconds" => 1.0)))
            stage(c) = staged_record(compute, "verified_pairs", (; cfg = c, out), pairs_stage_settings((; cfg = c, out), inp, :cuda);
                                     log = devnull, what = "verified pairs")
            rec, file = stage(cfg())
            @test matched[] == 1 && isfile(file) && startswith(basename(file), "verified_pairs_")
            @test first(stage(cfg())) == rec && matched[] == 1                # a second run with the same matching settings skips it
            back = pairs_from_record(rec)                                     # the record reads back exactly
            @test back isa VerifiedPairs && back.verified == vp.verified && back.inlier_matches == vp.inlier_matches &&
                  back.stats == vp.stats && back.scales == vp.scales
            prov(d, k) = string(haskey(d, k) ? d[k] : d[Symbol(k)])                  # LoMa's provenance, as recorded
            @test prov(back.loma.descriptor, "execution_provider") == "cuda" && prov(back.loma.matcher, "model") == "loma_b128"
            @test all(k -> back.features[k] == vp.features[k], 1:2)          # the described features as they were
            key(f, c; provider = :cuda) = VgtkEval.Experiments.run_key(f((; cfg = c, out), inp, provider))
            # the provider LoMa runs on keys both stages: a CPU record is not a CUDA run's
            @test key(pairs_stage_settings, cfg(); provider = :cpu) != key(pairs_stage_settings, cfg()) &&
                  key(registration_settings, cfg(); provider = :cpu) != key(registration_settings, cfg())
            # a changed averaging or bundle setting: a new registration, the SAME pairs (reused: not matched again)
            for c in (cfg(; averaging_optimization_strategy = strat("irls", "cauchy", "max_iter", 100)),
                      cfg(; bundle_optimization_strategy = strat("triggs", "cauchy", "efficiency_constant", 2.5)),
                      cfg(; bundle_optimization_strategy = strat("triggs", "huber", "efficiency_constant", 3.0)),
                      cfg(; cycle_tol_deg = 1.0), cfg(; orientation_tol_deg = 3.0))
                @test key(registration_settings, c) != key(registration_settings, cfg()) &&
                      key(pairs_stage_settings, c) == key(pairs_stage_settings, cfg())
                stage(c)
            end
            @test matched[] == 1
            # a changed matching setting, or a changed RAW file: NOT reused, matched again
            stage(cfg(; min_inliers = 30)); @test matched[] == 2
            write(joinpath(root, "raw", "b.ARW"), "b, rewritten"); stage(cfg(; min_inliers = 30)); @test matched[] == 3
            # the detector's keys key the pairs
            for c in (cfg(; selection = "spread"), cfg(; detection_image = "vst"), cfg(; max_detections = 4096))
                @test key(pairs_stage_settings, c) != key(pairs_stage_settings, cfg())
            end
            # pairs = "lag_overlap" aims the pairs past the lag by the rotation averaging: its strategy keys the pairs there
            @test key(pairs_stage_settings, cfg(; pairs = "lag_overlap", averaging_optimization_strategy = strat("irls", "cauchy", "max_iter", 100))) !=
                  key(pairs_stage_settings, cfg(; pairs = "lag_overlap"))
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
