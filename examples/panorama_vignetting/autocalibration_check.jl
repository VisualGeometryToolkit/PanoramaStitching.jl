# The 14 mm BUNDLE-EQUIVALENCE check — a CHECK, exit 1 on failure:
#
#   julia --project=examples/panorama_vignetting examples/panorama_vignetting/autocalibration_check.jl
#
# PanoramaStitching's `refine_camera_and_rotations` (the autocalibration of a rotating camera, under the robust-estimation
# contract) against PanoramaStitching 7bbbea03's camera-free rotation bundle — the code of the 14 mm autocalibration record
# (fx3-14mm-selfcal.cameras-v5.json) — on the SAME tracks: one 7bbbea03 round, its tracks, start rotations and result
# dumped as the declared input [inputs.equivalence_tracks] of autocalibration_check.toml. It checks that the two bundles
# agree, |Δ| ≤ 3σ per parameter with σ the record's. It is NOT a reproduction of the record: those tracks are one round,
# while the record came from a six-round bootstrap (2026-11-12: its own code reproduces it to its σ; today's registration
# gives other tracks, so the record is not re-registered here). Every input is under the data root (VGTK_DATA_ROOT), read
# only, its sha256 verified; every setting is a literal of autocalibration_check.toml; nothing is registered, so the check
# needs no GPU and no frames.
include(joinpath(@__DIR__, "run.jl"))
using Printf: @printf
import SHA

# A declared input of the check's config: its path under the data root, present and with the declared sha256.
function declared_input(cfg, id)
    t = cfg["inputs"][id]
    path = VgtkEval.Experiments.require_input(input_path(t["path"]); what = "input $id ($(t["tier"]))", provenance = t["provenance"])
    got = bytes2hex(open(SHA.sha256, path))
    got == t["sha256"] || error("input $id: $path has sha256 $got, the config declares $(t["sha256"])")
    return path
end

function autocalibration_check(config_path = joinpath(@__DIR__, "autocalibration_check.toml"))
    cfg = TOML.parsefile(config_path)
    fx = JSON.parsefile(declared_input(cfg, "equivalence_tracks"))
    start = VgtkIO.load(VgtkCore.BoundedCamera, declared_input(cfg, "start_camera"); name = cfg["inputs"]["start_camera"]["camera"])
    tracks = [[Int(o[1]) => SVector{2, Float64}(o[2]...) for o in t] for t in fx["tracks"]]
    rotations = Dict(k => SMatrix{3, 3, Float64, 9}(reshape(Float64.(v), 3, 3)) for (k, v) in enumerate(fx["rotations"]))
    b = cfg["bundle"]
    strategy = VgtkEstimation.optimization_strategy_from_toml(b["optimization_strategy"]; class = :refine_camera_and_rotations,
                                                              section = "bundle.optimization_strategy")
    res = nothing; camera = start
    for run in 1:b["runs"]
        res = refine_camera_and_rotations(camera, rotations, tracks; optimization_strategy = strategy, width = Float64(b["width"]),
                                          fixed = Tuple(Symbol.(b["fixed"])), max_iter = b["max_iter"], ftol = b["ftol"],
                                          xtol = b["xtol"])
        println("run $run: $(res.retcode), $(res.iterations) iterations, rms $(round(res.rms; digits = 4)) px, $(res.n) observations")
        camera = res.metadata.camera
        rotations = Dict(k => SMatrix{3, 3, Float64, 9}(R) for (k, R) in VgtkCore.estimate(res))
    end
    c = res.metadata.calibration
    k = cfg["check"]["tolerance_sigmas"]
    println("| parameter | 7bbbea03 (reference) | refine_camera_and_rotations | Δ | σ (record) | Δ/σ |")
    println("|---|---|---|---|---|---|")
    bad = Symbol[]
    for p in c.free_parameters
        ref = Float64(fx["reference"][String(p)]); got = Float64(c.parameters[p]); σ = Float64(fx["record_sigma"][String(p)])
        @printf("| %s | %.8g | %.8g | %+.4g | %.4g | %+.3f |\n", p, ref, got, got - ref, σ, (got - ref) / σ)
        abs(got - ref) <= k * σ || push!(bad, p)
    end
    isempty(bad) || (println("FAIL: |Δ| > $(k)σ for ", join(String.(bad), ", ")); return false)
    println("OKAY: the two bundles agree, every |Δ| ≤ $(k)σ")
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || error("usage: julia --project=<env> autocalibration_check.jl [<autocalibration_check.toml>]")
    autocalibration_check(ARGS...) || exit(1)
end
