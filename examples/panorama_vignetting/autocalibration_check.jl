# The 14 mm autocalibration record, reproduced under the robust-estimation contract: a CHECK, exit 1 on failure.
#
#   julia --project=examples/panorama_vignetting examples/panorama_vignetting/autocalibration_check.jl \
#         <fx3-14mm-selfcal.cameras-v5.json> [<updated record out>]
#
# The record (the desktop's, PanoramaStitching 7bbbea03) is the KB camera of the 14 mm indoor sequence (panorama_good/14mm,
# 11 frames, bin2) autocalibrated by the rotation bundle with f, cx, cy, k1…k4 free (aspect and skew held, frame 1's
# rotation the gauge), Cauchy at its loss width, bootstrapped over six rounds of register → autocalibrate → re-register from
# the night-star KB (panorama_good/derived/fx3-14mm-kb_fit.cameras-v4.json). Here ONE round is rebuilt: the registration
# with the record's camera and the record's registration settings (the round's input), then `refine_camera_and_rotations`
# from the same start with the Triggs{Cauchy} strategy at an explicit `width` = the record's loss_width_px. Its tracks are
# today's registration's, not the bootstrap's, so the parameters agree to the record's own standard deviations, not bit
# for bit: each |Δ| ≤ 3σ (k = 3 for the one rebuilt round; the coordinator's ruling). LoMa runs on the CPU
# (execution_provider = :cpu), so the check needs the frames under $VGTK_DATA_ROOT, no GPU.
#
# With a second argument the record is written there with its provenance naming this reproduction's function; its values
# and its lens_coverage (still marked PLACEHOLDER) are the record's own.
include(joinpath(@__DIR__, "run.jl"))
using Printf: @printf

function autocalibration_check(record_path, out_path = nothing)
    rec = JSON.parsefile(record_path)
    cam = rec["provenance"]["cam"]
    C = load_config(joinpath(@__DIR__, "config_fx3_14mm.toml"))          # the inputs, the decode and the start camera of record
    log = devnull
    inp = decoded_frames(C; masks = true, log)
    Hf, Wf, _ = size(inp.frames[1])
    start_camera = load_lens(C; raw_width = 2Wf, raw_height = 2Hf)        # the night-star KB, the record's start
    record_camera = VgtkIO.load(VgtkCore.BoundedCamera, record_path; name = "cam")
    # the round's registration: the record's camera and its [registration] settings (config_fx3_14mm_selfcal.toml)
    vp = verify_pairs(inp.frames, record_camera; frame_map = inp.frame_map, saturated = inp.saturated, tone_percentile = 99.5,
                      detector_margin_px = 30.0, asift_tilts = asift_tilts("imas25"), max_detections = 2048,
                      descriptor = :b, matcher = :b128, min_score = 0.0, pairs = :all, max_lag = 1, overlap_deg = 120.0,
                      min_inliers = 25, seed = 1,
                      scoring = VgtkCore.MsacScoring(VgtkEstimation.RotationProblem; threshold = 5.0),
                      max_trials = 10000, min_trials = 100, averaging_loss = VgtkCore.HuberShape(), execution_provider = :cpu)
    reg = register_rotations(vp, record_camera; averaging_loss = VgtkCore.HuberShape(), cycle_tol = nothing,
                             orientation_tol = nothing, bundle_loss = VgtkCore.CauchyShape(), bundle_efficiency_constant = 3.0,
                             bundle_max_iter = 100, bundle_ftol = 1.0e-10, bundle_xtol = 1.0e-10)
    tracks = [[k => vp.features[k].pixels[kp] for (k, kp) in t] for t in reg.tracks]
    rotations = Dict(k => SMatrix{3, 3, Float64, 9}(R) for (k, R) in enumerate(reg.rotations))
    res = refine_camera_and_rotations(start_camera, rotations, tracks;
                                      optimization_strategy = VgtkEstimation.Triggs{VgtkCore.CauchyShape}(
                                          efficiency_constant = VgtkCore.efficiency_constant(VgtkCore.CauchyShape(), 2)),
                                      width = Float64(cam["loss_width_px"]), fixed = (:aspect,),
                                      max_iter = 100, ftol = 1.0e-10, xtol = 1.0e-10)
    names = Tuple(Symbol.(cam["covariance_order"]))
    got = NamedTuple{VgtkCore.parameter_names(VgtkCore.model(res.metadata.camera))}(
        Tuple(VgtkCore.parameter_vector(VgtkCore.model(res.metadata.camera))))
    rows = [(; p, record = Float64(cam["parameters"][String(p)]), got = Float64(getproperty(got, p)),
             sigma = Float64(cam["sigma"][String(p)])) for p in names]
    println("autocalibration check: $(res.retcode), $(res.iterations) iterations, rms $(round(res.rms; digits = 4)) px " *
            "(record $(round(cam["rms_px"]; digits = 4))), $(res.n) observations (record $(cam["n_observations"]))")
    println("| parameter | record | reproduced | Δ | σ (record) | Δ/σ |")
    println("|---|---|---|---|---|---|")
    for r in rows
        d = r.got - r.record
        @printf("| %s | %.6g | %.6g | %+.4g | %.4g | %+.3f |\n", r.p, r.record, r.got, d, r.sigma, d / r.sigma)
    end
    bad = [r.p for r in rows if !(abs(r.got - r.record) <= 3 * r.sigma)]
    above1 = [r.p for r in rows if abs(r.got - r.record) > r.sigma]
    isempty(above1) || println("above 1σ: ", join(String.(above1), ", "))
    if !isempty(bad)
        println("FAIL: |Δ| > 3σ for ", join(String.(bad), ", "))
        return false
    end
    println("PASS: every |Δ| ≤ 3σ")
    if out_path !== nothing
        prov = rec["provenance"]
        prov["method"] = "rotating-camera autocalibration (no target; Hartley 1994; Agapito, Hayman, Reid 2001): " *
                         "PanoramaStitching's refine_camera_and_rotations, KB (f, cx, cy, k1..k4 free; aspect held, skew held by " *
                         "construction), the first frame's rotation the one gauge; values of the desktop's record (PanoramaStitching " *
                         "7bbbea03's rotation bundle with the camera free), reproduced to its sigmas by " *
                         "examples/panorama_vignetting/autocalibration_check.jl"
        prov["code"] = "PanoramaStitching refine_camera_and_rotations (ps-09 $(readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)));" *
                       " record of record: PanoramaStitching 7bbbea03 (" * String(prov["code"]) * ")"
        prov["reproduction"] = Dict("parameter" => [String(r.p) for r in rows], "reproduced" => [r.got for r in rows],
                                    "delta_over_sigma" => [(r.got - r.record) / r.sigma for r in rows], "tolerance_sigmas" => 3,
                                    "rms_px" => res.rms, "n_observations" => res.n)
        startswith(String(prov["lens_coverage"]), "PLACEHOLDER") || error("the record's lens_coverage lost its PLACEHOLDER mark")
        open(io -> JSON.print(io, rec, 2), out_path, "w")
        println("updated record: $out_path")
    end
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    1 <= length(ARGS) <= 2 || error("usage: julia --project=<env> autocalibration_check.jl <record.cameras-v5.json> [<out.json>]")
    autocalibration_check(ARGS...) || exit(1)
end
