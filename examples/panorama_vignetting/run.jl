# The panorama vignetting example: RAW frames of a camera that only rotated → their rotations → the camera's vignetting from
# the frames alone (interpolation free) → the seam scores before and after → corrected and uncorrected panoramas.
#
#     julia --project=<env> examples/panorama_vignetting/run.jl examples/panorama_vignetting/config_fx3_14mm.toml
#
# Every setting is in the config (a config of record: nothing here has a default). Inputs are read under the data root
# (VisualGeometryToolkitEval's data_root, $VGTK_DATA_ROOT); everything this writes goes under the results root
# (results_root, $VGTK_RESULTS_ROOT): <results>/panorama_vignetting/<name>/, its stages' records keyed by their settings
# (Eval's cached_result at Eval's run_key) and the run's outputs in run-<run_key(config)>/ with its run records. The stages are functions, so in a REPL
# `include("run.jl"); run_example("config.toml")` runs it all and each stage can be rerun alone. See README.md.

using LibRaw_jll                                   # Features' LibRaw decoder (its package extension)
using JLD2                                         # Features' FileCache: the decoded-RAW cache (its package extension)
using CUDA                                         # the [compute] device's kernels
# LoMa's ONNX Runtime CUDA provider (registration.execution_provider = "auto" or "cuda") takes its CUDA 12 runtime and cuDNN
# from this environment's CUDA_Runtime_jll and CUDNN_jll (VisualGeometryToolkitFeatures loads them for ONNX Runtime alone,
# whatever runtime CUDA.jl uses); "auto" runs on the CPU, warning why, where they or a device are missing.
using PanoramaStitching
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
import VisualGeometryToolkitFeatures as VgtkFeatures
import VisualGeometryToolkitIO as VgtkIO
import VisualGeometryToolkitEval as VgtkEval
using StaticArrays: SVector, SMatrix, SA
using LinearAlgebra: LinearAlgebra, norm, normalize, dot, cross, eigen, Symmetric
using Statistics: mean, median, quantile
using Printf: @sprintf
import PNGFiles, JSON, TOML

const FPN = PNGFiles.FixedPointNumbers
const CT = PNGFiles.ColorTypes

# ----------------------------------------------------------------------------- configuration

# Every input path is relative to the data root: VisualGeometryToolkitEval.Experiments' `data_root` ($VGTK_DATA_ROOT, README "Inputs").
input_path(rel) = joinpath(VgtkEval.Experiments.data_root(), rel)

# The vignetting keys of earlier releases (PanoramaStitching 0.4 renamed them): a table that holds one is refused by name, never
# read as the new setting.
const RENAMED_KEYS = ("weights" => "measurement_noise", "sampson_iterations" => "measurement_noise_iterations")
function refuse_renamed_keys(table, where)
    for (old, new) in RENAMED_KEYS
        haskey(table, old) && throw(ArgumentError("$where: the key $old was renamed to $new (PanoramaStitching 0.4; its tags too: " *
                                                  "see CHANGELOG.md)"))
    end
    return table
end

# The image circle of earlier releases (`[inputs] circle`): where light reaches the sensor is the camera's own lens coverage now,
# stored with the camera, never a setting of the run — a config that holds it is refused, never read.
function refuse_circle(inputs, where)
    haskey(inputs, "circle") && throw(ArgumentError("$where: the key circle is retired (PanoramaStitching 0.6): the image circle is " *
                                                    "the camera's lens_coverage, stored with the camera (VisualGeometryToolkitCore's " *
                                                    "BoundedCamera(model, extent; lens_coverage = Circle(…)), saved in its " *
                                                    "VisualGeometryToolkitIO camera record); remove the key"))
    return inputs
end

# The [registration] keys retired (PanoramaStitching 0.6): refused by name, never read.
const RETIRED_REGISTRATION_KEYS = ("bundle_iterations" => "the bundle is ONE LM run of its bundle_optimization_strategy at the verified pairs' " *
                                   "pooled SIMFIT++ noise scale (register_rotations); remove the key",
                                   "bundle_scale_multiple" => "the bundle's strategy is the table bundle_optimization_strategy",
                                   "bundle_efficiency_constant" => "the bundle's strategy is the table bundle_optimization_strategy (its efficiency_constant)",
                                   "bundle_loss" => "the bundle's strategy is the table bundle_optimization_strategy (its loss)",
                                   "averaging_loss" => "the averaging's strategy is the table averaging_optimization_strategy",
                                   "local_optimization" => "the pairs' RANSAC strategy table is local_optimization_strategy",
                                   "bundle_camera_free" => "the bundle holds the camera's calibration")
function refuse_retired_registration(table, where)
    for (old, why) in RETIRED_REGISTRATION_KEYS
        haskey(table, old) && throw(ArgumentError("$where: the key $old is retired: $why"))
    end
    return table
end

# The output folder of earlier releases (`output`, relative to the config): the results go under Eval's results root, keyed by
# the config — a config that names one is refused, never read.
function refuse_output(cfg, where)
    haskey(cfg, "output") && throw(ArgumentError("$where: the key output is retired: every result goes under VisualGeometryToolkitEval's " *
                                                 "results_root() (\$VGTK_RESULTS_ROOT), <results>/panorama_vignetting/<name>/, the run's " *
                                                 "outputs keyed by the config's run_key; remove the key"))
    return cfg
end

# The config of record and where its results go: `out` the set's folder under the results root (its stages' keyed records and
# the decode cache), `run` the run's outputs, keyed by the whole config (Eval's run_key), with its run records.
function load_config(path)
    cfg = TOML.parsefile(path)
    refuse_output(cfg, path)
    refuse_renamed_keys(cfg["vignetting"], "$path [vignetting]")
    refuse_circle(cfg["inputs"], "$path [inputs]")
    refuse_retired_registration(cfg["registration"], "$path [registration]")
    out = joinpath(VgtkEval.Experiments.results_root(), "panorama_vignetting", cfg["name"])
    run = joinpath(out, "run-" * VgtkEval.Experiments.run_key(cfg))
    mkpath(run)
    return (; cfg, out, run, path = abspath(path))
end

say(io, msg) = (println(msg); println(io, msg); flush(io))

# ----------------------------------------------------------------------------- the decode cache
#
# Each RAW frame is decoded once per content and decode settings by Features' `load_raw` into its one persistent cache (a
# FileCache, <output>/decoded/linear_raws.jld2; the entry is keyed by the RAW file's SHA-256, the mode, the gain and the
# decoder), and read back as decoded, value for value. The frames' and the planes' maps onto the camera's pixels are
# Features' (`raw_frame_map`, `bayer_plane_map`) of the set's first frame (all the set's frames share the Bayer phase).

"The frames as linear images (H × W × 3 Float32), their saturation masks with `masks = true`, their maps and the daylight white balance; each RAW decoded once, cached."
function decoded_frames(C; masks::Bool, log = devnull)
    dec = C.cfg["decode"]; mode = dec["mode"]; gain = Float64(dec["gain"])
    mode == "bin2" || error("decode.mode = $mode: the example runs on bin2's three planes")
    dir = joinpath(C.out, "decoded"); mkpath(dir)
    raw_dir = input_path(C.cfg["inputs"]["raw_dir"])
    raws = sort([f for f in readdir(raw_dir) if endswith(f, C.cfg["inputs"]["raw_extension"])])
    cache = VgtkFeatures.open_cache(joinpath(dir, "linear_raws.jld2"))
    lins = try
        map(raws) do f
            lin = VgtkFeatures.load_raw(joinpath(raw_dir, f); mode = Symbol(mode), gain, decoder = VgtkFeatures.LibRawDecoder(), cache)
            say(log, "  $f ($(count(lin.saturated)) saturated quads)")
            lin
        end
    finally
        VgtkFeatures.close_cache(cache)
    end
    lin1 = first(lins)
    return (; names = raws, frames = [Float32.(l.image) for l in lins], saturated = masks ? [l.saturated for l in lins] : BitMatrix[],
            frame_map = VgtkFeatures.raw_frame_map(lin1), plane_maps = ntuple(c -> VgtkFeatures.bayer_plane_map(lin1, c), 3),
            daylight_wb = collect(Float64.(lin1.metadata.daylight_wb)))
end

# ----------------------------------------------------------------------------- the lens

"""
The camera, a `BoundedCamera` of the `raw_width × raw_height` sensor: its back-projection (the fit, the registration's
bearings), its projection (the render, the sampling) and its `imaged_region` (the sensor ∩ its lens coverage ∩ its model's
valid domain: every mask). A grid lens (an old vgio grid file, read as a `SplineLens` camera whose lens coverage is the file's
domain; the file is never rewritten) gets its forward spline here, fitted over its imaged region.
"""
function load_lens(C; raw_width, raw_height)
    inp = C.cfg["inputs"]; path = input_path(inp["lens"])
    camera = if inp["lens_kind"] == "cameras"
        VgtkIO.load(VgtkCore.BoundedCamera, path; name = inp["lens_camera"])
    elseif inp["lens_kind"] == "grid"
        grid = VgtkIO.load(VgtkCore.BoundedCamera, path)                 # its domain is its lens coverage
        # its forward spline, fitted once over its imaged region and kept in the camera (Core's with_model: same sensor, coverage)
        VgtkCore.with_model(grid, VgtkCore.estimate(VgtkEstimation.fit_camera(VgtkCore.SplineLens, grid, VgtkCore.imaged_region(grid);
                                                                              tolerance = Float64(inp["spline_tolerance_px"]))))
    else
        error("inputs.lens_kind = $(inp["lens_kind"]) is not \"cameras\" or \"grid\"")
    end
    VgtkCore.extent(camera) == VgtkCore.Size2(raw_width, raw_height) ||
        error("inputs.lens: the camera's sensor is $(VgtkCore.extent(camera)), the frames' $(raw_width) × $(raw_height) camera pixels")
    return camera
end

# ----------------------------------------------------------------------------- registration (cached by its settings)

# The averaging's and the bundle's strategies: their [registration] tables, read by Estimation's one strategy reader.
averaging_strategy(r) = VgtkEstimation.optimization_strategy_from_toml(r["averaging_optimization_strategy"];
                                                                      class = :fit_rotation_averaging,
                                                                      section = "registration.averaging_optimization_strategy")
bundle_strategy(r) = VgtkEstimation.optimization_strategy_from_toml(r["bundle_optimization_strategy"]; class = :refine_rotations,
                                                                   section = "registration.bundle_optimization_strategy")

# The stages' records (the verified pairs, the registration): one JSON record each in the output folder, reused while its key —
# a SHA-256 of the record's format, the stage's settings, the ONNX Runtime provider LoMa runs on (`execution_provider = "auto"`
# resolved: a CPU record is not a CUDA run's, whose matches differ in the pairs at the network's threshold), the inputs, the
# decode settings and the RAW files' hashes — matches.
# `staged_record` is the one reader and writer: a record of other settings is another file (Eval's cached_result).
function stage_settings(C, inp, format, settings, provider::Symbol)
    return Dict{String, Any}("format" => format, "settings" => settings, "provider" => string(provider),
                             "inputs" => C.cfg["inputs"], "decode" => C.cfg["decode"],
                             "raw_sha256" => [open(VgtkCore.sha256_hex, joinpath(input_path(C.cfg["inputs"]["raw_dir"]), n)) for n in inp.names])
end
# A stage's record: VisualGeometryToolkitEval's keyed result cache (`cached_result`, keyed by Eval's `run_key` of the stage's
# settings), `<C.out>/<prefix>_<key>.jld2`, computed by `compute()` (a String-keyed Dict) when absent. Returns the record and
# its file.
function staged_record(compute, prefix, C, settings; log, what)
    key = VgtkEval.Experiments.run_key(settings)
    file = joinpath(C.out, "$(prefix)_$(key).jld2")
    isfile(file) && say(log, "$what: reused $file")
    rec, file = VgtkEval.Experiments.cached_result(_ -> compute(), prefix, settings, C.out; key = VgtkEval.Experiments.run_key,
                                                    gitpath = pkgdir(PanoramaStitching))
    return rec, file
end

# The registration's key: every [registration] setting and LoMa's provider. v3: PanoramaStitching 0.2.2 seeds each pair's sampler
# by stable_seed(seed, i, j), so a v2 record's rotations are not this code's; they are recomputed. v5: PanoramaStitching 0.6's
# bundle drops the observations not imaged at its start and counts them (`n_dropped_start`, replacing v3's start finiteness
# count), and holds the frames left with none (`frames_unconstrained`). v6: the key holds the provider LoMa runs on, and the
# record LoMa's provenance (`loma`). v7: the view-graph filters (Estimation's fit_filtered_rotation_averaging at `cycle_tol_deg` and
# `orientation_tol_deg`) before the tracks; the record holds the tolerances and the pairs they removed. v8: the bundle is one LM
# run at a multiple of the pairs' pooled SIMFIT++ scale (bundle_iterations retired); the record holds σ̂ and its iterations.
# v9: the verified pairs' v5 (the clip mask, Features' canonical detection order).
# v10: the robust-estimation contract: the averaging's and the bundle's strategies are tables (averaging_optimization_strategy,
# bundle_optimization_strategy, read by Estimation's reader), the record holds the bundle's strategy and its scale
# (bundle_scale_px), the pairs' strategy key is local_optimization_strategy.
# v11: the bundle's termination record, Core's ReturnCode and StopRule (bundle_return_code, bundle_stop_rule) in place of
# PanoramaStitching's outcome Symbol (bundle_outcome retired).
const REGISTRATION_FORMAT = "panorama_vignetting/rotations/v11 (the bundle's termination record, Core's ReturnCode and StopRule; the averaging on the pairs' Uncertain edges, the pairs with no covariance and the averaging's scale record; the averaging's and the bundle's optimization strategies, the bundle's scale record, the pairs' local_optimization_strategy; the clip mask and the canonical detection order of verified pairs v5; one bundle run at the pooled SIMFIT++ scale; the view-graph filters, their tolerances and removed pairs; LoMa's provider in the key and its provenance; the bundle's retcode, outcome, observations dropped and frames held at its start; pair samplers by stable_seed)"
registration_settings(C, inp, provider::Symbol) = stage_settings(C, inp, REGISTRATION_FORMAT, C.cfg["registration"], provider)

# The verified pairs' key: the [registration] settings of the features, the matching and the verification — every one but the
# rotation averaging's and the bundle's, so changing those reuses the pairs. With pairs = "lag_overlap" the averaging loss aims
# the pairs past the lag, so it keys the pairs too.
# v3: the detector's selection and detection image (`selection`, `detection_image` and its noise keys), every ASIFT detection
# refined (VisualGeometryToolkitFeatures 0.9), the kept keypoints the described set.
# v4: per pair its SIMFIT++ noise scale (sigma, success).
# v6: the pairs are scored on the keypoints' covariance shapes (keypoint_covariance_scale) and each records its absolute rotation
# covariance (VisualGeometryToolkitEstimation's uncertain_rotation): the averaging's Uncertain edges.
# v5: no keypoint within three scales of a saturated pixel (Features' detection mask: the frames' clip masks), the detections in
# Features' canonical order, the frames' map Features' raw_frame_map.
const PAIRS_FORMAT = "panorama_vignetting/verified_pairs/v6 (per pair its absolute rotation covariance (uncertain_rotation, the keypoints' covariance shapes in its residuals); the clip mask; Features' canonical detection order; per pair its SIMFIT++ scale; the detector's selection and detection image; LoMa's provider in the key and its provenance; features' pixels and covariances; per pair its rotation, counts and inlier matches)"
const AVERAGING_BUNDLE_KEYS = ("averaging_optimization_strategy", "cycle_tol_deg", "orientation_tol_deg", "bundle_optimization_strategy", "bundle_max_iter",
                               "bundle_ftol", "bundle_xtol", "bundle_whitening")
function pairs_settings(r)
    drop = r["pairs"] == "lag_overlap" ? filter(!=("averaging_optimization_strategy"), AVERAGING_BUNDLE_KEYS) : AVERAGING_BUNDLE_KEYS
    return Dict(k => v for (k, v) in r if k ∉ drop)
end
pairs_stage_settings(C, inp, provider::Symbol) = stage_settings(C, inp, PAIRS_FORMAT, pairs_settings(C.cfg["registration"]), provider)

# The verified pairs as a record and back (frame indices 1-based; a matrix row by row; a NaN — an unrefined keypoint's covariance
# — as null). The record keeps no descriptions: the pairs read back are `VerifiedPairs{FrameFeatures{Nothing}}`.
nan_null(x) = isnan(x) ? nothing : x
# LoMa's provenance (a `loma_provenance` NamedTuple per model, or a record's Dict read back) as a record: text values.
loma_record(loma) = Dict(string(k) => Dict(string(f) => string(v) for (f, v) in pairs(p)) for (k, p) in pairs(loma))
function pairs_record(vp::VerifiedPairs)
    feats = [Dict("pixels" => [collect(p) for p in f.pixels], "pixel_covariances" => [nan_null.(vec(Σ)) for Σ in f.pixel_covariances],
                  "n_detected" => f.n_detected) for f in vp.features]
    pairs = [Dict("i" => i, "j" => j, "rotation" => [collect(e.rotation[r, :]) for r in 1:3], "inliers" => e.inliers,
                  # the pair's absolute rotation covariance (rad², so(3) chart; uncertain_rotation), or null when not determined
                  "covariance" => e.covariance === nothing ? nothing : [collect(e.covariance[r, :]) for r in 1:3],
                  "inlier_matches" => [collect(m) for m in vp.inlier_matches[(i, j)]],
                  "matches" => vp.stats[(i, j)].matches, "rms_px" => vp.stats[(i, j)].rms_px, "rotation_deg" => vp.stats[(i, j)].rotation_deg,
                  "sigma_px" => nan_null(vp.scales[(i, j)].sigma), "sigma_success" => vp.scales[(i, j)].success)
             for ((i, j), e) in sort!(collect(vp.verified); by = first)]
    return Dict{String, Any}("features" => feats, "pairs" => pairs, "loma" => loma_record(vp.loma))
end
function pairs_from_record(rec)
    num(x) = x === nothing ? NaN : Float64(x)
    feats = [PanoramaStitching.FrameFeatures([SVector{2, Float64}(num.(p)...) for p in f["pixels"]],
                                             [SMatrix{2, 2, Float64, 4}(num.(Σ)...) for Σ in f["pixel_covariances"]], nothing, Int(f["n_detected"]))
             for f in rec["features"]]
    key(p) = (Int(p["i"]), Int(p["j"]))
    verified = Dict(key(p) => (rotation = SMatrix{3, 3, Float64, 9}(permutedims(reduce(hcat, [Float64.(r) for r in p["rotation"]]))),
                               inliers = Int(p["inliers"]),
                               covariance = p["covariance"] === nothing ? nothing :
                                            SMatrix{3, 3, Float64, 9}(permutedims(reduce(hcat, [Float64.(r) for r in p["covariance"]]))))
                    for p in rec["pairs"])
    matches = Dict(key(p) => [(Int(m[1]), Int(m[2])) for m in p["inlier_matches"]] for p in rec["pairs"])
    stats = Dict(key(p) => (matches = Int(p["matches"]), inliers = Int(p["inliers"]), rms_px = Float64(p["rms_px"]),
                            rotation_deg = Float64(p["rotation_deg"])) for p in rec["pairs"])
    loma = (; descriptor = rec["loma"]["descriptor"], matcher = rec["loma"]["matcher"])
    scales = Dict(key(p) => (sigma = num(p["sigma_px"]), success = Bool(p["sigma_success"])) for p in rec["pairs"])
    return VerifiedPairs(feats, verified, matches, stats, scales, loma)
end

# ASIFT's tilts, `[registration] asift_tilts`: a Features `tilt_schedule` name ("imas25", "untilted") or a list of [t, φ] pairs.
asift_tilts(t::AbstractString) = Symbol(t)
asift_tilts(t::AbstractVector) = [Tuple(Float64.(p)) for p in t]

# The detector's settings of [registration]: `selection` ("spread", Features' SSC), `detection_image` ("tone" or "vst") and, with
# "vst" only, the sensor's noise at the frames' ISO (`electrons_per_dn`, `read_noise_dn`) and the threshold `noise_sigmas`: a noise
# key with "tone" is refused, never read.
const DETECTION_NOISE_KEYS = ("electrons_per_dn", "read_noise_dn", "noise_sigmas")
function detector_settings(r)
    if r["detection_image"] == "vst"
        noise = (; electrons_per_dn = Float64(r["electrons_per_dn"]), read_noise = Float64(r["read_noise_dn"]), noise_sigmas = Float64(r["noise_sigmas"]))
    else
        inert = filter(k -> haskey(r, k), DETECTION_NOISE_KEYS)
        isempty(inert) || throw(ArgumentError("[registration] $(join(inert, ", ")): read only by detection_image = \"vst\", not \"$(r["detection_image"])\"; remove"))
        noise = nothing
    end
    return (; selection = Symbol(r["selection"]), detection_image = Symbol(r["detection_image"]), detection_noise = noise)
end

# The frames as VisualGeometryToolkitFeatures' LinearRaw (load_raw at the decode settings, the one RAW path): detection_image = "vst"
# reads their green cells' code spans, which the decode cache does not keep.
linear_raws(C, inp) = [VgtkFeatures.load_raw(joinpath(input_path(C.cfg["inputs"]["raw_dir"]), n); mode = Symbol(C.cfg["decode"]["mode"]),
                                             gain = Float64(C.cfg["decode"]["gain"])) for n in inp.names]

# The pairs' RANSAC local optimization, `[registration] local_optimization_strategy`: an Estimation strategy table
# (`{ type = "triggs", loss = "welsch", efficiency_constant = 2.5 }`, `{ type = "irls", loss = "tls" }`), read by Estimation's own
# loader (an unknown type, loss or key refused).
rotation_local_optimization(r) = VgtkEstimation.optimization_strategy_from_toml(r["local_optimization_strategy"]; class = :rotation,
                                                                                section = "registration.local_optimization_strategy")

# The verified pairs, cached (verified_pairs_<key>.jld2): the hour of ASIFT, LoMa and RANSAC of a set is spent once per matching
# setting, and written before the view graph is built, so a failure past it (the averaging, the bundle) loses none of it.
function verified_pairs(C, inp, lens, provider::Symbol; log)
    r = C.cfg["registration"]
    rec, _ = staged_record("verified_pairs", C, pairs_stage_settings(C, inp, provider); log, what = "verified pairs") do
        t0 = time()
        det = detector_settings(r)
        frames = det.detection_image === :vst ? linear_raws(C, inp) : inp.frames
        vp = verify_pairs(frames, lens; frame_map = inp.frame_map, tone_percentile = r["tone_percentile"], detector_margin_px = r["detector_margin_px"],
                          asift_tilts = asift_tilts(r["asift_tilts"]), saturated = det.detection_image === :vst ? nothing : inp.saturated,
                          max_detections = r["max_detections"], det..., descriptor = Symbol(r["descriptor"]), matcher = Symbol(r["matcher"]),
                          min_score = r["min_score"], pairs = Symbol(r["pairs"]), max_lag = r["max_lag"], overlap_deg = r["overlap_deg"],
                          min_inliers = r["min_inliers"], seed = r["seed"],
                          scoring = VgtkCore.MsacScoring(VgtkEstimation.RotationProblem; threshold = Float64(r["scoring_threshold_px"])),
                          local_optimization_strategy = rotation_local_optimization(r),
                          max_trials = r["max_trials"], min_trials = r["min_trials"], averaging_optimization_strategy = averaging_strategy(r),
                          execution_provider = provider, say = m -> say(log, m))
        merge!(pairs_record(vp), Dict{String, Any}("frames" => inp.names, "seconds" => time() - t0))
    end
    return (; pairs = pairs_from_record(rec), seconds = Float64(rec["seconds"]))
end

function registration(C, inp, lens; log)
    r = C.cfg["registration"]
    # "auto" → "cuda" or "cpu" (warning why) here, before the keys; "cuda" that cannot run is an error naming why
    provider = VgtkFeatures.resolve_execution_provider(Symbol(r["execution_provider"]))
    rec, file = staged_record("registration", C, registration_settings(C, inp, provider); log, what = "registration") do
        vp = verified_pairs(C, inp, lens, provider; log)
        t0 = time()
        reg = register_rotations(vp.pairs, lens; averaging_optimization_strategy = averaging_strategy(r),
                                 cycle_tol = deg2rad(r["cycle_tol_deg"]), orientation_tol = deg2rad(r["orientation_tol_deg"]),
                                 bundle_optimization_strategy = bundle_strategy(r),
                                 bundle_max_iter = r["bundle_max_iter"], bundle_ftol = r["bundle_ftol"], bundle_xtol = r["bundle_xtol"],
                                 say = m -> say(log, m))
        Rs = [SMatrix{3, 3, Float64, 9}(R) for R in reg.rotations]
        Dict{String, Any}("convention" => "d = R_k·f_k, the first frame's camera frame the common frame; R row by row",
                          "frames" => inp.names, "rotations" => [[collect(R[i, :]) for i in 1:3] for R in Rs],
                          # the seconds of the whole registration: the verified pairs' (as computed, also when reused) and the rest
                          # the verified pairs, and those the view-graph filters removed (0-based frame indices) at their tolerances
                          "pairs" => length(reg.pairs), "pairs_removed" => [[i - 1, j - 1] for (i, j) in reg.removed],
                          # the verified pairs whose rotation covariance is not determined (not averaged), 0-based, and the
                          # averaging's scale record (its edges' covariances absolute: scale 1, ScaleUpstream)
                          "pairs_no_covariance" => [[i - 1, j - 1] for (i, j) in reg.no_covariance],
                          "averaging_scale" => VgtkCore.noise_scale(reg.averaging),
                          "averaging_scale_source" => string(VgtkCore.scale_source(reg.averaging)),
                          "cycle_tol_deg" => Float64(r["cycle_tol_deg"]), "orientation_tol_deg" => Float64(r["orientation_tol_deg"]),
                          "bundle_rms_px" => reg.bundle.rms, "seconds" => vp.seconds + time() - t0,
                          # the bundle's termination record (Core's ReturnCode and StopRule, and the legacy retcode Symbol); the observations it
                          # dropped at its start (a start prediction off the camera model's valid domain); the frames it held at their
                          # start rotation (left with no kept observation: their 0-based indices)
                          "bundle_retcode" => string(VgtkCore.retcode_symbol(reg.bundle)), "bundle_return_code" => string(VgtkCore.retcode(reg.bundle)),
                          "bundle_stop_rule" => string(VgtkCore.stop_rule(reg.bundle)),
                          "bundle_dropped_start" => reg.bundle.metadata.n_dropped_start,
                          "bundle_frames_held" => [k - 1 for k in reg.bundle.metadata.frames_unconstrained],
                          # the bundle's robust loss width (camera px), its strategy and its scale (the residual core scale, the
                          # sensor point's σ of IntensityAndPositionNoise)
                          "bundle_loss_width_px" => reg.bundle.metadata.loss_width,
                          "bundle_optimization_strategy" => VgtkEstimation.config_dict(bundle_strategy(r)),
                          "bundle_scale_px" => VgtkCore.noise_scale(reg.bundle),
                          # the bundle's noise scale (the kept pairs' pooled SIMFIT++ σ̂, px) and the LM iterations its one run took
                          "bundle_sigma_px" => reg.bundle_sigma, "bundle_iterations_run" => reg.bundle.iterations,
                          # what ran LoMa (VisualGeometryToolkitFeatures' loma_provenance, carried by the verified pairs): the models,
                          # the ONNX Runtime provider used, its threads and, on CUDA, the driver, runtime and cuDNN versions
                          "loma" => loma_record(reg.loma))
    end
    # a record written before 2026-11-04 holds no bundle loss width: no point scale (IntensityAndPositionNoise then refuses to run;
    # delete the record to recompute it)
    return (; rotations = [SMatrix{3, 3, Float64, 9}(reduce(vcat, permutedims.(Vector{Float64}.(R)))) for R in rec["rotations"]],
            pairs = rec["pairs"], pairs_removed = rec["pairs_removed"], pairs_no_covariance = rec["pairs_no_covariance"],
            bundle_rms_px = rec["bundle_rms_px"],
            bundle_sigma_px = rec["bundle_sigma_px"], bundle_iterations_run = rec["bundle_iterations_run"], seconds = something(rec["seconds"], 0.0),
            bundle_retcode = rec["bundle_retcode"], bundle_return_code = rec["bundle_return_code"], bundle_stop_rule = rec["bundle_stop_rule"],
            bundle_dropped_start = rec["bundle_dropped_start"], bundle_frames_held = rec["bundle_frames_held"], provenance = get(rec, "provenance", nothing),
            loma = rec["loma"], point_scale_px = get(rec, "bundle_scale_px", nothing), record = file)
end

# The vertical of a roll-free sweep: every frame's image x axis is horizontal, so the vertical is the direction orthogonal to all
# of them (the eigenvector of the least eigenvalue of Σ x_k x_kᵀ), signed against the frames' mean image-down axis. It holds
# whatever the frames' pitch, unlike the axis of the frames' rotations (measured 2026-11-04: that axis was 9.3° off on the
# rooftop's pitched sweep and bent its horizon; 3.7° on the 14 mm set). The up frame puts it at −y and the first frame's
# forward on the horizon. Returns the up frame and the eigenvalues (the least one ≈ 0 when the sweep is roll-free).
function level_up(rotations)
    A = zeros(3, 3)
    for R in rotations
        x = SMatrix{3, 3, Float64, 9}(R)[:, 1]; A .+= x * x'
    end
    E = eigen(Symmetric(A)); a = SVector{3}(E.vectors[:, 1])
    down = sum(SMatrix{3, 3, Float64, 9}(R)[:, 2] for R in rotations)
    a = dot(a, down) > 0 ? -a : a
    z = SVector(0.0, 0.0, 1.0); zt = normalize(z - dot(z, a) * a); yt = -a
    return hcat(cross(yt, zt), yt, zt), E.values
end

# ----------------------------------------------------------------------------- vignetting spaces from the config

# The principal point: the pixel of the optical axis (0, 0, 1) through the camera's projection.
principal_point(lens) = (u = VgtkCore.pointwise!(VgtkCore.imaging_transform(VgtkCore.model(lens)), [SVector(0.0, 0.0)],
                                                 [SVector(0.0, 0.0, 1.0)]); u[1])

# The fit's domain (camera pixels) is the package's own `PanoramaStitching.frame_region`, which the fit, the sampling and the
# renders read. Its extreme points: its bounding box.
domain_box(lens, W, H, rim) = (b = VgtkCore.Rect2(PanoramaStitching.frame_region(lens, W, H, rim)); (SVector{2, Float64}(b.origin), SVector{2, Float64}(b.widths)))
# The centre of the camera's lens coverage, an image circle (or ellipse): `radial_center = "circle_center"`.
function coverage_center(lens)
    c = VgtkCore.lens_coverage(lens)
    c isa Union{VgtkCore.Circle, VgtkCore.Ellipse} ||
        error("vignetting.radial_center = \"circle_center\": the camera's lens coverage is a $(nameof(typeof(c))), not an image circle")
    return SVector{2, Float64}(c.center[1], c.center[2])
end
# The domain's farthest radius from `center`: an image circle's less the rim margin, else the farthest corner of its box.
function domain_radius(lens, center, W, H, rim)
    c = VgtkCore.lens_coverage(lens)
    c isa VgtkCore.Circle && return norm(SVector(c.center[1], c.center[2]) - center) + c.r - rim
    o, e = domain_box(lens, W, H, rim)
    return maximum(norm(SVector(x, y) - center) for x in (o[1], o[1] + e[1]), y in (o[2], o[2] + e[2]))
end

# V's model and start (VisualGeometryToolkitCore's LogVignetting: a spline of log V, log V ≡ 0 at the start) and its free directions:
# a radial profile's pole-regular subspace, every control point of a tensor surface. `kind` is a tag of Core's VIGNETTING_TAGS
# ("radial", "tensor_product"; the retired "tensor" is refused by name), the field log V is a spline of.
function vignetting_space(kind, C, lens, W, H)
    v = C.cfg["vignetting"]; rim = Float64(C.cfg["domain"]["rim_margin_px"])
    T = VgtkCore.VIGNETTING_TAGS[kind]
    if T === VgtkCore.RadialVignetting
        rc = v["radial_center"]
        center = rc == "principal_point" ? SVector{2, Float64}(principal_point(lens)) :
                 rc == "circle_center" ? coverage_center(lens) : SVector{2, Float64}(rc...)
        radius = v["radial_radius"] == "domain" ? domain_radius(lens, center, W, H, rim) + 1.0 : Float64(v["radial_radius"])
        s = VgtkCore.RadialVignetting(v["radial_controls"], center, radius; degree = 3)
        m = VgtkCore.LogVignetting(VgtkCore.with_control_points(s, zeros(VgtkCore.n_control_points(s))))
        return m, VgtkCore.pole_regular_basis(m)
    elseif T === VgtkCore.TensorProductVignetting
        v["tensor_box"] == "domain" || error("vignetting.tensor_box = $(v["tensor_box"]) is not \"domain\"")
        o, e = domain_box(lens, W, H, rim)
        s = VgtkCore.TensorProductVignetting(Tuple(v["tensor_controls"]), VgtkCore.Rect(o[1], o[2], e[1], e[2]); degree = 3)
        n = VgtkCore.n_control_points(s)
        m = VgtkCore.LogVignetting(VgtkCore.with_control_points(s, zeros(n)))
        return m, Matrix{Float64}(LinearAlgebra.I, n, n)
    end
    error("vignetting model \"$kind\" is not \"radial\" or \"tensor_product\" (log V is the spline of one)")
end

# The fit's measurement noise of the config: in the value only, or in the value and the sensor position with the sensor point's σ
# `point_scale_px` (the registration's).
function fit_measurement_noise(v, point_scale_px)
    N = PanoramaStitching.MEASUREMENT_NOISES[v["measurement_noise"]]      # "intensity" or "intensity_and_position" (an ArgumentError otherwise)
    N === IntensityNoise && return IntensityNoise()
    point_scale_px === nothing && error("vignetting.measurement_noise = \"intensity_and_position\" needs the registration's point " *
                                        "scale (its registration record holds no bundle_scale_px: delete it to recompute the registration)")
    return IntensityAndPositionNoise(; point_covariance = VgtkCore.CovMat{2}(σ = point_scale_px), log_intensity_scale = v["log_intensity_scale"],
                                     iterations = v["measurement_noise_iterations"])
end

# One fit of the config's settings; `vignetting` given (a calibration's V) holds V fixed: the frames' gains alone.
# `point_scale_px` the sensor point's σ of IntensityAndPositionNoise (the registration's `point_scale_px`).
function fit(kind, C, lens, dev, rotations, W, H; vignetting = nothing, point_scale_px = nothing)
    v = C.cfg["vignetting"]; d = C.cfg["domain"]
    m, Z = vignetting === nothing ? vignetting_space(kind, C, lens, W, H) :
           (vignetting, zeros(0, 0))                                      # any vignetting held fixed (a Core model or a callable)
    s = v["solver"]
    solver = s == "cg" ? ConjugateGradient(; tolerance = v["cg_tolerance"], max_iter = v["cg_max_iter"]) :
             s == "cholesky" ? SparseCholesky() : error("vignetting.solver = $s is not \"cg\" or \"cholesky\"")
    gc = PanoramaStitching.GAIN_CORRECTIONS[v["gain_correction"]]                        # "none", "intensity" or "channel" (an ArgumentError otherwise)
    return fit_vignetting(m, lens, dev.frames, rotations; free_directions = Z, gain_correction = gc === nothing ? nothing : gc(),
                          saturated = dev.saturated, frame_map = dev.frame_map,
                          rim_margin_px = d["rim_margin_px"], fov_deg = d["fov_deg"], plane = d["plane"], plane_maps = dev.plane_maps,
                          chart_spacing_deg = v["chart_spacing_deg"], penalty_order = v["penalty_order"], penalty_weight = v["penalty_weight"],
                          vignetting_penalty_weight = v["vignetting_penalty_weight"], solver,
                          measurement_noise = fit_measurement_noise(v, point_scale_px))
end
# The frames' exposure gains of a fit (K × 3), for the seam score and the renders: its metadata's when it fitted them, else none.
fit_gains(C, res) = C.cfg["vignetting"]["gain_correction"] == "none" ? nothing : res.metadata.gains
# A K × 3 gain matrix as text, one frame per row: g_k,R / g_k,G / g_k,B.
gains_text(G; d = 4) = join([join(round.(G[k, :]; digits = d), " / ") for k in axes(G, 1)], ", ")

# ----------------------------------------------------------------------------- views, renders, images

"No vignetting correction: V ≡ 1."
no_vignetting(u, f) = 1.0

"The covered (lon, lat) box in the up frame: a coarse equirectangular render's seen pixels."
function coverage_extent(lens, frames, rotations, up, frame_map; δ = 0.5)
    v = panorama_view(VgtkCore.Equirectangular(), (-180.0, 180.0, -89.0, 89.0), δ; up)
    r = render_panorama(lens, frames, rotations, v; vignetting = no_vignetting, frame_map,
                        rim_margin_px = 0.0, fov_deg = 89.0)
    seen = findall(>(0), Array(r.coverage))
    rows = extrema(I -> I[1], seen); cols = extrema(I -> I[2], seen)
    return (-180 + δ * (cols[1] - 1), -180 + δ * cols[2], 89 - δ * rows[2], 89 - δ * (rows[1] - 1))
end

"An H × W × 3 linear image as 8-bit sRGB at a fixed exposure, each plane times its white-balance factor (NaN, no frame: black): Features' `srgb_image`."
save_srgb(path, img, exposure, wb) = PNGFiles.save(path, VgtkFeatures.srgb_image(img; exposure, white_balance = wb))
"A seam-ratio map: red where the other frames are brighter than the shown one, blue darker, ±`lim` in log; grey unmeasured."
function save_ratio(path, M; lim = 0.3)
    col(v) = isnan(v) ? CT.RGB{FPN.N0f8}(0.5, 0.5, 0.5) : (x = clamp(v / lim, -1, 1);
              x >= 0 ? CT.RGB{FPN.N0f8}(1, 1 - x, 1 - x) : CT.RGB{FPN.N0f8}(1 + x, 1 + x, 1))
    PNGFiles.save(path, col.(M))
end

# ----------------------------------------------------------------------------- the record of V

function write_vignetting(C, stem, res, W, H)
    v = VgtkCore.estimate(res)
    kx = VgtkCore.knot_vectors(v)
    # a LogVignetting's control points are log V's (V = exp(s)); a V-form spline's (held fixed) are V's
    logform = v isa VgtkCore.LogVignetting; s = logform ? VgtkCore.log_spline(v) : v
    rec = Dict{String, Any}("gauge" => "mean log V = 0 over the fit domain's pixels", "control_points_of" => logform ? "log V" : "V",
                            "control_points" => collect(vec(Matrix(VgtkCore.control_points(v)))),
                            "knots" => collect.(kx), "degree" => collect(VgtkCore.spline_degree(v)),
                            "pixel_frame" => "camera pixels, COLMAP (full-resolution sensor)")
    if s isa VgtkCore.RadialVignetting
        c = VgtkCore.center(v); r = VgtkCore.field_radius(v)
        rec["model"] = Dict("kind" => VgtkCore.tag_of(VgtkCore.VIGNETTING_TAGS, VgtkCore.RadialVignetting), "center" => collect(c), "radius" => r, "controls" => VgtkCore.n_control_points(v),
                            "spline" => "VisualGeometryToolkitCore RadialVignetting: a B-spline in ρ/radius on clamped knots over [0, 1]")
        open(joinpath(C.run, stem * "_profile.csv"), "w") do io
            println(io, "rho_px,V")
            for ρ in 0.0:4.0:r
                println(io, @sprintf("%.1f,%.6f", ρ, v(c + SVector(ρ, 0.0))))
            end
        end
    else
        rec["model"] = Dict("kind" => VgtkCore.tag_of(VgtkCore.VIGNETTING_TAGS, VgtkCore.TensorProductVignetting), "controls" => collect(size(VgtkCore.control_points(v))),
                            "spline" => "VisualGeometryToolkitCore TensorProductVignetting: a tensor-product B-spline on clamped knots, x fastest")
    end
    open(io -> JSON.print(io, rec, 2), joinpath(C.run, stem * ".json"), "w")
    # the map over the sensor, every 32 px (x, y, V), and as an image normalized to its maximum
    open(joinpath(C.run, stem * "_map.csv"), "w") do io
        println(io, "x_px,y_px,V")
        for y in 16.0:32.0:H, x in 16.0:32.0:W
            println(io, @sprintf("%.1f,%.1f,%.6f", x, y, v(SVector(x, y))))
        end
    end
    M = [v(SVector(2j - 1.0, 2i - 1.0)) for i in 1:(H ÷ 2), j in 1:(W ÷ 2)]
    PNGFiles.save(joinpath(C.run, stem * "_map.png"), CT.Gray{FPN.N0f8}.(clamp.(M ./ maximum(M), 0, 1)))
    return M
end

# V's summary on the domain: its range, where its maximum is, and its largest antipodal asymmetry about `center`.
function vignetting_summary(v, lens, center, W, H, rim, fov_mask)
    pts = [SVector(2j - 1.0, 2i - 1.0) for i in 1:2:(H ÷ 2), j in 1:2:(W ÷ 2)]
    pts = filter(fov_mask, vec(pts))
    lv = [log(v(u)) for u in pts]
    imax = argmax(lv)
    ρ = maximum(norm(u - center) for u in pts)
    asym = 0.0
    for r in range(0, 0.95ρ, length = 40), φ in range(0, π, length = 73)[1:(end - 1)]
        a = center + r * SVector(cos(φ), sin(φ)); b = center - r * SVector(cos(φ), sin(φ))
        (fov_mask(a) && fov_mask(b)) || continue
        asym = max(asym, abs(v(a) / v(b) - 1))
    end
    return (; V_min = exp(minimum(lv)), V_max = exp(lv[imax]), max_at = pts[imax], max_offset_px = norm(pts[imax] - center),
            asymmetry = asym)
end

# ----------------------------------------------------------------------------- the run

function run_example(config_path)
    C = load_config(config_path)
    log = open(joinpath(C.run, "run.log"), "w")
    T = Dict{String, Float64}()
    say(log, "panorama_vignetting: $(C.cfg["name"]) ($(C.path)), output $(C.run)")
    t = time(); inp = decoded_frames(C; masks = true, log); T["decode (or cache)"] = time() - t
    K = length(inp.frames); Hf, Wf, _ = size(inp.frames[1]); W, H = 2Wf, 2Hf
    say(log, "frames: $K, $(Wf) × $(Hf) bin2 ($(W) × $(H) camera pixels), saturated quads $(sum(count, inp.saturated))")
    lens = load_lens(C; raw_width = W, raw_height = H)
    reg = registration(C, inp, lens; log)
    T["registration"] = reg.seconds
    uc = C.cfg["render"]["up"]
    uc in ("level", "first_frame") || error("render.up = $uc is not \"level\" or \"first_frame\"")
    up = uc == "level" ? level_up(reg.rotations)[1] : SMatrix{3, 3, Float64, 9}(LinearAlgebra.I)
    uc == "level" && say(log, "up frame: level (the frames' x axes span a plane to $(round(level_up(reg.rotations)[2][1]; sigdigits = 2)), least eigenvalue)")

    # the frames on the compute backend: the images (renders) and the masks (fit, seams) apart. The backend changes no result
    # (device and host agree to 1e-6 in log V, the package's tests); PANORAMA_VIGNETTING_DEVICE overrides the config's for one run
    # (a shared GPU without free memory).
    device = get(ENV, "PANORAMA_VIGNETTING_DEVICE", C.cfg["compute"]["device"])
    device in ("cuda", "cpu") || error("compute.device = $device is not \"cuda\" or \"cpu\"")
    say(log, "compute: $device$(haskey(ENV, "PANORAMA_VIGNETTING_DEVICE") ? " (PANORAMA_VIGNETTING_DEVICE)" : "")")
    move = device == "cuda" ? CuArray : identity
    dev = (; frames = move.(inp.frames), saturated = move.(inp.saturated), inp.frame_map, inp.plane_maps)   # the maps: isbits
    kind = C.cfg["vignetting"]["model"]
    fkw = (; point_scale_px = reg.point_scale_px)
    fit(kind, C, lens, dev, reg.rotations, W, H; fkw...)                          # compile
    t = time(); res = fit(kind, C, lens, dev, reg.rotations, W, H; fkw...); T["fit_vignetting ($kind, warm)"] = time() - t
    m = res.metadata
    say(log, "fit_vignetting ($kind): $(res.retcode), $(res.iterations) iterations, weighted rms $(round(res.rms; digits = 4)), " *
             "MAD $(round(m.residual_mad; digits = 4)), $(res.n) rows, $(m.n_pixels) pixels, chart $(m.chart.n)², $(m.n_active) active")
    say(log, "  measurement noise $(m.measurement_noise) (the model's log I moved by $(m.log_change) in the last solve); " *
             "stages (s): $(m.seconds)")
    C.cfg["vignetting"]["gain_correction"] == "none" ||
        say(log, "  gains ($(C.cfg["vignetting"]["gain_correction"]); g_k R / G / B, g₁ = 1): $(gains_text(m.gains))")
    M = write_vignetting(C, "vignetting", res, W, H)

    # the seam scores, before and after
    sc = C.cfg["seams"]; d = C.cfg["domain"]
    ext = sc["extent"] == "coverage" ? coverage_extent(lens, dev.frames, reg.rotations, up, dev.frame_map) : Tuple(sc["extent"])
    sview = panorama_view(VgtkCore.Equirectangular(), ext, sc["deg_per_pixel"]; up)
    skw = (; saturated = dev.saturated, frame_map = dev.frame_map, rim_margin_px = d["rim_margin_px"], fov_deg = d["fov_deg"],
           plane = d["plane"], plane_maps = dev.plane_maps, chunk = sc["chunk"])
    t = time()
    before = score_seams(lens, dev.frames, reg.rotations, sview; vignetting = no_vignetting, skw...)
    after = score_seams(lens, dev.frames, reg.rotations, sview; vignetting = VgtkCore.estimate(res), gains = fit_gains(C, res), skw...)
    T["score_seams (two)"] = time() - t
    save_ratio(joinpath(C.run, "seams_uncorrected.png"), Array(before.ratio)); save_ratio(joinpath(C.run, "seams_corrected.png"), Array(after.ratio))
    say(log, "seams (saturated pixels excluded): uncorrected rms $(round(before.rms; digits = 4)) MAD $(round(before.mad; digits = 4)); " *
             "corrected rms $(round(after.rms; digits = 4)) MAD $(round(after.mad; digits = 4)); $(after.n) pixels")

    # the renders: the real images (no saturation mask), corrected and uncorrected at one exposure
    rc = C.cfg["render"]
    rext = rc["extent"] == "coverage" ? ext : Tuple(rc["extent"])
    rkw = (; frame_map = dev.frame_map, rim_margin_px = d["rim_margin_px"], fov_deg = d["fov_deg"], plane_maps = dev.plane_maps)
    V = VgtkCore.estimate(res)
    # the corrected renders divide by the frames' gains as well; the uncorrected ones show the frames as they are
    render(view, vig) = Array(render_panorama(lens, dev.frames, reg.rotations, view; vignetting = vig,
                                              gains = vig === no_vignetting ? nothing : fit_gains(C, res), rkw...).image)
    t = time()
    ℓ = Float64(rc["cylindrical_max_lat_deg"])                 # tan φ grows without bound: the cylindrical's latitude cap
    cyl = panorama_view(VgtkCore.Cylindrical(), (rext[1], rext[2], max(rext[3], -ℓ), min(rext[4], ℓ)), rc["cylindrical_deg_per_pixel"]; up)
    cyl_c = render(cyl, V); cyl_u = render(cyl, no_vignetting)
    g = filter(!isnan, vec(view(cyl_c, :, :, 2)))
    exposure = 1 / quantile(g, rc["exposure_percentile"] / 100)
    # for viewing only: LibRaw's daylight white balance (normalized to G = 1), or none (camera raw colour)
    wb = rc["white_balance"] == "daylight" ? inp.daylight_wb ./ inp.daylight_wb[2] :
         rc["white_balance"] == "none" ? [1.0, 1.0, 1.0] : error("render.white_balance = $(rc["white_balance"]) is not \"daylight\" or \"none\"")
    save(name, img) = save_srgb(joinpath(C.run, name * ".png"), img, exposure, wb)
    save("cylindrical_corrected", cyl_c); save("cylindrical_uncorrected", cyl_u)
    cyl_c = cyl_u = nothing
    ea = panorama_view(VgtkCore.CylindricalEqualArea(), rext, rc["equal_area_deg_per_pixel"]; up)
    save("equal_area_corrected", render(ea, V)); save("equal_area_uncorrected", render(ea, no_vignetting))
    rw, rh = rc["rectilinear_size"]; f = rw / 2 / tand(rc["rectilinear_hfov_deg"] / 2)
    rcam = VgtkCore.CameraModel(f * VgtkCore.px, SA[rw / 2 * VgtkCore.px, rh / 2 * VgtkCore.px]; lens = VgtkCore.Lens(VgtkCore.Perspective()))
    rect = panorama_view(rcam, rw, rh; rotation = up)
    save("rectilinear_corrected", render(rect, V)); save("rectilinear_uncorrected", render(rect, no_vignetting))
    T["renders (six)"] = time() - t

    # the comparison fits (not used for the renders)
    rim = Float64(d["rim_margin_px"])
    domain = PanoramaStitching.frame_region(lens, W, H, rim)
    in_domain(u) = VgtkCore.Point2(u[1], u[2]) ∈ domain
    pp = SVector{2, Float64}(principal_point(lens))
    rows = Any[(kind, res, after, vignetting_summary(V, lens, pp, W, H, rim, in_domain))]
    for k in C.cfg["vignetting"]["compare_models"]
        t = time(); r2 = fit(k, C, lens, dev, reg.rotations, W, H; fkw...); T["fit_vignetting ($k, compare)"] = time() - t
        s2 = score_seams(lens, dev.frames, reg.rotations, sview; vignetting = VgtkCore.estimate(r2), gains = fit_gains(C, r2), skw...)
        write_vignetting(C, "vignetting_compare_$k", r2, W, H)
        push!(rows, (k, r2, s2, vignetting_summary(VgtkCore.estimate(r2), lens, pp, W, H, rim, in_domain)))
        say(log, "compare $k: $(r2.retcode), weighted rms $(round(r2.rms; digits = 4)), seams rms $(round(s2.rms; digits = 4)) MAD $(round(s2.mad; digits = 4))")
    end
    write_summary(C, inp, reg, rows, before, ext, exposure, T, W, H, pp)
    close(log)
    return (; C, inp, lens, reg, res, before, after, rows, T)
end

function write_summary(C, inp, reg, rows, before, ext, exposure, T, W, H, pp)
    f(x; d = 4) = string(round(x; digits = d))
    open(joinpath(C.run, "summary.md"), "w") do io
        println(io, "# $(C.cfg["name"]): panorama vignetting\n")
        println(io, "Config of record: `$(basename(C.path))`. Data root `\$VGTK_DATA_ROOT`; RAW `$(C.cfg["inputs"]["raw_dir"])`, lens `$(C.cfg["inputs"]["lens"])`.\n")
        println(io, "- Frames: $(length(inp.frames)), decoded $(C.cfg["decode"]["mode"]) gain $(C.cfg["decode"]["gain"]) → $(W ÷ 2) × $(H ÷ 2); saturated quads (all frames): $(sum(count, inp.saturated)).")
        bund = "$(f(reg.bundle_rms_px; d = 3)) px (one LM run of $(reg.bundle_iterations_run) iterations, the strategy " *
               "$(C.cfg["registration"]["bundle_optimization_strategy"]) at σ̂ $(f(reg.bundle_sigma_px; d = 3)) px, the pairs' pooled SIMFIT++ scale)"
        println(io, "- Registration: $(reg.pairs) verified pairs ($(length(reg.pairs_no_covariance)) with no determined rotation " *
                    "covariance, not averaged$(isempty(reg.pairs_no_covariance) ? "" : ": " * join(["$(p[1])-$(p[2])" for p in reg.pairs_no_covariance], ", "))" *
                    "; $(length(reg.pairs_removed)) removed by the view-graph filters" *
                    "$(isempty(reg.pairs_removed) ? "" : ": " * join(["$(p[1])-$(p[2])" for p in reg.pairs_removed], ", "))), bundle rms $bund$(reg.provenance === nothing ? "" : "; $(reg.provenance)"); " *
                    "bundle termination $(reg.bundle_return_code) ($(reg.bundle_stop_rule); retcode $(reg.bundle_retcode)), $(reg.bundle_dropped_start) observations dropped at its start: not imaged there; " *
                    "$(isempty(reg.bundle_frames_held) ? "no frame" : "frames $(join(reg.bundle_frames_held, ", "))") held at the start " *
                    "rotation: no observation kept).")
        lm = reg.loma["descriptor"]
        println(io, "- LoMa ($(lm["model"]), $(reg.loma["matcher"]["model"])): ONNX Runtime $(lm["onnxruntime"]) on $(lm["execution_provider"])" *
                    "$(lm["execution_provider"] == "cuda" ? " (CUDA runtime $(lm["cuda_runtime"]), cuDNN $(lm["cudnn"]), driver $(lm["cuda_driver"]))" : ""), " *
                    "$(lm["threads"]) threads (registration.execution_provider = \"$(C.cfg["registration"]["execution_provider"])\").")
        println(io, "- Principal point (the ray (0, 0, 1)): ($(f(pp[1]; d = 1)), $(f(pp[2]; d = 1))) px.")
        println(io, "- Compute backend: $(get(ENV, "PANORAMA_VIGNETTING_DEVICE", C.cfg["compute"]["device"])).")
        println(io, "- Seam raster: equirectangular, $(C.cfg["seams"]["deg_per_pixel"])°/px over lon $(f(ext[1]; d = 1))…$(f(ext[2]; d = 1))°, lat $(f(ext[3]; d = 1))…$(f(ext[4]; d = 1))° (up frame).")
        println(io, "- Render exposure (all six PNGs): $(f(exposure)) × linear, white balance $(C.cfg["render"]["white_balance"]) (viewing only), then sRGB. " *
                    "The renders show the real images: saturated pixels keep their clipped values.\n")
        println(io, "## Seam spread (log ratio to the shown frame; saturated pixels EXCLUDED)\n")
        println(io, "| V divided out | rms | 1.4826·MAD | pixels |\n|---|---|---|---|")
        println(io, "| none | $(f(before.rms)) | $(f(before.mad)) | $(before.n) |")
        for (k, r, s, _) in rows
            println(io, "| $k$(k == C.cfg["vignetting"]["model"] ? " (of record)" : "") | $(f(s.rms)) | $(f(s.mad)) | $(s.n) |")
        end
        println(io, "\n## The fits\n")
        println(io, "| model | retcode | iterations | weighted rms (log I) | rms | 1.4826·MAD | rows | V min / max on the domain | max at (px) | max − principal point (px) | largest antipodal asymmetry |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|---|")
        for (k, r, s, v) in rows
            m = r.metadata
            println(io, "| $k | $(r.retcode) | $(r.iterations) | $(f(r.rms)) | $(f(m.residual_rms)) | $(f(m.residual_mad)) | $(r.n) | " *
                        "$(f(v.V_min; d = 3)) / $(f(v.V_max; d = 3)) | ($(f(v.max_at[1]; d = 0)), $(f(v.max_at[2]; d = 0))) | " *
                        "$(f(v.max_offset_px; d = 0)) | $(f(100v.asymmetry; d = 1)) % |")
        end
        println(io, "\nFrame gains (vignetting.gain_correction = $(C.cfg["vignetting"]["gain_correction"])): " *
                    (C.cfg["vignetting"]["gain_correction"] != "none" ? "g_k R / G / B of the fit of record (g₁ = 1) $(gains_text(rows[1][2].metadata.gains)); the seam score and the corrected renders divide them out." :
                     "none fitted, the frames taken at one exposure."))
        println(io, "\nStage times of the fit of record (s): $(rows[1][2].metadata.seconds)\n")
        println(io, "## Wall times (s)\n")
        for (k, v) in sort(collect(T); by = first)
            println(io, "- $k: $(f(v; d = 2))")
        end
        println(io, "\n## Files\n")
        for x in sort(readdir(C.run))
            isfile(joinpath(C.run, x)) && println(io, "- `$x`")
        end
    end
    # the run's records (VisualGeometryToolkitEval): its config of record, its key, its driver and the inputs it read
    raws = [input_path(joinpath(C.cfg["inputs"]["raw_dir"], n)) for n in inp.names]
    VgtkEval.Experiments.write_run_records(C.run, C.cfg; driver = "examples/panorama_vignetting/run.jl", config_source = C.path,
                                           key = VgtkEval.Experiments.run_key(C.cfg), inputs = [raws; input_path(C.cfg["inputs"]["lens"])])
    return C.run
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: julia --project=<env> run.jl <config.toml>")
    run_example(ARGS[1])
end
