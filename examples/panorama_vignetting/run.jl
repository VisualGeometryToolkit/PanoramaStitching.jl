# The panorama vignetting example: RAW frames of a camera that only rotated → their rotations → the camera's vignetting from
# the frames alone (interpolation free) → the seam scores before and after → corrected and uncorrected panoramas.
#
#     julia --project=<env> examples/panorama_vignetting/run.jl examples/panorama_vignetting/config_fx3_14mm.toml
#
# Every setting is in the config (a config of record: nothing here has a default). Inputs are read under the data root
# (VisualGeometryToolkitEval's data_root, $VGTK_DATA_ROOT); everything this writes goes under the results root
# (results_root, $VGTK_RESULTS_ROOT): <results>/panorama_vignetting/<name>/, each step a keyed product of Eval's (`Product`,
# produced or loaded by `produce`, DrWatson's produce_or_load): verified_pairs/<key>/, registration/<key>/ and the run's outputs
# in run/<key>/, each with its records. The stages are functions, so in a REPL
# `include("run.jl"); run_example("config.toml")` runs it all (a produced run is reused: `force = true`, or `--force`, produces it
# again) and each stage can be rerun alone. See README.md.

using LibRaw_jll                                   # Features' LibRaw decoder (its package extension)
using CUDA                                         # the [compute] device's kernels
# LoMa's ONNX Runtime CUDA provider (registration.execution_provider = "auto" or "cuda") takes its CUDA 12 runtime and cuDNN
# from this environment's CUDA_Runtime_jll and CUDNN_jll (VisualGeometryToolkitFeatures loads them for ONNX Runtime alone,
# whatever runtime CUDA.jl uses); "auto" runs on the CPU, warning why, where they or a device are missing.
using PanoramaStitching
import VisualGeometryToolkitCore as VgtkCore
import VisualRadiometryToolkitCore as VrtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
import VisualGeometryToolkitFeatures as VgtkFeatures
import VisualGeometryToolkitIO as VgtkIO
import VisualGeometryToolkitEval as VgtkEval
using StaticArrays: SVector, SMatrix, SA
using LinearAlgebra: LinearAlgebra, norm, normalize, dot, cross, eigen, Symmetric
using Statistics: mean, median, quantile
using Printf: @sprintf                             # compare_arms.jl's tables (it includes this file)
import PNGFiles, JSON, TOML                        # JSON: the records of compare_arms.jl and autocalibration_check.jl (they include this file)

const FPN = PNGFiles.FixedPointNumbers
const CT = PNGFiles.ColorTypes

# ----------------------------------------------------------------------------- configuration

# Every input path is relative to the data root: VisualGeometryToolkitEval.Experiments' `data_root` ($VGTK_DATA_ROOT, README "Inputs").
input_path(rel) = joinpath(VgtkEval.Experiments.data_root(), rel)

# The example's configuration versions (VisualGeometryToolkitCore's ConfigVersions; decision 11: a table is named after the
# algorithm or problem it configures). A config states its `config_version`; one without it is version 1 (PanoramaStitching 0.11
# and earlier: every registration setting in [registration], the fit's in [vignetting]), read through the step below, so the code
# reads the current tables only and a stored config keeps loading.
const EXAMPLE_CONFIG_VERSIONS = VgtkCore.ConfigVersions("panorama_vignetting",
    # version 1 → 2 (PanoramaStitching 0.12): the pairs' RANSAC, the lag pairs, the averaging, the bundle and the fit in their own
    # tables; the bundle (check_tracks) and the fit's solver by tag
    VgtkCore.ConfigMigration(; tables = ["name", "inputs", "compute", "decode", "registration", "domain", "vignetting", "seams", "render"],
        renames = ["registration.outlier_halfwidth_px" => "rotation_problem.ransac.outlier_halfwidth_px",
                   "registration.local_optimization_strategy" => "rotation_problem.ransac.local_optimization_strategy",
                   "registration.max_trials" => "ransac.max_trials", "registration.min_trials" => "ransac.min_trials",
                   "registration.max_lag" => "lag_overlap.max_lag",
                   "registration.averaging_optimization_strategy" => "rotation_averaging.optimization_strategy",
                   "registration.bundle_optimization_strategy" => "rotation_bundle_problem.levenberg_marquardt.optimization_strategy",
                   "registration.bundle_max_iter" => "rotation_bundle_problem.levenberg_marquardt.stops.max_iter",
                   "registration.bundle_ftol" => "rotation_bundle_problem.levenberg_marquardt.stops.ftol",
                   "registration.bundle_xtol" => "rotation_bundle_problem.levenberg_marquardt.stops.xtol",
                   "vignetting.chart_spacing_deg" => "vignetting_self_calibration.chart_spacing_deg",
                   "vignetting.penalty_order" => "vignetting_self_calibration.penalty_order",
                   "vignetting.penalty_weight" => "vignetting_self_calibration.penalty_weight",
                   "vignetting.vignetting_penalty_weight" => "vignetting_self_calibration.vignetting_penalty_weight",
                   "vignetting.gain_correction" => "vignetting_self_calibration.gain_correction",
                   "vignetting.observation_noise" => "vignetting_self_calibration.observation_noise",
                   "vignetting.observation_noise_iterations" => "vignetting_self_calibration.observation_noise_iterations",
                   "vignetting.cg_tolerance" => "vignetting_self_calibration.stops.rtol",
                   "vignetting.cg_max_iter" => "vignetting_self_calibration.stops.max_iter"],
        values = ["registration.check_tracks" => t -> ("registration.bundle" => t ? "track_check" : "levenberg_marquardt"),
                  "vignetting.solver" => s -> ("vignetting_self_calibration.solver" => get(Dict("cg" => "normal_cg", "cholesky" => "sparse_cholesky"), s, s))]);
    tables = ["name", "inputs", "compute", "decode", "registration", "domain", "vignetting", "seams", "render", "rotation_problem", "ransac",
              "lag_overlap", "rotation_averaging", "rotation_bundle_problem", "vignetting_self_calibration"])

# The algorithms a config names by tag (the tag rule: VisualGeometryToolkitCore's TagRegistry): the pairs verified, the bundle's
# refine (a TrackCheck of the LevenbergMarquardt, or that LevenbergMarquardt alone) and the fit's linear solver.
const PAIR_SELECTIONS = VgtkCore.TagRegistry(:pair_selection, "Pairs", "all" => AllPairs, "lag_overlap" => LagOverlap;
                                             paths = [r"registration\.pairs"])
const BUNDLE_REFINES = VgtkCore.TagRegistry(:bundle_refine, "", "track_check" => TrackCheck, "levenberg_marquardt" => VgtkCore.LevenbergMarquardt;
                                            paths = [r"registration\.bundle"])
const LINEAR_SOLVERS = VgtkCore.TagRegistry(:linear_solver, "", "normal_cg" => VgtkCore.NormalCG, "sparse_cholesky" => SparseCholesky;
                                            paths = [r"vignetting_self_calibration\.solver"])

# The vignetting keys of earlier releases (PanoramaStitching 0.4 renamed the weights, 0.7 the noise: an observation's, the
# terminology's word): a table that holds one is refused by name, never read as the new setting. The 0.4–0.6 key is spelled in
# fragments (check_naming retires its word from every identifier).
const RENAMED_KEYS = ("weights" => ("observation_noise", "0.4"), "sampson_iterations" => ("observation_noise_iterations", "0.4"),
                      "measure" * "ment_noise" => ("observation_noise", "0.7"),
                      "measure" * "ment_noise_iterations" => ("observation_noise_iterations", "0.7"))
# The vignetting key retired with no successor key (PanoramaStitching 0.7; spelled in fragments: check_naming retires the name):
# the position noise's constant σ of log I is a specified RelativeNoise of the frames' sensor model.
const RETIRED_VIGNETTING_KEYS = ("log_intensity" * "_scale" => "the value's noise is the frames' sensor model (Core's RelativeNoise, " *
                                 "specified: σ_rel = the old σ of log I; arms.toml relative_noise)",)
function refuse_renamed_keys(table, where)
    for (old, why) in RETIRED_VIGNETTING_KEYS
        haskey(table, old) && throw(ArgumentError("$where: the key $old is retired: $why (PanoramaStitching 0.7)"))
    end
    for (old, (new, version)) in RENAMED_KEYS
        haskey(table, old) && throw(ArgumentError("$where: the key $old was renamed to $new (PanoramaStitching $version: " *
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

# The table only `pairs = "lag_overlap"` reads ([lag_overlap]: its max_lag), refused with "all" when the config is read (no inert
# setting), so a config fails the same way whether or not its stages are cached.
function refuse_inert_pair_keys(cfg, where)
    cfg["registration"]["pairs"] == "all" && haskey(cfg, "lag_overlap") &&
        throw(ArgumentError("$where: [lag_overlap] is read only by pairs = \"lag_overlap\"; remove it with pairs = \"all\""))
    return cfg
end

# The output folder of earlier releases (`output`, relative to the config): the results go under Eval's results root, keyed by
# the config — a config that names one is refused, never read.
function refuse_output(cfg, where)
    haskey(cfg, "output") && throw(ArgumentError("$where: the key output is retired: every result goes under VisualGeometryToolkitEval's " *
                                                 "results_root() (\$VGTK_RESULTS_ROOT), <results>/panorama_vignetting/<name>/, each step " *
                                                 "a keyed product (Eval's Product); remove the key"))
    return cfg
end

# The frames' sensor records are stated (no default): `[decode] sensor_characterization` (the frames') and `[registration]
# sensor_characterization` (the detector's), each a vgio/sensor_characterization file under the data root or "none" — a config
# without either is refused when it is read, before any file is.
function refuse_undeclared_sensor_record(table, where)
    haskey(table, "sensor_characterization") ||
        throw(ArgumentError("$where sensor_characterization: required — the body's characterisation (a vgio/sensor_characterization file under the data root) the frames are decoded under, or \"none\" (the file's black, no defects); [registration]'s, with detection_image = \"tone\", is [decode]'s"))
    return table
end

# Every refusal of a config of record when it is read (`load_config`), before any file is: the keys of earlier releases, the
# inert pair keys, the undeclared sensor records; and with detection_image = "tone" the detector reads the decoded frames, so
# its record is [decode]'s (with "vst" it decodes the frames again, under its own).
function refuse_config(cfg, where)
    refuse_output(cfg, where)
    refuse_renamed_keys(cfg["vignetting"], "$where [vignetting]")
    refuse_renamed_keys(get(cfg, "vignetting_self_calibration", Dict{String, Any}()), "$where [vignetting_self_calibration]")
    refuse_circle(cfg["inputs"], "$where [inputs]")
    # the retired [registration] keys: PanoramaStitching's one list, the keywords' (refused by name, never read)
    PanoramaStitching.refuse_retired_registration(cfg["registration"], "$where [registration]")
    refuse_inert_pair_keys(cfg, where)
    dec = refuse_undeclared_sensor_record(cfg["decode"], "$where [decode]")
    reg = refuse_undeclared_sensor_record(cfg["registration"], "$where [registration]")
    reg["detection_image"] == "tone" && reg["sensor_characterization"] != dec["sensor_characterization"] &&
        throw(ArgumentError("$where [registration] sensor_characterization = $(repr(reg["sensor_characterization"])): with detection_image = \"tone\" " *
                            "the detector reads the decoded frames, decoded under [decode] sensor_characterization = $(repr(dec["sensor_characterization"])); state that record"))
    detector_settings(reg)                                            # the detector's keys: refused here, not after the decode
    return cfg
end

# The config of record and where its results go: `experiment` the set's products under the results root
# (`panorama_vignetting/<name>`: Eval's `Product`s, each in <results>/<experiment>/<product>/<key>/), `out` that folder (the decode
# cache), `code` the code trees every product is keyed by (Eval's `current_code_trees`: PanoramaStitching's and every stack
# package's this environment loads; computed once: a `Product` would compute them again each time).
function load_config(path)
    # the keys of earlier releases refused by name first (a version-1 file holds them where that version did), then the file read
    # through the example's converter at the current version, and refused again there
    raw = TOML.parsefile(path)
    refuse_renamed_keys(get(raw, "vignetting", Dict{String, Any}()), "$path [vignetting]")
    cfg = refuse_config(VgtkCore.migrate_config(raw, EXAMPLE_CONFIG_VERSIONS; source = path), path)
    experiment = "panorama_vignetting/" * cfg["name"]
    out = joinpath(VgtkEval.Experiments.results_root(), experiment)
    return (; cfg, out, experiment, code = VgtkEval.Experiments.current_code_trees(), path = abspath(path))
end

say(io, msg) = (println(msg); println(io, msg); flush(io))

# ----------------------------------------------------------------------------- the decode cache
#
# Each RAW frame is decoded once per content and decode settings by Features' `load_raw` into its one persistent cache (a
# FileCache, <output>/decoded/linear_raws.jld2; the entry is keyed by the RAW file's SHA-256, the mode, the gain and the
# decoder), and read back as decoded, value for value. The frames' and the planes' maps onto the camera's pixels are
# Core's (`raw_frame_map`, `bayer_plane_map`) of the set's first frame (all the set's frames share the Bayer phase).

# The decoded-RAW cache is OPT-IN (`[decode] cache = true`): a derived store (Features' FileCache, keyed by the RAW file's SHA-256, the noise digest,
# the mode, the gain and the decoder; regenerable from the RAW files), backed by JLD2 through Features' package extension. This
# file loads no JLD2 itself: VisualGeometryToolkitEval (imported above) depends on it, so it is loaded with Eval and Features'
# extension with it — a caller that includes this file needs no JLD2 of its own.
"""
    decoded_frames(C; masks, log = devnull) -> (; names, frames, saturated, frame_map, plane_maps, daylight_wb, models, sensor_characterization_sha256)

The frames as linear images (H × W × 3 Float32), their saturation masks with `masks = true`, their maps and the daylight white
balance; each RAW decoded once (cached), under its frame's sensor model: `[decode] sensor_characterization` (required, no default) is
the body's characterisation (a vgio/sensor_characterization file under the data root) or `"none"`. Each frame's model is
VisualGeometryToolkitCore's resolver `sensor_model` (a lookup) at the frame's capture settings (Features' `extract_capture_settings`,
the RAW files' EXIF): Features' `load_raw` subtracts its black (a dark per exposure, the measured channel black, or — `"none"` — the
file's) and joins its defects to `saturated`; the frames are the values (an uncertain image's `value_of`). `models` are the frames'
models; `sensor_characterization_sha256` the record's file's SHA-256 (`nothing` for `"none"`), which the fit's and the seams' records
state, so a consumer can check that the frames were decoded under the record its own calibration used.
"""
function decoded_frames(C; masks::Bool, log = devnull)
    dec = C.cfg["decode"]; mode = dec["mode"]; gain = Float64(dec["gain"])
    mode == "bin2" || error("decode.mode = $mode: the example runs on bin2's three planes")
    char = String(dec["sensor_characterization"])
    dir = joinpath(C.out, "decoded"); mkpath(dir)
    raw_dir = input_path(C.cfg["inputs"]["raw_dir"])
    raws = sort([f for f in readdir(raw_dir) if endswith(f, C.cfg["inputs"]["raw_extension"])])
    models = sensor_models(char, length(raws)) do
        Dict(k => VgtkFeatures.load_raw_metadata(joinpath(raw_dir, f)) for (k, f) in enumerate(raws))
    end
    sha = sensor_record_sha256(char)
    decode(k, cache) = (f = raws[k];
                        lin = VgtkFeatures.load_raw(joinpath(raw_dir, f); mode = Symbol(mode), gain, sensor_model = models[k], decoder = VgtkFeatures.LibRawDecoder(), cache);
                        say(log, "  $f ($(count(lin.saturated)) saturated or defect quads; $(VgtkCore.component_sources(models[k]).reason))"); lin)
    lins = if dec["cache"]
        cache = VgtkFeatures.open_cache(joinpath(dir, "linear_raws.jld2"))
        try
            map(k -> decode(k, cache), eachindex(raws))
        finally
            VgtkFeatures.close_cache(cache)
        end
    else
        map(k -> decode(k, nothing), eachindex(raws))
    end
    lin1 = first(lins)
    return (; names = raws, frames = [Float32.(VgtkCore.value_of.(l.image)) for l in lins], saturated = masks ? [l.saturated for l in lins] : BitMatrix[],
            frame_map = VgtkCore.raw_frame_map(lin1), plane_maps = ntuple(c -> VgtkCore.bayer_plane_map(lin1, c), 3),
            daylight_wb = collect(Float64.(lin1.metadata.daylight_wb)), models, sensor_characterization_sha256 = sha)
end

# The frames' sensor models: per frame VisualGeometryToolkitCore's `sensor_model` (the ONE resolver, a lookup) of the body's
# characterisation `char` (IO's `load` of the vgio/sensor_characterization file under the data root, or "none": the absent model)
# at the frame's capture settings (Features' `extract_capture_settings` of `sources()`, frame index => its RAW metadata or mosaic).
# "none" reads no capture (the absent model needs none, and a set whose EXIF Features cannot read decodes as before).
function sensor_models(sources, char::AbstractString, n::Integer)
    char == "none" && return fill(VgtkCore.SensorModel(), n)
    record = VgtkIO.load(VgtkCore.SensorCharacterization, input_path(char))
    capture = VgtkFeatures.extract_capture_settings(sources())
    return [VgtkCore.sensor_model(record, capture, k) for k in 1:n]
end
# A sensor record's content hash (Eval's `file_sha256`, cached in the process), or `nothing` for "none".
sensor_record_sha256(char::AbstractString) = char == "none" ? nothing : VgtkEval.Experiments.file_sha256(input_path(char))

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
        VgtkCore.with_model(grid, VgtkCore.solve(VgtkCore.EstimationProblem(VgtkEstimation.CameraRegressionProblem(VgtkCore.SplineLens, grid, VgtkCore.imaged_region(grid))),
                                                 VgtkEstimation.CameraRegression(; tolerance = Float64(inp["spline_tolerance_px"]))).u)
    else
        error("inputs.lens_kind = $(inp["lens_kind"]) is not \"cameras\" or \"grid\"")
    end
    VgtkCore.extent(camera) == VgtkCore.Size2(raw_width, raw_height) ||
        error("inputs.lens: the camera's sensor is $(VgtkCore.extent(camera)), the frames' $(raw_width) × $(raw_height) camera pixels")
    return camera
end

# ----------------------------------------------------------------------------- registration (cached by its settings)

# The averaging's and the bundle's strategies: their [registration] tables, read by Estimation's one strategy reader.
averaging_strategy(cfg) = VgtkEstimation.optimization_strategy_from_toml(cfg["rotation_averaging"]["optimization_strategy"];
                                                                        class = :rotation_averaging, section = "rotation_averaging.optimization_strategy")
bundle_strategy(cfg) = VgtkEstimation.optimization_strategy_from_toml(cfg["rotation_bundle_problem"]["levenberg_marquardt"]["optimization_strategy"];
                                                                     class = :rotation_bundle_problem,
                                                                     section = "rotation_bundle_problem.levenberg_marquardt.optimization_strategy")

# The stages' records (the verified pairs, the registration): each an Eval `Product` of the set (`stage_product`), its record
# `<name>.jld2` in its keyed directory, reused while its key — a SHA-256 of the record's format, the stage's settings, the ONNX
# Runtime provider LoMa runs on (`execution_provider = "auto"` resolved: a CPU record is not a CUDA run's, whose matches differ
# in the pairs at the network's threshold), the inputs, the decode settings, the RAW files' hashes, the code trees and the
# products it reads (their outputs' content hashes: the registration is the verified pairs' consumer), and the files the frames
# and the camera are read from by content (the decode's sensor record, the lens) — matches.
# `produce_stage` is their one writer, `stage_record` their one reader.
function stage_settings(C, inp, format, settings, provider::Symbol)
    return Dict{String, Any}("format" => format, "settings" => settings, "provider" => string(provider),
                             "inputs" => C.cfg["inputs"], "decode" => C.cfg["decode"],
                             "raw_sha256" => [VgtkEval.Experiments.file_sha256(raw) for raw in raw_paths(C, inp)],
                             "decode_sensor_characterization_sha256" => something(sensor_record_sha256(C.cfg["decode"]["sensor_characterization"]), "none"),
                             "lens_sha256" => VgtkEval.Experiments.file_sha256(input_path(C.cfg["inputs"]["lens"])))
end
raw_paths(C, inp) = [joinpath(input_path(C.cfg["inputs"]["raw_dir"]), n) for n in inp.names]
stage_product(C, name, settings; upstream = nothing) =
    VgtkEval.Experiments.Product(C.experiment, name, settings; output = name * ".jld2", upstream, code = C.code)
const DRIVER = "examples/panorama_vignetting/run.jl"
# A stage's product produced: Eval's `produce`, `compute()` (the record, a String-keyed Dict) run when it is not produced;
# `stage_record` reads the record back (Eval's `load_output`).
function produce_stage(compute, p, C; log, what, inputs = String[], force = false)
    ran = false
    VgtkEval.Experiments.produce(_ -> (ran = true; compute()), p; driver = DRIVER, config_source = C.path, inputs, force)
    ran || say(log, "$what: reused $(VgtkEval.Experiments.product_dir(p))")
    return p
end
stage_record(p) = VgtkEval.Experiments.load_output(Dict{String, Any}, p)

# The registration's key: every [registration] setting and LoMa's provider. v3: PanoramaStitching 0.2.2 seeds each pair's sampler
# by stable_seed(seed, i, j), so a v2 record's rotations are not this code's; they are recomputed. v5: PanoramaStitching 0.6's
# bundle drops the observations not imaged at its start and counts them (`n_dropped_start`, replacing v3's start finiteness
# count), and holds the frames left with none (`frames_unconstrained`). v6: the key holds the provider LoMa runs on, and the
# record LoMa's provenance (`loma`). v7: the view-graph filters (Estimation's FilteredRotationAveraging at `cycle_tol_deg` and
# `orientation_tol_deg`) before the tracks; the record holds the tolerances and the pairs they removed. v8: the bundle is one LM
# run at a multiple of the pairs' pooled SIMFIT++ scale (bundle_iterations retired); the record holds σ̂ and its iterations.
# v9: the verified pairs' v5 (the clip mask, Features' canonical detection order).
# v10: the robust-estimation contract: the averaging's and the bundle's strategies are tables (averaging_optimization_strategy,
# bundle_optimization_strategy, read by Estimation's reader), the record holds the bundle's strategy and its scale
# (bundle_scale_px), the pairs' strategy key is local_optimization_strategy.
# v11: the bundle's termination record, Core's ReturnCode and StopRule (bundle_return_code, bundle_stop_rule) in place of
# PanoramaStitching's outcome Symbol (bundle_outcome retired).
# v17: the near-duplicate pairs by VisualGeometryToolkitCore's view_overlap (`near_duplicate_overlap`, of the smaller footprint;
# PanoramaStitching 0.8), not an optical-axis angle.
# v16: the near-duplicate pairs left out of the chaining (`near_duplicate_deg`), the track check (`check_tracks`: per tested
# solve its removals past 3.035 σ̂, rms and inlier rms), the bundle's inlier rms (residuals within its loss width).
# v15: `bundle_fit`, the bundle's termination record as Core's values (the results file's `fit`); `pairs_removed` the pairs the view-graph filters removed only (a pair with no determined covariance is in
# pairs_no_covariance alone; PanoramaStitching 0.7); the format's words the terminology's (detections).
# v14: detection covariances off: the bundle's observations the detections' pixels, its scale and width in pixels.
# v13: the bundle's scale the kept pairs' RANSAC scales pooled upstream (Estimation's PooledScale; SIMFIT++ deleted).
# v12: every bundle observation whitened by its detection's covariance, the detector's (the whitening switch retired); the bundle's
# scale σ₀ the variance factor of those covariances (bundle_scale, bundle_sigma, bundle_loss_width: no pixel unit); no point
# scale for the position noise.
const REGISTRATION_FORMAT = "panorama_vignetting/rotations/v17 (the near-duplicate pairs by view_overlap's shared fraction of the smaller footprint; the kept pairs; the near-duplicate pairs not chained; the track check, per solve its removals and rms; the bundle's inlier rms; pairs_removed the filters' only; plain detections, the bundle in pixels; the bundle's scale the pairs' RANSAC scales pooled upstream; every bundle observation whitened by its detection's covariance, the scale σ₀ of those covariances; the bundle's termination record, Core's ReturnCode and StopRule; the averaging on the pairs' Uncertain edges, the pairs with no covariance and the averaging's scale record; the averaging's and the bundle's optimization strategies, the bundle's scale record, the pairs' local_optimization_strategy; the clip mask and the canonical detection order of verified pairs v5; one bundle run at the pooled SIMFIT++ scale; the view-graph filters, their tolerances and removed pairs; LoMa's provider in the key and its provenance; the bundle's retcode, outcome, observations dropped and frames held at its start; pair samplers by stable_seed)"
# The registration's settings: its own table and those of the algorithms it composes.
const REGISTRATION_TABLES = ("registration", "rotation_problem", "ransac", "lag_overlap", "rotation_averaging", "rotation_bundle_problem")
registration_settings(C, inp, provider::Symbol) =
    stage_settings(C, inp, REGISTRATION_FORMAT, Dict{String, Any}(t => C.cfg[t] for t in REGISTRATION_TABLES if haskey(C.cfg, t)), provider)

# The verified pairs' key: the [registration] settings of the features, the matching and the verification — every one but the
# rotation averaging's and the bundle's, so changing those reuses the pairs. With pairs = "lag_overlap" the averaging loss aims
# the pairs past the lag, so it keys the pairs too.
# v3: the detector's selection and detection image (`selection`, `detection_image` and its noise keys), every ASIFT detection
# refined (VisualGeometryToolkitFeatures 0.9), the kept detections the described set.
# v4: per pair its SIMFIT++ noise scale (sigma, success).
# v6: the pairs are scored on the detections' covariance shapes (their median normalisation, retired) and each records its absolute rotation
# covariance (VisualGeometryToolkitEstimation's uncertain_rotation): the averaging's Uncertain edges.
# v5: no detection within three scales of a saturated pixel (Features' detection mask: the frames' clip masks), the detections in
# Features' canonical order, the frames' map Features' raw_frame_map.
# v10: detection covariances off (the detector's default): the pairs' residuals and scales in pixels.
# v12: :lag_overlap's pairs past the lag those whose frames share any view (VisualGeometryToolkitCore's view_overlap;
# PanoramaStitching 0.8; overlap_deg retired).
# v11: the format's words the terminology's (detections, correspondences: PanoramaStitching 0.7); the configs of record
# state no max_lag / overlap_deg with pairs = "all" (refused: no inert setting).
# v9: the pairs scored by Student-t + uniform at outlier_halfwidth_px (the MSAC scoring_threshold retired), each pair's ŝ that
# scorer's.
# v8: per pair its RANSAC scorer's own ŝ and degrees of freedom (sigma, dof); SIMFIT++ deleted.
# v7: the features are Features' described features as detected (their covariances the detector's, no normalisation), the pairs'
# residuals whitened by them (scoring_threshold in sigmas of the unit noise; the px threshold retired), per pair its rms and σ̂
# in those sigmas.
const PAIRS_FORMAT = "panorama_vignetting/verified_pairs/v12 (:lag_overlap's pairs past the lag by view_overlap; the terminology's words; plain detections, pixel residuals of the correspondences; Student-t + uniform pair scorer at outlier_halfwidth_px; per pair its RANSAC scorer's ŝ and dof; Features' described features, the detector's covariances; the pairs' residuals in sigmas of their unit noise; per pair its absolute rotation covariance (uncertain_rotation, the detections' covariance shapes in its residuals); the clip mask; Features' canonical detection order; per pair its SIMFIT++ scale; the detector's selection and detection image; LoMa's provider in the key and its provenance; features' pixels and covariances; per pair its rotation, counts and inlier matches)"
# The verified pairs' settings: [registration] but the keys of the steps after them (the view-graph filters, the near-duplicates,
# the bundle), the pairs'
# RANSAC ([rotation_problem], [ransac]), the lag pairs ([lag_overlap]) and, with "lag_overlap", the averaging that aims them.
const AFTER_PAIRS_KEYS = ("cycle_tol_deg", "orientation_tol_deg", "near_duplicate_overlap", "bundle")
function pairs_settings(cfg)
    r = cfg["registration"]
    s = Dict{String, Any}("registration" => Dict(k => v for (k, v) in r if k ∉ AFTER_PAIRS_KEYS))
    for t in ("rotation_problem", "ransac", "lag_overlap")
        haskey(cfg, t) && (s[t] = cfg[t])
    end
    r["pairs"] == "lag_overlap" && (s["rotation_averaging"] = cfg["rotation_averaging"])
    return s
end
# The detector's characterisation keys the pairs by its content too (its artifacts' SHA-256s are in it), beside its path in the
# settings; with "vst" only: with "tone" it is [decode]'s (`refuse_config`), whose content `stage_settings` keys already.
function pairs_stage_settings(C, inp, provider::Symbol)
    r = C.cfg["registration"]
    st = stage_settings(C, inp, PAIRS_FORMAT, pairs_settings(C.cfg), provider)
    r["detection_image"] == "vst" && r["sensor_characterization"] != "none" &&
        (st["sensor_characterization_sha256"] = sensor_record_sha256(r["sensor_characterization"]))
    return st
end

# The verified pairs as a record and back (frame indices 1-based; a matrix row by row): the features are the pairs' own,
# Features' described features (`StructArray{DescribedFeature}`, the detections with their covariances and descriptions), held
# as they are by the record's JLD2 file.
nan_null(x) = isnan(x) ? nothing : x
# LoMa's provenance (a `loma_provenance` NamedTuple per model, or a record's Dict read back) as a record: text values.
loma_record(loma) = Dict(string(k) => Dict(string(f) => string(v) for (f, v) in pairs(p)) for (k, p) in pairs(loma))
function pairs_record(vp::VerifiedPairs)
    pairs = [Dict("i" => i, "j" => j, "rotation" => [collect(e.rotation[r, :]) for r in 1:3], "inliers" => e.inliers,
                  # the pair's absolute rotation covariance (rad², so(3) chart; uncertain_rotation), or null when not determined
                  "covariance" => e.covariance === nothing ? nothing : [collect(e.covariance[r, :]) for r in 1:3],
                  "inlier_matches" => [collect(m) for m in vp.inlier_matches[(i, j)]],
                  "matches" => vp.stats[(i, j)].matches, "rms" => vp.stats[(i, j)].rms, "rotation_deg" => vp.stats[(i, j)].rotation_deg,
                  "sigma" => nan_null(vp.scales[(i, j)].sigma), "dof" => vp.scales[(i, j)].dof)
             for ((i, j), e) in sort!(collect(vp.verified); by = first)]
    return Dict{String, Any}("features" => vp.features, "pairs" => pairs, "loma" => loma_record(vp.loma))
end
function pairs_from_record(rec)
    num(x) = x === nothing ? NaN : Float64(x)
    key(p) = (Int(p["i"]), Int(p["j"]))
    verified = Dict(key(p) => (rotation = SMatrix{3, 3, Float64, 9}(permutedims(reduce(hcat, [Float64.(r) for r in p["rotation"]]))),
                               inliers = Int(p["inliers"]),
                               covariance = p["covariance"] === nothing ? nothing :
                                            SMatrix{3, 3, Float64, 9}(permutedims(reduce(hcat, [Float64.(r) for r in p["covariance"]]))))
                    for p in rec["pairs"])
    matches = Dict(key(p) => [(Int(m[1]), Int(m[2])) for m in p["inlier_matches"]] for p in rec["pairs"])
    stats = Dict(key(p) => (matches = Int(p["matches"]), inliers = Int(p["inliers"]), rms = Float64(p["rms"]),
                            rotation_deg = Float64(p["rotation_deg"])) for p in rec["pairs"])
    loma = (; descriptor = rec["loma"]["descriptor"], matcher = rec["loma"]["matcher"])
    scales = Dict(key(p) => (sigma = num(p["sigma"]), dof = Int(p["dof"])) for p in rec["pairs"])
    return VerifiedPairs(rec["features"], verified, matches, stats, scales, loma)
end

# ASIFT's tilts, `[registration] asift_tilts`: a Features `tilt_schedule` name ("imas25", "untilted") or a list of [t, φ] pairs.
asift_tilts(t::AbstractString) = Symbol(t)
asift_tilts(t::AbstractVector) = [Tuple(Float64.(p)) for p in t]

# The detector's settings of [registration]: `selection` ("spread", Features' SSC), `detection_image` ("tone" or "vst") and, with
# "vst" only, the threshold `noise_sigmas` and the record its frames are decoded again under, `sensor_characterization` (always
# stated, never defaulted: with "tone" it is [decode]'s, `refuse_config`, and the detector reads the decoded frames), their models
# VisualGeometryToolkitCore's resolver's (`frame_models`). `noise_sigmas` with "tone" is refused, never read; the 0.6 constants
# `electrons_per_dn` and `read_noise_dn` are refused by name (PanoramaStitching.refuse_retired_registration). `refuse_config`
# runs it when the config is read.
function detector_settings(r)
    if r["detection_image"] == "vst"
        noise_sigmas = Float64(r["noise_sigmas"]); characterization = String(r["sensor_characterization"])
    else
        haskey(r, "noise_sigmas") && throw(ArgumentError("[registration] noise_sigmas: read only by detection_image = \"vst\", not \"$(r["detection_image"])\"; remove"))
        noise_sigmas = characterization = nothing
    end
    return (; selection = Symbol(r["selection"]), detection_image = Symbol(r["detection_image"]), noise_sigmas, characterization)
end

# The frames' sensor models (`sensor_models`, of the detector's characterisation) and their LinearRaws decoded with them (the one
# black subtraction, Core's `LinearRaw`).
function frame_models(C, inp, characterization::AbstractString)
    dir = input_path(C.cfg["inputs"]["raw_dir"])
    mosaics = Dict(k => VgtkFeatures.load_raw_mosaic(joinpath(dir, n)) for (k, n) in enumerate(inp.names))
    models = sensor_models(() -> mosaics, characterization, length(inp.names))
    frames = [VgtkFeatures.load_raw(mosaics[k]; mode = Symbol(C.cfg["decode"]["mode"]), gain = Float64(C.cfg["decode"]["gain"]),
                                    sensor_model = models[k]) for k in eachindex(inp.names)]
    return frames, models
end

# The pairs' RANSAC local optimization, `[registration] local_optimization_strategy`: an Estimation strategy table
# (`{ type = "triggs", loss = "welsch", efficiency_constant = 2.5 }`, `{ type = "irls", loss = "tls" }`), read by Estimation's own
# loader (an unknown type, loss or key refused).
rotation_local_optimization(cfg) = VgtkEstimation.optimization_strategy_from_toml(cfg["rotation_problem"]["ransac"]["local_optimization_strategy"];
                                                                                  class = :rotation, section = "rotation_problem.ransac.local_optimization_strategy")
# The pairs' RANSAC ([rotation_problem.ransac], [ransac]): the Student-t + uniform scorer at `outlier_halfwidth_px` (Jeffreys scale
# and flat label priors), the local optimization above and the trial budget.
pair_ransac(cfg) = VgtkEstimation.Ransac(; scoring = VgtkCore.StudentTUniformScoring(VgtkEstimation.RotationProblem,
                                                                                    Float64(cfg["rotation_problem"]["ransac"]["outlier_halfwidth_px"]);
                                                                                    scale_prior = VgtkCore.JeffreysPrior(), label_prior = VgtkCore.FlatLabelPrior()),
                                         local_optimization_strategy = rotation_local_optimization(cfg),
                                         config = VgtkEstimation.RansacConfig(; max_trials = cfg["ransac"]["max_trials"], min_trials = cfg["ransac"]["min_trials"]))
# The pairs verified (`registration.pairs`, PAIR_SELECTIONS): AllPairs, or LagOverlap at [lag_overlap] max_lag, its averaging
# aiming the pairs past the lag the registration's [rotation_averaging].
pair_selection(cfg) = _pair_selection(PAIR_SELECTIONS[cfg["registration"]["pairs"]], cfg)
_pair_selection(::Type{AllPairs}, cfg) = AllPairs()
_pair_selection(::Type{LagOverlap}, cfg) = LagOverlap(; max_lag = cfg["lag_overlap"]["max_lag"], averaging = rotation_averaging(cfg))
rotation_averaging(cfg) = VgtkEstimation.RotationAveraging(; optimization_strategy = averaging_strategy(cfg))
# The view-graph filters ([registration] cycle_tol_deg, orientation_tol_deg: the averaging's tolerances, degrees; their table is
# [registration] until the naming gate accepts an upstream algorithm's tag as a table name) and the averaging, and the bundle ([rotation_bundle_problem.
# levenberg_marquardt]; `registration.bundle`, BUNDLE_REFINES: a TrackCheck of it, or it alone) at its stops.
filtered_averaging(cfg) = VgtkEstimation.FilteredRotationAveraging(; cycle_tol = deg2rad(cfg["registration"]["cycle_tol_deg"]),
                                                                   orientation_tol = deg2rad(cfg["registration"]["orientation_tol_deg"]),
                                                                   averaging = rotation_averaging(cfg))
bundle_algorithm(cfg) = _bundle_algorithm(BUNDLE_REFINES[cfg["registration"]["bundle"]], VgtkCore.LevenbergMarquardt(; optimization_strategy = bundle_strategy(cfg)))
_bundle_algorithm(::Type{TrackCheck}, lm) = TrackCheck(lm)
_bundle_algorithm(::Type{<:VgtkCore.LevenbergMarquardt}, lm) = lm
bundle_stops(cfg) = (s = cfg["rotation_bundle_problem"]["levenberg_marquardt"]["stops"]; VgtkCore.LmStops(; max_iter = s["max_iter"], ftol = s["ftol"], xtol = s["xtol"]))

# The verified pairs' product (verified_pairs/<key>/verified_pairs.jld2): the hour of ASIFT, LoMa and RANSAC of a set is spent
# once per matching setting, and produced before the view graph is built, so a failure past it (the averaging, the bundle)
# loses none of it. `produce_pairs` produces it when it is not produced and returns it; the registration's key reads its output's
# hash, its compute the record.
pairs_product(C, inp, provider::Symbol) = stage_product(C, "verified_pairs", pairs_stage_settings(C, inp, provider))
function produce_pairs(C, inp, lens, provider::Symbol; log)
    r = C.cfg["registration"]
    produce_stage(pairs_product(C, inp, provider), C; log, what = "verified pairs", inputs = raw_paths(C, inp)) do
        t0 = time()
        det = detector_settings(r)
        frames, models = det.detection_image === :vst ? frame_models(C, inp, det.characterization) : (inp.frames, nothing)
        vp = verify_pairs(frames, lens; frame_map = inp.frame_map, tone_percentile = r["tone_percentile"], detector_margin_px = r["detector_margin_px"],
                          asift_tilts = asift_tilts(r["asift_tilts"]), saturated = det.detection_image === :vst ? nothing : inp.saturated,
                          max_detections = r["max_detections"], det.selection, det.detection_image, sensor_models = models, det.noise_sigmas, descriptor = Symbol(r["descriptor"]), matcher = Symbol(r["matcher"]),
                          min_score = r["min_score"], pairs = pair_selection(C.cfg), min_inliers = r["min_inliers"], seed = r["seed"],
                          ransac = pair_ransac(C.cfg), execution_provider = provider, say = m -> say(log, m))
        merge!(pairs_record(vp), Dict{String, Any}("frames" => inp.names, "seconds" => time() - t0))
    end
end

function registration(C, inp, lens; log)
    r = C.cfg["registration"]
    # "auto" → "cuda" or "cpu" (warning why) here, before the keys; "cuda" that cannot run is an error naming why
    provider = VgtkFeatures.resolve_execution_provider(Symbol(r["execution_provider"]))
    # the verified pairs first: the registration's key folds their output's content hash (Eval's Merkle chain), so pairs
    # produced again with other outputs are registered again
    pp = produce_pairs(C, inp, lens, provider; log)
    p = stage_product(C, "registration", registration_settings(C, inp, provider); upstream = ("verified_pairs" => pp,))
    produce_stage(p, C; log, what = "registration", inputs = raw_paths(C, inp)) do
        rp = stage_record(pp)
        vp = (; pairs = pairs_from_record(rp), seconds = Float64(rp["seconds"]))
        t0 = time()
        reg = register_rotations(vp.pairs, lens; frame_map = inp.frame_map, averaging = filtered_averaging(C.cfg), bundle = bundle_algorithm(C.cfg),
                                 bundle_stops = bundle_stops(C.cfg), near_duplicate_overlap = r["near_duplicate_overlap"], say = m -> say(log, m))
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
                          "bundle_rms_px" => reg.bundle.rms, "bundle_inlier_rms_px" => reg.bundle.metadata.inlier_rms, "seconds" => vp.seconds + time() - t0,
                          # the near-duplicate pairs (averaged, not chained; 0-based) and the track check's counts with the first
                          # bundle's rms and inlier rms (px)
                          "pairs_near_duplicate" => [[i - 1, j - 1] for (i, j) in reg.near_duplicates],
                          # the pairs the view-graph filters kept: the averaging's edges (0-based)
                          "pairs_kept" => sort!([[i - 1, j - 1] for (i, j) in keys(reg.pairs) if haskey(reg.graph, i, j)]),
                          "track_check" => [Dict{String, Any}("removed" => h.removed, "rms_px" => h.rms, "inlier_rms_px" => h.inlier_rms) for h in reg.track_check],
                          # the bundle's termination record (Core's ReturnCode and StopRule); the observations it
                          # dropped at its start (a start prediction off the camera model's valid domain); the frames it held at their
                          # start rotation (left with no kept observation: their 0-based indices)
                          "bundle_return_code" => string(reg.bundle.retcode),
                          "bundle_stop_rule" => string(reg.bundle.stop_rule),
                          "bundle_dropped_start" => reg.bundle.metadata.n_dropped_start,
                          "bundle_frames_held" => [k - 1 for k in reg.bundle.metadata.frames_unconstrained],
                          # the bundle's robust loss width and scale (px) and its strategy
                          "bundle_loss_width" => reg.bundle.metadata.loss_width,
                          "bundle_optimization_strategy" => VgtkEstimation.config_dict(bundle_strategy(C.cfg)),
                          "bundle_scale" => VgtkCore.noise_scale(reg.bundle),
                          # the bundle's noise scale (the kept pairs' pooled RANSAC ŝ, px of the detections' covariances) and the LM
                          # iterations its one run took
                          "bundle_sigma" => reg.bundle_sigma, "bundle_iterations_run" => reg.bundle.stats.nsteps,
                          # the bundle's termination record as Core's values (the results file's `fit`: vgio/panorama's)
                          "bundle_fit" => (retcode = reg.bundle.retcode, stop_rule = reg.bundle.stop_rule,
                                           iterations = reg.bundle.stats.nsteps, rms = reg.bundle.rms, n = reg.bundle.n),
                          # what ran LoMa (VisualGeometryToolkitFeatures' loma_provenance, carried by the verified pairs): the models,
                          # the ONNX Runtime provider used, its threads and, on CUDA, the driver, runtime and cuDNN versions
                          "loma" => loma_record(reg.loma))
    end
    rec = stage_record(p)
    return (; rotations = [SMatrix{3, 3, Float64, 9}(reduce(vcat, permutedims.(Vector{Float64}.(R)))) for R in rec["rotations"]],
            pairs = rec["pairs"], pairs_removed = rec["pairs_removed"], pairs_no_covariance = rec["pairs_no_covariance"],
            bundle_rms_px = rec["bundle_rms_px"], bundle_inlier_rms_px = rec["bundle_inlier_rms_px"],
            pairs_near_duplicate = rec["pairs_near_duplicate"], pairs_kept = rec["pairs_kept"], track_check = rec["track_check"],
            bundle_sigma = rec["bundle_sigma"], bundle_iterations_run = rec["bundle_iterations_run"], seconds = something(rec["seconds"], 0.0),
            bundle_return_code = rec["bundle_return_code"], bundle_stop_rule = rec["bundle_stop_rule"],
            bundle_dropped_start = rec["bundle_dropped_start"], bundle_frames_held = rec["bundle_frames_held"], provenance = get(rec, "provenance", nothing),
            loma = rec["loma"], product = p,
            # the bundle's termination record (no estimate, no metadata): what vgio/panorama's `fit` stores
            bundle_fit = (b = rec["bundle_fit"]; VgtkCore.FitResult(nothing, b.retcode, b.stop_rule, VgtkCore.SolveStats(; nsteps = b.iterations), b.rms, b.n, nothing)))
end

# ----------------------------------------------------------------------------- vignetting spaces from the config

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
    c = VgtkCore.lens_coverage(lens, rim)                                 # Core's: the coverage less the rim margin
    c isa VgtkCore.Circle && return norm(SVector(c.center[1], c.center[2]) - center) + c.r
    o, e = domain_box(lens, W, H, rim)
    return maximum(norm(SVector(x, y) - center) for x in (o[1], o[1] + e[1]), y in (o[2], o[2] + e[2]))
end

# V's model and start (VisualRadiometryToolkitCore's LogVignetting: a spline of log V, log V ≡ 0 at the start) and its free directions:
# a radial profile's pole-regular subspace, every control point of a tensor surface. `kind` is a tag of Core's VIGNETTING_TAGS
# ("radial", "tensor_product"; the retired "tensor" is refused by name), the field log V is a spline of.
function vignetting_space(kind, C, lens, W, H)
    v = C.cfg["vignetting"]; rim = Float64(C.cfg["domain"]["rim_margin_px"])
    T = VrtkCore.VIGNETTING_TAGS[kind]
    if T === VrtkCore.RadialVignetting
        rc = v["radial_center"]
        center = rc == "principal_point" ? SVector{2, Float64}(VgtkCore.principal_point(VgtkCore.model(lens).intrinsics)) :
                 rc == "circle_center" ? coverage_center(lens) : SVector{2, Float64}(rc...)
        radius = v["radial_radius"] == "domain" ? domain_radius(lens, center, W, H, rim) + 1.0 : Float64(v["radial_radius"])
        s = VrtkCore.RadialVignetting(v["radial_controls"], center, radius; degree = 3)
        m = VrtkCore.LogVignetting(VrtkCore.with_control_points(s, zeros(VrtkCore.n_control_points(s))))
        return m, VrtkCore.pole_regular_basis(m)
    elseif T === VrtkCore.TensorProductVignetting
        v["tensor_box"] == "domain" || error("vignetting.tensor_box = $(v["tensor_box"]) is not \"domain\"")
        o, e = domain_box(lens, W, H, rim)
        s = VrtkCore.TensorProductVignetting(Tuple(v["tensor_controls"]), VgtkCore.Rect(o[1], o[2], e[1], e[2]); degree = 3)
        n = VrtkCore.n_control_points(s)
        m = VrtkCore.LogVignetting(VrtkCore.with_control_points(s, zeros(n)))
        return m, Matrix{Float64}(LinearAlgebra.I, n, n)
    end
    error("vignetting model \"$kind\" is not \"radial\" or \"tensor_product\" (log V is the spline of one)")
end

# The fit's observation noise of the config beside the value's (the frames' own: their variance, or the read-noise fallback of a
# plain image): none, or the sensor position with the sensor point's σ `point_scale_px` (camera pixels).
function fit_observation_noise(v, point_scale_px)
    N = PanoramaStitching.OBSERVATION_NOISES[v["observation_noise"]]      # "none" or "position" (an ArgumentError otherwise)
    N === nothing && return nothing
    point_scale_px === nothing && error("vignetting.observation_noise = \"position\" is refused in this release: it awaits " *
                                        "each pixel's position noise from the frames' rotation covariance (after the release)")
    return PositionNoise(; point_covariance = VgtkCore.CovMat{2}(σ = point_scale_px), iterations = v["observation_noise_iterations"])
end

# The CG's stops ([vignetting_self_calibration.stops]: rtol, max_iter); a Cholesky solve reads none, and a stops table with it is
# refused (no inert setting).
fit_stops(::VgtkCore.NormalCG, v) = VgtkCore.ToleranceStops(; rtol = v["stops"]["rtol"], max_iter = v["stops"]["max_iter"])
fit_stops(::SparseCholesky, v) = haskey(v, "stops") ?
    error("vignetting_self_calibration.stops: read only by the NormalCG solver; remove it with SparseCholesky") : nothing
# One fit of the config's settings; `vignetting` given (a calibration's V) holds V fixed: the frames' gains alone.
# `point_scale_px` the sensor point's σ of PositionNoise (camera pixels).
function fit(kind, C, lens, dev, rotations, W, H; vignetting = nothing, point_scale_px = nothing)
    v = C.cfg["vignetting_self_calibration"]; d = C.cfg["domain"]
    m, Z = vignetting === nothing ? vignetting_space(kind, C, lens, W, H) :
           (vignetting, zeros(0, 0))                                      # any vignetting held fixed (a Core model or a callable)
    solver = LINEAR_SOLVERS[v["solver"]]()                                # NormalCG or SparseCholesky by tag (an ArgumentError otherwise)
    stops = fit_stops(solver, v)
    gc = PanoramaStitching.GAIN_CORRECTIONS[v["gain_correction"]]                        # "none", "intensity" or "channel" (an ArgumentError otherwise)
    problem = VgtkCore.EstimationProblem(VrtkCore.AbstractVignetting, dev.frames,
                                         (; camera = lens, rotations, saturated = dev.saturated, frame_map = dev.frame_map,
                                          plane_maps = dev.plane_maps, vignetting = m, free_directions = Z))
    return VgtkCore.solve(problem, VignettingSelfCalibration(; gain_correction = gc === nothing ? nothing : gc(),
                          rim_margin_px = d["rim_margin_px"], saturation_margin_px = d["saturation_margin_px"], fov_deg = d["fov_deg"], plane = d["plane"],
                          chart_spacing_deg = v["chart_spacing_deg"], penalty_order = v["penalty_order"], penalty_weight = v["penalty_weight"],
                          vignetting_penalty_weight = v["vignetting_penalty_weight"], solver,
                          observation_noise = fit_observation_noise(v, point_scale_px)); stops)
end
# The frames' exposure gains of a fit (K × 3), for the seam score and the renders: its metadata's when it fitted them, else none.
fit_gains(C, res) = C.cfg["vignetting_self_calibration"]["gain_correction"] == "none" ? nothing : res.metadata.gains
# A K × 3 gain matrix as text, one frame per row: g_k,R / g_k,G / g_k,B.
gains_text(G; d = 4) = join([join(round.(G[k, :]; digits = d), " / ") for k in axes(G, 1)], ", ")

# ----------------------------------------------------------------------------- views, renders, images

"No vignetting correction: V ≡ 1."
no_vignetting(u, f) = 1.0

"An H × W × 3 linear image as 8-bit sRGB at a fixed exposure, each plane times its white-balance factor (NaN, no frame: black): Features' `srgb_image`."
save_srgb(path, img, exposure, wb) = PNGFiles.save(path, VgtkFeatures.srgb_image(img; exposure, white_balance = wb))
"A seam-ratio map: red where the other frames are brighter than the shown one, blue darker, ±`lim` in log; grey unmeasured."
function save_ratio(path, M; lim = 0.3)
    col(v) = isnan(v) ? CT.RGB{FPN.N0f8}(0.5, 0.5, 0.5) : (x = clamp(v / lim, -1, 1);
              x >= 0 ? CT.RGB{FPN.N0f8}(1, 1 - x, 1 - x) : CT.RGB{FPN.N0f8}(1 + x, 1 + x, 1))
    PNGFiles.save(path, col.(M))
end

# ----------------------------------------------------------------------------- the results of record

# A run's results of record (a fit's): PanoramaStitching's vgio/panorama/v2.0 record (`save_panorama`) — the camera, the frames' rotations camera → world
# (d = R_k·f_k, the world the first frame's camera frame; zero translations: the camera only rotates; each frame's RAW file
# its image), the fitted V, the frames' gains (none when vignetting_self_calibration.gain_correction = "none") and the bundle's termination record (`fit`) — read back by
# `load_panorama`; and V over the frame's pixels (one value per frame pixel, at its centre in the camera's pixels: the
# frames' map), a Float32 `vgio/raster` read back by `VgtkIO.load_raster`. The panorama record's provenance states the sensor record
# the frames were decoded under (`sensor_characterization_sha256`, `nothing` for "none": `decoded_frames`).
function write_results(C, stem, lens, names, rotations, bundle_fit, res, frame_map, (Hf, Wf); sensor_characterization_sha256)
    v = res.u
    tr = VgtkCore.Trajectory(VgtkCore.CameraFrame, VgtkCore.WorldFrame,
                             Dict(k => VgtkCore.EuclideanMap(R, zero(SVector{3, Float64}) .* VgtkCore.m) for (k, R) in enumerate(rotations));
                             images = Dict(enumerate(names)))
    # (the file's generator names PanoramaStitching beside IO: `save_panorama` states it)
    save_panorama(joinpath(C.run, stem * ".json"), lens, tr, v, fit_gains(C, res), bundle_fit;
                  provenance = (; sensor_characterization_sha256 = something(sensor_characterization_sha256, "none")))
    # V over the frame's pixel grid (each frame pixel's centre through the frames' map): a DATA result, IO's Float32
    # `vgio/raster` (owner ruling "RAW + results only"), `values[r, c]` the frame's pixel at row r, column c
    M = Float32[v(frame_map(SVector(j - 0.5, i - 0.5))) for i in 1:Hf, j in 1:Wf]
    VgtkIO.with_caller(PanoramaStitching) do
        VgtkIO.save_raster(joinpath(C.run, stem * "_map.json"), M; frame = VgtkCore.PixelFrame{:colmap}, nan = :none)
    end
    return nothing
end

# ----------------------------------------------------------------------------- the run

function run_example(config_path; force = false)
    C = load_config(config_path)
    # the run's directory is its product's, keyed by the config, the code, this driver and the registration's output (Eval's
    # Merkle chain): known once the registration is produced, so until then the log is the set's stages.log (the decode and the
    # stages, on disk as they run), copied into the run's run.log
    mkpath(C.out)
    stages_log = joinpath(C.out, "stages.log")
    log = open(stages_log, "w")
    T = Dict{String, Float64}()
    say(log, "panorama_vignetting: $(C.cfg["name"]) ($(C.path))")
    t = time(); inp = decoded_frames(C; masks = true, log); T["decode (or cache)"] = time() - t
    K = length(inp.frames); Hf, Wf, _ = size(inp.frames[1]); W, H = Int.(PanoramaStitching._covered_extent(inp.frame_map, Wf, Hf))   # the frames' map: (2Wf, 2Hf) for :bin2
    say(log, "frames: $K, $(Wf) × $(Hf) bin2 ($(W) × $(H) camera pixels), saturated quads $(sum(count, inp.saturated))")
    lens = load_lens(C; raw_width = W, raw_height = H)
    reg = registration(C, inp, lens; log)
    close(log)
    T["registration"] = reg.seconds
    # the run: its fits, seams and renders are this driver's code, so the driver's content keys it (Eval keys a product by the
    # packages' code trees, never a driver's source; the stages above key their records' formats instead)
    # (Eval's run_key leaves a config's [render] out, as panel knobs; here it sets the seams' up frame and the renders: a part)
    p = VgtkEval.Experiments.Product(C.experiment, "run", C.cfg; output = "summary.md", files = Dict("driver" => @__FILE__),
                                     parts = Dict("render" => C.cfg["render"]), upstream = ("registration" => reg.product,), code = C.code)
    ran = false
    dir = VgtkEval.Experiments.produce(p; driver = DRIVER, config_source = C.path, force,
                                       inputs = [raw_paths(C, inp); input_path(C.cfg["inputs"]["lens"])]) do dir
        ran = true
        open(joinpath(dir, "run.log"), "w") do io
            write(io, read(stages_log))
            say(io, "output $dir")
            run_outputs((; C..., run = dir), inp, lens, reg, T, W, H; log = io)
        end
        nothing                                                       # the run writes its own outputs, summary.md last
    end
    ran || println("run: reused $dir")
    return dir
end

# The run's outputs in its product's directory `C.run`: the fit of record, the seams before and after, the renders, the
# comparison fits and summary.md (last).
function run_outputs(C, inp, lens, reg, T, W, H; log)
    Hf, Wf, _ = size(inp.frames[1])
    uc = C.cfg["render"]["up"]
    uc in ("level", "first_frame") || error("render.up = $uc is not \"level\" or \"first_frame\"")
    sweep = VgtkCore.solve(VgtkCore.EstimationProblem(SweepUp, reg.rotations), RollFreeUp())   # the sweep's up; its least eigenvalue the roll-free check
    up = uc == "level" ? up_frame(sweep.u) : SMatrix{3, 3, Float64, 9}(LinearAlgebra.I)
    uc == "level" && say(log, "up frame: level (the frames' x axes span a plane to $(round(sweep.metadata.eigenvalues[1]; sigdigits = 2)), least eigenvalue)")

    # the frames on the compute backend: the images (renders) and the masks (fit, seams) apart. The backend changes no result
    # (device and host agree to 1e-6 in log V, the package's tests); PANORAMA_VIGNETTING_DEVICE overrides the config's for one run
    # (a shared GPU without free memory).
    device = get(ENV, "PANORAMA_VIGNETTING_DEVICE", C.cfg["compute"]["device"])
    device in ("cuda", "cpu") || error("compute.device = $device is not \"cuda\" or \"cpu\"")
    say(log, "compute: $device$(haskey(ENV, "PANORAMA_VIGNETTING_DEVICE") ? " (PANORAMA_VIGNETTING_DEVICE)" : "")")
    move = device == "cuda" ? CuArray : identity
    dev = (; frames = move.(inp.frames), saturated = move.(inp.saturated), inp.frame_map, inp.plane_maps)   # the maps: isbits
    kind = C.cfg["vignetting"]["model"]
    fit(kind, C, lens, dev, reg.rotations, W, H)                          # compile
    t = time(); res = fit(kind, C, lens, dev, reg.rotations, W, H); T["VignettingSelfCalibration ($kind, warm)"] = time() - t
    m = res.metadata
    say(log, "VignettingSelfCalibration ($kind): $(res.retcode), $(res.stats.nsteps) iterations, weighted rms $(round(res.rms; digits = 4)), " *
             "MAD $(round(m.residual_mad; digits = 4)), $(res.n) rows, $(m.n_pixels) pixels, chart $(m.chart.n)², $(m.n_active) active")
    say(log, "  observation noise $(m.observation_noise) (the model's log I moved by $(m.log_change) in the last solve); " *
             "stages (s): $(m.seconds)")
    C.cfg["vignetting_self_calibration"]["gain_correction"] == "none" ||
        say(log, "  gains ($(C.cfg["vignetting_self_calibration"]["gain_correction"]); g_k R / G / B, g₁ = 1): $(gains_text(m.gains))")
    write_results(C, "panorama", lens, inp.names, reg.rotations, reg.bundle_fit, res, inp.frame_map, (Hf, Wf); inp.sensor_characterization_sha256)

    # the seam scores, before and after
    sc = C.cfg["seams"]; d = C.cfg["domain"]
    ext = sc["extent"] == "coverage" ? coverage_extent(lens, dev.frames, reg.rotations, up, dev.frame_map) : Tuple(sc["extent"])
    sview = panorama_view(VgtkCore.Equirectangular(), ext, sc["deg_per_pixel"]; up)
    skw = (; saturated = dev.saturated, saturation_margin_px = d["saturation_margin_px"], frame_map = dev.frame_map,
           rim_margin_px = d["rim_margin_px"], fov_deg = d["fov_deg"], plane = d["plane"], plane_maps = dev.plane_maps, chunk = sc["chunk"])
    t = time()
    before = score_seams(lens, dev.frames, reg.rotations, sview; vignetting = no_vignetting, skw...)
    after = score_seams(lens, dev.frames, reg.rotations, sview; vignetting = res.u, gains = fit_gains(C, res), skw...)
    T["score_seams (two)"] = time() - t
    save_ratio(joinpath(C.run, "seams_uncorrected.png"), Array(before.ratio)); save_ratio(joinpath(C.run, "seams_corrected.png"), Array(after.ratio))
    say(log, "seams (saturated directions excluded from every frame, the fit's symmetric masks): uncorrected rms $(round(before.rms; digits = 4)) MAD $(round(before.mad; digits = 4)); " *
             "corrected rms $(round(after.rms; digits = 4)) MAD $(round(after.mad; digits = 4)); $(after.n) pixels")

    # the renders: the real images (no saturation mask), corrected and uncorrected at one exposure
    rc = C.cfg["render"]
    rext = rc["extent"] == "coverage" ? ext : Tuple(rc["extent"])
    rkw = (; frame_map = dev.frame_map, rim_margin_px = d["rim_margin_px"], fov_deg = d["fov_deg"], plane_maps = dev.plane_maps)
    V = res.u
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
    pp = SVector{2, Float64}(VgtkCore.principal_point(VgtkCore.model(lens).intrinsics))
    rows = Any[(kind, res, after, vignetting_summary(V, pp, inp.frame_map, (Hf, Wf); mask = in_domain))]
    for k in C.cfg["vignetting"]["compare_models"]
        t = time(); r2 = fit(k, C, lens, dev, reg.rotations, W, H); T["VignettingSelfCalibration ($k, compare)"] = time() - t
        s2 = score_seams(lens, dev.frames, reg.rotations, sview; vignetting = r2.u, gains = fit_gains(C, r2), skw...)
        write_results(C, "panorama_compare_$k", lens, inp.names, reg.rotations, reg.bundle_fit, r2, inp.frame_map, (Hf, Wf); inp.sensor_characterization_sha256)
        push!(rows, (k, r2, s2, vignetting_summary(r2.u, pp, inp.frame_map, (Hf, Wf); mask = in_domain)))
        say(log, "compare $k: $(r2.retcode), weighted rms $(round(r2.rms; digits = 4)), seams rms $(round(s2.rms; digits = 4)) MAD $(round(s2.mad; digits = 4))")
    end
    write_summary(C, inp, reg, rows, before, ext, exposure, T, W, H, pp)
    return nothing
end

function write_summary(C, inp, reg, rows, before, ext, exposure, T, W, H, pp)
    f(x; d = 4) = string(round(x; digits = d))
    open(joinpath(C.run, "summary.md"), "w") do io
        println(io, "# $(C.cfg["name"]): panorama vignetting\n")
        println(io, "Config of record: `$(basename(C.path))`. Data root `\$VGTK_DATA_ROOT`; RAW `$(C.cfg["inputs"]["raw_dir"])`, lens `$(C.cfg["inputs"]["lens"])`.\n")
        println(io, "- Frames: $(length(inp.frames)), decoded $(C.cfg["decode"]["mode"]) gain $(C.cfg["decode"]["gain"]) → $(size(inp.frames[1], 2)) × $(size(inp.frames[1], 1)); saturated or defect quads (all frames): $(sum(count, inp.saturated)).")
        println(io, "- Sensor record of the frames' decode (and of the seam scores): $(something(inp.sensor_characterization_sha256, "none (the file's black, no defects)")) ([decode] sensor_characterization = $(C.cfg["decode"]["sensor_characterization"])).")
        bund = "$(f(reg.bundle_rms_px; d = 3)) px (one LM run of $(reg.bundle_iterations_run) iterations, the strategy " *
               "$(C.cfg["rotation_bundle_problem"]["levenberg_marquardt"]["optimization_strategy"]) at σ̂ $(f(reg.bundle_sigma; d = 4)), the pairs' pooled RANSAC scale of the detections' covariances)"
        println(io, "- Registration: $(reg.pairs) verified pairs ($(length(reg.pairs_no_covariance)) with no determined rotation " *
                    "covariance, not averaged$(isempty(reg.pairs_no_covariance) ? "" : ": " * join(["$(p[1])-$(p[2])" for p in reg.pairs_no_covariance], ", "))" *
                    "; $(length(reg.pairs_removed)) removed by the view-graph filters" *
                    "$(isempty(reg.pairs_removed) ? "" : ": " * join(["$(p[1])-$(p[2])" for p in reg.pairs_removed], ", "))), bundle rms $bund$(reg.provenance === nothing ? "" : "; $(reg.provenance)"); " *
                    "bundle termination $(reg.bundle_return_code) ($(reg.bundle_stop_rule)), $(reg.bundle_dropped_start) observations dropped at its start: not imaged there; " *
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
            println(io, "| $k | $(r.retcode) | $(r.stats.nsteps) | $(f(r.rms)) | $(f(m.residual_rms)) | $(f(m.residual_mad)) | $(r.n) | " *
                        "$(f(v.V_min; d = 3)) / $(f(v.V_max; d = 3)) | ($(f(v.max_at[1]; d = 0)), $(f(v.max_at[2]; d = 0))) | " *
                        "$(f(v.max_offset_px; d = 0)) | $(f(100v.asymmetry; d = 1)) % |")
        end
        println(io, "\nFrame gains (vignetting_self_calibration.gain_correction = $(C.cfg["vignetting_self_calibration"]["gain_correction"])): " *
                    (C.cfg["vignetting_self_calibration"]["gain_correction"] != "none" ? "g_k R / G / B of the fit of record (g₁ = 1) $(gains_text(rows[1][2].metadata.gains)); the seam score and the corrected renders divide them out." :
                     "none fitted, the frames taken at one exposure."))
        println(io, "\nStage times of the fit of record (s): $(rows[1][2].metadata.seconds)\n")
        println(io, "## Wall times (s)\n")
        for (k, v) in sort(collect(T); by = first)
            println(io, "- $k: $(f(v; d = 2))")
        end
        println(io, "\n## Files\n")
        for x in sort(readdir(C.run))
            isfile(joinpath(C.run, x)) && !startswith(x, "_") && println(io, "- `$x`")    # not the product's records
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    cfgs = filter(!=("--force"), ARGS)
    length(cfgs) == 1 || error("usage: julia --project=<env> run.jl <config.toml> [--force]")
    run_example(only(cfgs); force = "--force" in ARGS)              # --force: the run produced again (its stages are reused)
end
