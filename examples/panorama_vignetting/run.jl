# The panorama vignetting example: RAW frames of a camera that only rotated → their rotations → the camera's vignetting from
# the frames alone (interpolation free) → the seam scores before and after → corrected and uncorrected panoramas.
#
#     julia --project=<env> examples/panorama_vignetting/run.jl examples/panorama_vignetting/config_fx3_14mm.toml
#
# Every setting is in the config (a config of record: nothing here has a default). Inputs are read under $VGTK_DATA_ROOT; the
# outputs, and the decode cache, go to the config's `output` folder. The stages are functions, so in a REPL
# `include("run.jl"); run_example("config.toml")` runs it all and each stage can be rerun alone. See README.md.

using LibRaw_jll                                   # Features' LibRaw decoder (its package extension)
using CUDA
# ONNX Runtime's CUDA provider (registration.execution_provider = "cuda") needs cuDNN loaded beside CUDA, and CUDA's runtime
# 12.x (CUDA.set_runtime_version!(v"12.9")); cuDNN is loaded only where the environment holds it, so one without it runs "cpu"
if Base.find_package("cuDNN") !== nothing
    import cuDNN
end
using PanoramaStitching
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
import VisualGeometryToolkitFeatures as VgtkFeatures
import VisualGeometryToolkitIO as VgtkIO
using StaticArrays: SVector, SMatrix, SA
using LinearAlgebra: LinearAlgebra, norm, normalize, dot, cross, eigen, Symmetric
using Statistics: mean, median, quantile
using Printf: @sprintf
import PNGFiles, JSON, TOML

const FPN = PNGFiles.FixedPointNumbers
const CT = PNGFiles.ColorTypes

# ----------------------------------------------------------------------------- configuration

data_root() = get(ENV, "VGTK_DATA_ROOT", "") |> r -> isempty(r) ?
    error("VGTK_DATA_ROOT is not set: bind it to the data root that holds the config's inputs (README, \"Inputs\")") : r
input_path(rel) = joinpath(data_root(), rel)

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

function load_config(path)
    cfg = TOML.parsefile(path)
    refuse_renamed_keys(cfg["vignetting"], "$path [vignetting]")
    refuse_circle(cfg["inputs"], "$path [inputs]")
    out = joinpath(dirname(abspath(path)), cfg["output"])
    mkpath(out)
    return (; cfg, out, path = abspath(path))
end

say(io, msg) = (println(msg); println(io, msg); flush(io))

# ----------------------------------------------------------------------------- the decode cache
#
# Each RAW frame is decoded once (Features' load_raw_mosaic, then LinearRaw: exactly load_raw) and cached under
# <output>/decoded as two files side by side:
#   <stem>.png            16-bit RGB: per plane the photosite CODE SUM of the quad — R and B 2·raw, G raw₁ + raw₂ — so the
#                         linear value is (code/2 − black_p)/(white − black_p)·gain: lossless (load_raw's Float64 values to a
#                         few ulp, no 16-bit quantization of the linear value)
#   <stem>_saturated.png  8-bit, 0 or 1: load_raw's per-quad saturation (any photosite at the white level)
# and decode.json once per set: the decode settings, the black and white levels, the planes' sample offsets and per frame the
# RAW file's SHA-256. A frame is decoded again when its RAW's hash or a decode setting changes.

const CACHE_FORMAT = "panorama_vignetting/decode/v2 (code sums, daylight_wb)"

# the planes' black levels: R, G (both greens), B
function plane_blacks(mosaic)
    b = mosaic.black
    b[2] == b[4] || error("the two greens' black levels differ ($(b[2]), $(b[4])): the code-sum cache needs them equal")
    return (b[1], b[2], b[3])
end

function encode_codes(lin, blacks, white, gain)
    H, W, _ = size(lin.image)
    codes = Array{UInt16}(undef, H, W, 3)
    for p in 1:3, j in 1:W, i in 1:H
        c = 2 * (lin.image[i, j, p] / gain * (white - blacks[p]) + blacks[p])
        r = round(c)
        abs(c - r) < 1.0e-6 && 0 <= r <= typemax(UInt16) || error("pixel ($i, $j, $p): $c is not a code sum")
        codes[i, j, p] = UInt16(r)
    end
    return codes
end
decode_codes(codes, blacks, white, gain) =
    Float32.(cat(((Float64.(view(codes, :, :, p)) ./ 2 .- blacks[p]) ./ (white - blacks[p]) .* gain for p in 1:3)...; dims = 3))

function save_codes(path, codes)
    n(x) = reinterpret(FPN.N0f16, x)
    PNGFiles.save(path, CT.RGB{FPN.N0f16}.(n.(view(codes, :, :, 1)), n.(view(codes, :, :, 2)), n.(view(codes, :, :, 3))))
end
function load_codes(path)
    img = PNGFiles.load(path)
    return cat((reinterpret.(UInt16, getfield.(img, f)) for f in (:r, :g, :b))...; dims = 3)
end
save_mask(path, m) = PNGFiles.save(path, CT.Gray{FPN.N0f8}.(reinterpret.(FPN.N0f8, UInt8.(m))))
load_mask(path) = reinterpret.(UInt8, CT.gray.(PNGFiles.load(path))) .!= 0

"The frames as linear images (H × W × 3 Float32) and, with `masks = true`, their saturation masks; decoded once, cached."
function decoded_frames(C; masks::Bool, log = devnull)
    dec = C.cfg["decode"]; mode = dec["mode"]; gain = Float64(dec["gain"])
    mode == "bin2" || error("decode.mode = $mode: the code-sum cache is written for bin2")
    dir = joinpath(C.out, "decoded"); mkpath(dir)
    raws = sort([f for f in readdir(input_path(C.cfg["inputs"]["raw_dir"])) if endswith(f, C.cfg["inputs"]["raw_extension"])])
    meta_path = joinpath(dir, "decode.json")
    meta = isfile(meta_path) ? JSON.parsefile(meta_path) : Dict{String, Any}()
    same_settings = get(meta, "format", "") == CACHE_FORMAT && get(meta, "mode", "") == mode && get(meta, "gain", NaN) == gain
    entries = same_settings ? Dict(e["raw"] => e for e in get(meta, "frames", [])) : Dict{String, Any}()
    frames = Array{Float32, 3}[]; sats = BitMatrix[]; new_entries = Any[]
    blacks = same_settings ? Tuple(Float64.(meta["black"])) : nothing; white = same_settings ? Float64(meta["white"]) : NaN
    offsets = same_settings ? [Float64.(o) for o in meta["plane_offsets"]] : nothing
    wb = same_settings ? Float64.(meta["daylight_wb"]) : nothing
    for f in raws
        src = joinpath(input_path(C.cfg["inputs"]["raw_dir"]), f); stem = splitext(f)[1]
        h = open(VgtkCore.sha256_hex, src)
        img_path = joinpath(dir, stem * ".png"); sat_path = joinpath(dir, stem * "_saturated.png")
        e = get(entries, f, nothing)
        if e !== nothing && e["sha256"] == h && isfile(img_path) && isfile(sat_path)
            codes = load_codes(img_path)
            masks && push!(sats, load_mask(sat_path))
        else
            mosaic = VgtkFeatures.load_raw_mosaic(src; decoder = VgtkFeatures.LibRawDecoder())
            lin = VgtkFeatures.LinearRaw(mosaic; mode = Symbol(mode), gain)
            bl = plane_blacks(mosaic)
            (blacks === nothing || bl == blacks) && (isnan(white) || white == mosaic.white) ||
                error("$f: black/white levels differ from the set's ($bl, $(mosaic.white) vs $blacks, $white)")
            blacks = bl; white = mosaic.white; offsets = [Vector(o) for o in lin.plane_offsets]
            wb === nothing && (wb = collect(Float64.(mosaic.metadata.daylight_wb)))
            codes = encode_codes(lin, blacks, white, gain)
            back = decode_codes(codes, blacks, white, gain)
            maximum(abs.(Float64.(back) .- Float32.(lin.image))) <= 2 * eps(1.0f0) ||
                error("$f: the cache does not round-trip load_raw's values")
            save_codes(img_path, codes); save_mask(sat_path, lin.saturated)
            masks && push!(sats, lin.saturated)
            say(log, "  decoded $f ($(count(lin.saturated)) saturated quads)")
        end
        push!(frames, decode_codes(codes, blacks, white, gain))
        push!(new_entries, Dict("raw" => f, "sha256" => h, "image" => stem * ".png", "saturated" => stem * "_saturated.png"))
    end
    open(meta_path, "w") do io
        JSON.print(io, Dict("format" => CACHE_FORMAT, "mode" => mode, "gain" => gain, "black" => collect(blacks), "white" => white,
                            "plane_offsets" => offsets, "daylight_wb" => wb, "raw_dir" => C.cfg["inputs"]["raw_dir"],
                            "frames" => new_entries), 2)
    end
    return (; names = raws, frames, saturated = sats, plane_offsets = [SVector{2, Float64}(o...) for o in offsets], daylight_wb = wb,
            levels = (; black = collect(blacks), white, gain))
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

const LOSSES = Dict("huber" => VgtkCore.HuberShape(), "cauchy" => VgtkCore.CauchyShape())

# The registration's cache key: its record's format, its settings, the inputs, the decode settings and the RAW files' hashes.
# v3: PanoramaStitching 0.2.2 seeds each pair's sampler by stable_seed(seed, i, j), so a v2 record's rotations are not this
# code's; they are recomputed.
const REGISTRATION_FORMAT = "panorama_vignetting/rotations/v3 (the bundle's retcode, outcome and start finiteness; pair samplers by stable_seed)"
function registration_key(C, inp)
    canon(d) = join(["$k=$(d[k])" for k in sort(collect(keys(d)))], ";")
    return VgtkCore.sha256_hex(join([REGISTRATION_FORMAT, canon(C.cfg["registration"]), canon(C.cfg["inputs"]), canon(C.cfg["decode"]),
                                     (open(VgtkCore.sha256_hex, joinpath(input_path(C.cfg["inputs"]["raw_dir"]), n)) for n in inp.names)...], "|"))
end

function registration(C, inp, lens; log)
    r = C.cfg["registration"]
    key = registration_key(C, inp)
    path = joinpath(C.out, "rotations.json")
    if isfile(path)
        rec = JSON.parsefile(path)
        # a record written before 2026-11-04 holds no bundle loss width: no point scale (IntensityAndPositionNoise then refuses to run;
        # delete the record to recompute it)
        if rec["key"] == key
            say(log, "registration: reused $(path)$(haskey(rec, "provenance") ? " ($(rec["provenance"]))" : "")")
            return (; rotations = [SMatrix{3, 3, Float64, 9}(reduce(vcat, permutedims.(Vector{Float64}.(R)))) for R in rec["rotations"]],
                    pairs = rec["pairs"], bundle_rms_px = rec["bundle_rms_px"], seconds = something(rec["seconds"], 0.0),
                    bundle_retcode = rec["bundle_retcode"], bundle_outcome = rec["bundle_outcome"],
                    bundle_start_nonfinite = rec["bundle_start_nonfinite"], provenance = get(rec, "provenance", nothing),
                    point_scale_px = haskey(rec, "bundle_loss_width_px") ? rec["bundle_loss_width_px"] / rec["bundle_scale_multiple"] : nothing)
        end
    end
    ep = r["execution_provider"]
    ep in ("cpu", "cuda") || error("registration.execution_provider = $ep is not \"cpu\" or \"cuda\"")
    ep == "cuda" && !isdefined(@__MODULE__, :cuDNN) &&
        error("registration.execution_provider = \"cuda\" needs cuDNN in the environment (ONNX Runtime's CUDA provider)")
    t0 = time()
    reg = register_rotations(inp.frames, lens; raw_scale = 2, tone_percentile = r["tone_percentile"], detector_margin_px = r["detector_margin_px"],
                             asift_tilts = r["asift_tilts"] == "imas25" ? nothing : [Tuple(t) for t in r["asift_tilts"]],
                             max_detections = r["max_detections"], descriptor = Symbol(r["descriptor"]), matcher = Symbol(r["matcher"]),
                             min_score = r["min_score"], pairs = Symbol(r["pairs"]), max_lag = r["max_lag"], overlap_deg = r["overlap_deg"],
                             min_inliers = r["min_inliers"], seed = r["seed"],
                             scoring = VgtkCore.MsacScoring(VgtkEstimation.RotationProblem; threshold = Float64(r["scoring_threshold_px"])),
                             max_trials = r["max_trials"], min_trials = r["min_trials"], averaging_loss = LOSSES[r["averaging_loss"]],
                             bundle_loss = LOSSES[r["bundle_loss"]], bundle_scale_multiple = r["bundle_scale_multiple"],
                             bundle_iterations = r["bundle_iterations"], bundle_max_iter = r["bundle_max_iter"],
                             bundle_ftol = r["bundle_ftol"], bundle_xtol = r["bundle_xtol"], execution_provider = Symbol(ep),
                             say = m -> say(log, m))
    secs = time() - t0
    Rs = [SMatrix{3, 3, Float64, 9}(R) for R in reg.rotations]
    rec = Dict("key" => key, "convention" => "d = R_k·f_k, the first frame's camera frame the common frame; R row by row",
               "frames" => inp.names, "rotations" => [[collect(R[i, :]) for i in 1:3] for R in Rs],
               "pairs" => length(reg.pairs), "bundle_rms_px" => reg.bundle.rms, "seconds" => secs,
               # the bundle's outcome (RotationBundleMetadata): converged, nonfinite_start (its start not finite: a start prediction
               # outside the lens's domain), not_run, or another retcode; the observations not finite at the start
               "bundle_retcode" => string(reg.bundle.retcode), "bundle_outcome" => string(reg.bundle.metadata.outcome),
               "bundle_start_nonfinite" => reg.bundle.metadata.start_nonfinite,
               # the bundle's robust loss width (camera px) and its scale multiple: the residual core scale, the sensor point's σ of
               # IntensityAndPositionNoise
               "bundle_loss_width_px" => reg.bundle.metadata.loss_width, "bundle_scale_multiple" => Float64(r["bundle_scale_multiple"]))
    open(io -> JSON.print(io, rec, 2), path, "w")
    return (; rotations = Rs, pairs = length(reg.pairs), bundle_rms_px = reg.bundle.rms, seconds = secs, bundle_retcode = rec["bundle_retcode"],
            bundle_outcome = rec["bundle_outcome"], bundle_start_nonfinite = rec["bundle_start_nonfinite"], provenance = nothing,
            point_scale_px = reg.bundle.metadata.loss_width / r["bundle_scale_multiple"])
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
                                        "scale (its record rotations.json holds no bundle_loss_width_px: delete it to recompute the registration)")
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
                          saturated = dev.saturated, raw_scale = 2,
                          rim_margin_px = d["rim_margin_px"], fov_deg = d["fov_deg"], plane = d["plane"], plane_offsets = dev.plane_offsets,
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
function coverage_extent(lens, frames, rotations, up; δ = 0.5)
    v = panorama_view(VgtkCore.Equirectangular(), (-180.0, 180.0, -89.0, 89.0), δ; up)
    r = render_panorama(lens, frames, rotations, v; vignetting = no_vignetting, raw_scale = 2,
                        rim_margin_px = 0.0, fov_deg = 89.0)
    seen = findall(>(0), Array(r.coverage))
    rows = extrema(I -> I[1], seen); cols = extrema(I -> I[2], seen)
    return (-180 + δ * (cols[1] - 1), -180 + δ * cols[2], 89 - δ * rows[2], 89 - δ * (rows[1] - 1))
end

srgb(x) = x <= 0.0031308f0 ? 12.92f0 * x : 1.055f0 * x^(1 / 2.4f0) - 0.055f0
"An H × W × 3 linear image as 8-bit sRGB at a fixed exposure, each plane times its white-balance factor (NaN, no frame: black)."
function save_srgb(path, img, exposure, wb)
    q = clamp.(ifelse.(isnan.(img), 0.0f0, Float32.(img) .* (Float32(exposure) .* reshape(Float32.(wb), 1, 1, 3))), 0.0f0, 1.0f0)
    n(x) = reinterpret(FPN.N0f8, round(UInt8, 255 * srgb(x)))
    PNGFiles.save(path, CT.RGB{FPN.N0f8}.(n.(view(q, :, :, 1)), n.(view(q, :, :, 2)), n.(view(q, :, :, 3))))
end
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
        open(joinpath(C.out, stem * "_profile.csv"), "w") do io
            println(io, "rho_px,V")
            for ρ in 0.0:4.0:r
                println(io, @sprintf("%.1f,%.6f", ρ, v(c + SVector(ρ, 0.0))))
            end
        end
    else
        rec["model"] = Dict("kind" => VgtkCore.tag_of(VgtkCore.VIGNETTING_TAGS, VgtkCore.TensorProductVignetting), "controls" => collect(size(VgtkCore.control_points(v))),
                            "spline" => "VisualGeometryToolkitCore TensorProductVignetting: a tensor-product B-spline on clamped knots, x fastest")
    end
    open(io -> JSON.print(io, rec, 2), joinpath(C.out, stem * ".json"), "w")
    # the map over the sensor, every 32 px (x, y, V), and as an image normalized to its maximum
    open(joinpath(C.out, stem * "_map.csv"), "w") do io
        println(io, "x_px,y_px,V")
        for y in 16.0:32.0:H, x in 16.0:32.0:W
            println(io, @sprintf("%.1f,%.1f,%.6f", x, y, v(SVector(x, y))))
        end
    end
    M = [v(SVector(2j - 1.0, 2i - 1.0)) for i in 1:(H ÷ 2), j in 1:(W ÷ 2)]
    PNGFiles.save(joinpath(C.out, stem * "_map.png"), CT.Gray{FPN.N0f8}.(clamp.(M ./ maximum(M), 0, 1)))
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
    log = open(joinpath(C.out, "run.log"), "w")
    T = Dict{String, Float64}()
    say(log, "panorama_vignetting: $(C.cfg["name"]) ($(C.path)), output $(C.out)")
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
    dev = (; frames = move.(inp.frames), saturated = move.(inp.saturated), inp.plane_offsets, inp.levels)
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
    ext = sc["extent"] == "coverage" ? coverage_extent(lens, dev.frames, reg.rotations, up) : Tuple(sc["extent"])
    sview = panorama_view(VgtkCore.Equirectangular(), ext, sc["deg_per_pixel"]; up)
    skw = (; saturated = dev.saturated, raw_scale = 2, rim_margin_px = d["rim_margin_px"], fov_deg = d["fov_deg"],
           plane = d["plane"], plane_offsets = dev.plane_offsets, chunk = sc["chunk"])
    t = time()
    before = score_seams(lens, dev.frames, reg.rotations, sview; vignetting = no_vignetting, skw...)
    after = score_seams(lens, dev.frames, reg.rotations, sview; vignetting = VgtkCore.estimate(res), gains = fit_gains(C, res), skw...)
    T["score_seams (two)"] = time() - t
    save_ratio(joinpath(C.out, "seams_uncorrected.png"), Array(before.ratio)); save_ratio(joinpath(C.out, "seams_corrected.png"), Array(after.ratio))
    say(log, "seams (saturated pixels excluded): uncorrected rms $(round(before.rms; digits = 4)) MAD $(round(before.mad; digits = 4)); " *
             "corrected rms $(round(after.rms; digits = 4)) MAD $(round(after.mad; digits = 4)); $(after.n) pixels")

    # the renders: the real images (no saturation mask), corrected and uncorrected at one exposure
    rc = C.cfg["render"]
    rext = rc["extent"] == "coverage" ? ext : Tuple(rc["extent"])
    rkw = (; raw_scale = 2, rim_margin_px = d["rim_margin_px"], fov_deg = d["fov_deg"], plane_offsets = dev.plane_offsets)
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
    save(name, img) = save_srgb(joinpath(C.out, name * ".png"), img, exposure, wb)
    save("cylindrical_corrected", cyl_c); save("cylindrical_uncorrected", cyl_u)
    cyl_c = cyl_u = nothing
    if occursin("14mm", C.cfg["name"])
        open(joinpath(C.out, "READY_14mm.txt"), "w") do io
            println(io, "cylindrical_corrected:   ", joinpath(C.out, "cylindrical_corrected.png"))
            println(io, "cylindrical_uncorrected: ", joinpath(C.out, "cylindrical_uncorrected.png"))
            println(io, "exposure (both): $(round(exposure; digits = 4)) (1 / the $(rc["exposure_percentile"]) % G of the corrected); " *
                        "white balance $(rc["white_balance"]) $(round.(wb; digits = 3))")
            println(io, "seam spread, log, saturated pixels excluded: uncorrected rms $(round(before.rms; digits = 4)) " *
                        "1.4826·MAD $(round(before.mad; digits = 4)); corrected rms $(round(after.rms; digits = 4)) " *
                        "1.4826·MAD $(round(after.mad; digits = 4)) ($(after.n) pixels seen by two frames or more)")
        end
    end
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
    open(joinpath(C.out, "summary.md"), "w") do io
        println(io, "# $(C.cfg["name"]): panorama vignetting\n")
        println(io, "Config of record: `$(basename(C.path))`. Data root `\$VGTK_DATA_ROOT`; RAW `$(C.cfg["inputs"]["raw_dir"])`, lens `$(C.cfg["inputs"]["lens"])`.\n")
        println(io, "- Frames: $(length(inp.frames)), decoded $(C.cfg["decode"]["mode"]) gain $(C.cfg["decode"]["gain"]) → $(W ÷ 2) × $(H ÷ 2); saturated quads (all frames): $(sum(count, inp.saturated)).")
        bund = reg.bundle_rms_px === nothing ? "not recorded" : "$(f(reg.bundle_rms_px; d = 3)) px"
        println(io, "- Registration: $(reg.pairs) verified pairs, bundle rms $bund$(reg.provenance === nothing ? "" : "; $(reg.provenance)"); " *
                    "bundle outcome $(reg.bundle_outcome) (retcode $(reg.bundle_retcode), $(reg.bundle_start_nonfinite) observations not finite at the start).")
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
        for x in sort(readdir(C.out))
            isfile(joinpath(C.out, x)) && println(io, "- `$x`")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: julia --project=<env> run.jl <config.toml>")
    run_example(ARGS[1])
end
