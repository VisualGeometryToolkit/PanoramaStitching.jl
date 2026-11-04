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
using PanoramaStitching
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
import VisualGeometryToolkitFeatures as VgtkFeatures
import VisualGeometryToolkitIO as VgtkIO
using StaticArrays: SVector, SMatrix, SA
using LinearAlgebra: LinearAlgebra, norm, normalize, dot, cross, eigen, Symmetric
using Statistics: mean, median, quantile
using Printf: @sprintf
import PNGFiles, JSON, SHA, TOML

const FPN = PNGFiles.FixedPointNumbers
const CT = PNGFiles.ColorTypes

# ----------------------------------------------------------------------------- configuration

data_root() = get(ENV, "VGTK_DATA_ROOT", "") |> r -> isempty(r) ?
    error("VGTK_DATA_ROOT is not set: bind it to the data root that holds the config's inputs (README, \"Inputs\")") : r
input_path(rel) = joinpath(data_root(), rel)

function load_config(path)
    cfg = TOML.parsefile(path)
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

sha256_file(path) = open(io -> bytes2hex(SHA.sha256(io)), path)

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
        h = sha256_file(src)
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
    return (; names = raws, frames, saturated = sats, plane_offsets = [SVector{2, Float64}(o...) for o in offsets], daylight_wb = wb)
end

# ----------------------------------------------------------------------------- the lens

"The camera (back-projection: the fit, the registration's bearings), its forward map (the render, the sampling) and the region."
function load_lens(C; raw_width, raw_height)
    inp = C.cfg["inputs"]; path = input_path(inp["lens"])
    if inp["lens_kind"] == "cameras"
        camera = VgtkCore.model(VgtkIO.load(Dict{Symbol, VgtkCore.BoundedCamera}, path)[Symbol(inp["lens_camera"])])
        forward = camera
    elseif inp["lens_kind"] == "grid"
        camera = VgtkIO.load(VgtkCore.CentralGenericModel, path)
        c = inp["circle"]; region = VgtkCore.Rect(0, 0, raw_width, raw_height) ∩ VgtkCore.Circle(VgtkCore.Point2(c[1], c[2]), c[3])
        forward = VgtkCore.estimate(VgtkEstimation.fit_camera(VgtkCore.SplineProjection, camera, region;
                                                               tolerance = Float64(inp["spline_tolerance_px"])))
    else
        error("inputs.lens_kind = $(inp["lens_kind"]) is not \"cameras\" or \"grid\"")
    end
    c = inp["circle"]
    circle = c == "none" ? VgtkCore.WholePlane() : VgtkCore.Circle(VgtkCore.Point2(Float64(c[1]), Float64(c[2])), Float64(c[3]))
    return (; camera, forward, circle)
end

# ----------------------------------------------------------------------------- registration (cached by its settings)

const LOSSES = Dict("huber" => VgtkCore.HuberShape(), "cauchy" => VgtkCore.CauchyShape())

# The registration's cache key: its settings, the inputs, the decode settings and the RAW files' hashes.
function registration_key(C, inp)
    canon(d) = join(["$k=$(d[k])" for k in sort(collect(keys(d)))], ";")
    return bytes2hex(SHA.sha256(join([canon(C.cfg["registration"]), canon(C.cfg["inputs"]), canon(C.cfg["decode"]),
                                      (sha256_file(joinpath(input_path(C.cfg["inputs"]["raw_dir"]), n)) for n in inp.names)...], "|")))
end

function registration(C, inp, lens; log)
    r = C.cfg["registration"]
    key = registration_key(C, inp)
    path = joinpath(C.out, "rotations.json")
    if isfile(path)
        rec = JSON.parsefile(path)
        if rec["key"] == key
            say(log, "registration: reused $(path)$(haskey(rec, "provenance") ? " ($(rec["provenance"]))" : "")")
            return (; rotations = [SMatrix{3, 3, Float64, 9}(reduce(vcat, permutedims.(Vector{Float64}.(R)))) for R in rec["rotations"]],
                    pairs = rec["pairs"], bundle_rms_px = rec["bundle_rms_px"], seconds = something(rec["seconds"], 0.0),
                    provenance = get(rec, "provenance", nothing))
        end
    end
    t0 = time()
    reg = register_rotations(inp.frames, lens.camera; projection = lens.forward, raw_scale = 2, circle = lens.circle,
                             tone_percentile = r["tone_percentile"], detector_margin_px = r["detector_margin_px"],
                             asift_tilts = r["asift_tilts"] == "imas25" ? nothing : [Tuple(t) for t in r["asift_tilts"]],
                             max_detections = r["max_detections"], descriptor = Symbol(r["descriptor"]), matcher = Symbol(r["matcher"]),
                             min_score = r["min_score"], pairs = Symbol(r["pairs"]), max_lag = r["max_lag"], overlap_deg = r["overlap_deg"],
                             min_inliers = r["min_inliers"], seed = r["seed"],
                             scoring = VgtkCore.MsacScoring(VgtkEstimation.RotationProblem; threshold = Float64(r["scoring_threshold_px"])),
                             max_trials = r["max_trials"], min_trials = r["min_trials"], averaging_loss = LOSSES[r["averaging_loss"]],
                             bundle_loss = LOSSES[r["bundle_loss"]], bundle_scale_multiple = r["bundle_scale_multiple"],
                             bundle_iterations = r["bundle_iterations"], bundle_max_iter = r["bundle_max_iter"],
                             bundle_ftol = r["bundle_ftol"], bundle_xtol = r["bundle_xtol"], say = m -> say(log, m))
    secs = time() - t0
    Rs = [SMatrix{3, 3, Float64, 9}(R) for R in reg.rotations]
    rec = Dict("key" => key, "convention" => "d = R_k·f_k, the first frame's camera frame the common frame; R row by row",
               "frames" => inp.names, "rotations" => [[collect(R[i, :]) for i in 1:3] for R in Rs],
               "pairs" => length(reg.pairs), "bundle_rms_px" => reg.bundle.rms, "seconds" => secs)
    open(io -> JSON.print(io, rec, 2), path, "w")
    return (; rotations = Rs, pairs = length(reg.pairs), bundle_rms_px = reg.bundle.rms, seconds = secs, provenance = nothing)
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

# The principal point: the pixel of the optical axis (0, 0, 1) through the forward map.
principal_point(lens) = (u = VgtkCore.pointwise!(lens.forward isa VgtkCore.CameraModel ? VgtkCore.imaging_transform(lens.forward) : lens.forward,
                                                 [SVector(0.0, 0.0)], [SVector(0.0, 0.0, 1.0)]); u[1])

# The domain's extreme points: a circle less the rim margin inside the sensor, or the sensor less its 2 px border.
function domain_box(lens, W, H, rim)
    c = lens.circle
    if c isa VgtkCore.WholePlane
        return SVector(2.0, 2.0), SVector(W - 4.0, H - 4.0)
    end
    r = c.r - rim; lo = max.(SVector(2.0, 2.0), SVector(c.center[1] - r, c.center[2] - r))
    hi = min.(SVector(W - 2.0, H - 2.0), SVector(c.center[1] + r, c.center[2] + r))
    return lo, hi - lo
end
function domain_radius(lens, center, W, H, rim)
    c = lens.circle
    c isa VgtkCore.WholePlane && return maximum(norm(SVector(x, y) - center) for x in (2.0, W - 2.0), y in (2.0, H - 2.0))
    return norm(SVector(c.center[1], c.center[2]) - center) + c.r - rim
end

function vignetting_space(kind, C, lens, W, H)
    v = C.cfg["vignetting"]; rim = Float64(C.cfg["domain"]["rim_margin_px"])
    if kind == "radial"
        rc = v["radial_center"]
        center = rc == "principal_point" ? SVector{2, Float64}(principal_point(lens)) :
                 rc == "circle_center" ? SVector{2, Float64}(lens.circle.center[1], lens.circle.center[2]) : SVector{2, Float64}(rc...)
        radius = v["radial_radius"] == "domain" ? domain_radius(lens, center, W, H, rim) + 1.0 : Float64(v["radial_radius"])
        return RadialVignetting(center, radius, v["radial_controls"])
    elseif kind == "tensor"
        v["tensor_box"] == "domain" || error("vignetting.tensor_box = $(v["tensor_box"]) is not \"domain\"")
        o, e = domain_box(lens, W, H, rim)
        return TensorVignetting(o, e, Tuple(v["tensor_controls"]))
    end
    error("vignetting model \"$kind\" is not \"radial\" or \"tensor\"")
end

function fit(kind, C, lens, dev, rotations, W, H)
    v = C.cfg["vignetting"]; d = C.cfg["domain"]
    space = vignetting_space(kind, C, lens, W, H)
    s = v["solver"]
    solver = s == "cg" ? ConjugateGradient(; tolerance = v["cg_tolerance"], max_iter = v["cg_max_iter"]) :
             s == "cholesky" ? SparseCholesky() : error("vignetting.solver = $s is not \"cg\" or \"cholesky\"")
    return fit_vignetting(space, lens.camera, dev.frames, rotations; saturated = dev.saturated, circle = lens.circle, raw_scale = 2,
                          rim_margin_px = d["rim_margin_px"], fov_deg = d["fov_deg"], plane = d["plane"], plane_offsets = dev.plane_offsets,
                          chart_spacing_deg = v["chart_spacing_deg"], penalty_order = v["penalty_order"], penalty_weight = v["penalty_weight"],
                          vignetting_penalty_weight = v["vignetting_penalty_weight"], frame_gains = v["frame_gains"], solver)
end
# The frames' exposure gains of a fit, for the seam score and the renders: its metadata's when it fitted them, else none.
fit_gains(C, res) = C.cfg["vignetting"]["frame_gains"] ? res.metadata.gains : nothing

# ----------------------------------------------------------------------------- views, renders, images

struct NoVignetting end
(::NoVignetting)(u, f) = 1.0

"The covered (lon, lat) box in the up frame: a coarse equirectangular render's seen pixels."
function coverage_extent(lens, frames, rotations, up; δ = 0.5)
    v = panorama_view(VgtkCore.Equirectangular(), (-180.0, 180.0, -89.0, 89.0), δ; up)
    r = render_panorama(lens.forward, frames, rotations, v; vignetting = NoVignetting(), circle = lens.circle, raw_scale = 2,
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
    v = VgtkCore.estimate(res); s = v.space
    rec = Dict{String, Any}("gauge" => "mean log V = 0 over the fit domain's pixels", "coefficients_of" => "log V",
                            "coefficients" => collect(v.coefficients), "pixel_frame" => "camera pixels, COLMAP (full-resolution sensor)")
    if s isa RadialVignetting
        rec["space"] = Dict("kind" => "radial", "center" => collect(s.center), "radius" => s.radius, "controls" => s.n,
                            "spline" => "uniform cubic B-spline in ρ on [0, radius]")
        open(joinpath(C.out, stem * "_profile.csv"), "w") do io
            println(io, "rho_px,V")
            for ρ in 0.0:4.0:s.radius
                println(io, @sprintf("%.1f,%.6f", ρ, v(s.center + SVector(ρ, 0.0), nothing)))
            end
        end
    else
        rec["space"] = Dict("kind" => "tensor", "origin" => collect(s.origin), "extent" => collect(s.extent), "controls" => collect(s.n),
                            "spline" => "uniform cubic tensor B-spline, x fastest")
    end
    open(io -> JSON.print(io, rec, 2), joinpath(C.out, stem * ".json"), "w")
    # the map over the sensor, every 32 px (x, y, V), and as an image normalized to its maximum
    open(joinpath(C.out, stem * "_map.csv"), "w") do io
        println(io, "x_px,y_px,V")
        for y in 16.0:32.0:H, x in 16.0:32.0:W
            println(io, @sprintf("%.1f,%.1f,%.6f", x, y, v(SVector(x, y), nothing)))
        end
    end
    M = [v(SVector(2j - 1.0, 2i - 1.0), nothing) for i in 1:(H ÷ 2), j in 1:(W ÷ 2)]
    PNGFiles.save(joinpath(C.out, stem * "_map.png"), CT.Gray{FPN.N0f8}.(clamp.(M ./ maximum(M), 0, 1)))
    return M
end

# V's summary on the domain: its range, where its maximum is, and its largest antipodal asymmetry about `center`.
function vignetting_summary(v, lens, center, W, H, rim, fov_mask)
    pts = [SVector(2j - 1.0, 2i - 1.0) for i in 1:2:(H ÷ 2), j in 1:2:(W ÷ 2)]
    pts = filter(fov_mask, vec(pts))
    lv = [log_vignetting(v, u) for u in pts]
    imax = argmax(lv)
    ρ = maximum(norm(u - center) for u in pts)
    asym = 0.0
    for r in range(0, 0.95ρ, length = 40), φ in range(0, π, length = 73)[1:(end - 1)]
        a = center + r * SVector(cos(φ), sin(φ)); b = center - r * SVector(cos(φ), sin(φ))
        (fov_mask(a) && fov_mask(b)) || continue
        asym = max(asym, abs(exp(log_vignetting(v, a) - log_vignetting(v, b)) - 1))
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
    dev = (; frames = move.(inp.frames), saturated = move.(inp.saturated), inp.plane_offsets)
    # A grid lens's SplineProjection cannot yet enter Core's image_warp on a device (Core has no Adapt rule for the BoundedMap
    # and ComposedTransformation that wrap it in the warp's map): its renders run on the host, the same code. The fit and the
    # seam score (sample_frames: the forward map alone, moved to the device) stay on the device.
    host_render = device == "cuda" && lens.forward isa VgtkCore.SplineProjection
    host_render && say(log, "renders: on the host (a SplineProjection in Core's device image_warp needs Core's Adapt rules)")
    rframes = host_render ? inp.frames : dev.frames
    kind = C.cfg["vignetting"]["model"]
    fit(kind, C, lens, dev, reg.rotations, W, H)                                  # compile
    t = time(); res = fit(kind, C, lens, dev, reg.rotations, W, H); T["fit_vignetting ($kind, warm)"] = time() - t
    m = res.metadata
    say(log, "fit_vignetting ($kind): $(res.retcode), $(res.iterations) iterations, weighted rms $(round(res.rms; digits = 4)), " *
             "MAD $(round(m.residual_mad; digits = 4)), $(res.n) rows, $(m.n_pixels) pixels, chart $(m.chart.n)², $(m.n_active) active")
    say(log, "  stages (s): $(m.seconds)")
    C.cfg["vignetting"]["frame_gains"] && say(log, "  frame gains g_k (g₁ = 1): $(join(round.(m.gains; digits = 4), ", "))")
    M = write_vignetting(C, "vignetting", res, W, H)

    # the seam scores, before and after
    sc = C.cfg["seams"]; d = C.cfg["domain"]
    ext = sc["extent"] == "coverage" ? coverage_extent(lens, rframes, reg.rotations, up) : Tuple(sc["extent"])
    sview = panorama_view(VgtkCore.Equirectangular(), ext, sc["deg_per_pixel"]; up)
    skw = (; saturated = dev.saturated, circle = lens.circle, raw_scale = 2, rim_margin_px = d["rim_margin_px"], fov_deg = d["fov_deg"],
           plane = d["plane"], plane_offsets = dev.plane_offsets, chunk = sc["chunk"])
    t = time()
    before = score_seams(lens.forward, dev.frames, reg.rotations, sview; vignetting = NoVignetting(), skw...)
    after = score_seams(lens.forward, dev.frames, reg.rotations, sview; vignetting = VgtkCore.estimate(res), gains = fit_gains(C, res), skw...)
    T["score_seams (two)"] = time() - t
    save_ratio(joinpath(C.out, "seams_uncorrected.png"), Array(before.ratio)); save_ratio(joinpath(C.out, "seams_corrected.png"), Array(after.ratio))
    say(log, "seams (saturated pixels excluded): uncorrected rms $(round(before.rms; digits = 4)) MAD $(round(before.mad; digits = 4)); " *
             "corrected rms $(round(after.rms; digits = 4)) MAD $(round(after.mad; digits = 4)); $(after.n) pixels")

    # the renders: the real images (no saturation mask), corrected and uncorrected at one exposure
    rc = C.cfg["render"]
    rext = rc["extent"] == "coverage" ? ext : Tuple(rc["extent"])
    rkw = (; circle = lens.circle, raw_scale = 2, rim_margin_px = d["rim_margin_px"], fov_deg = d["fov_deg"], plane_offsets = dev.plane_offsets)
    V = VgtkCore.estimate(res)
    # the corrected renders divide by the frames' gains as well; the uncorrected ones show the frames as they are
    render(view, vig) = Array(render_panorama(lens.forward, rframes, reg.rotations, view; vignetting = vig,
                                              gains = vig isa NoVignetting ? nothing : fit_gains(C, res), rkw...).image)
    t = time()
    ℓ = Float64(rc["cylindrical_max_lat_deg"])                 # tan φ grows without bound: the cylindrical's latitude cap
    cyl = panorama_view(VgtkCore.Cylindrical(), (rext[1], rext[2], max(rext[3], -ℓ), min(rext[4], ℓ)), rc["cylindrical_deg_per_pixel"]; up)
    cyl_c = render(cyl, V); cyl_u = render(cyl, NoVignetting())
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
    save("equal_area_corrected", render(ea, V)); save("equal_area_uncorrected", render(ea, NoVignetting()))
    rw, rh = rc["rectilinear_size"]; f = rw / 2 / tand(rc["rectilinear_hfov_deg"] / 2)
    rcam = VgtkCore.CameraModel(f * VgtkCore.px, SA[rw / 2 * VgtkCore.px, rh / 2 * VgtkCore.px]; lens = VgtkCore.Lens(VgtkCore.Perspective()))
    rect = panorama_view(rcam, rw, rh; rotation = up)
    save("rectilinear_corrected", render(rect, V)); save("rectilinear_uncorrected", render(rect, NoVignetting()))
    T["renders (six)"] = time() - t

    # the comparison fits (not used for the renders)
    rim = Float64(d["rim_margin_px"])
    in_domain(u) = (lens.circle isa VgtkCore.WholePlane || norm(SVector(u[1] - lens.circle.center[1], u[2] - lens.circle.center[2])) < lens.circle.r - rim) &&
                   2 <= u[1] <= W - 2 && 2 <= u[2] <= H - 2
    pp = SVector{2, Float64}(principal_point(lens))
    rows = Any[(kind, res, after, vignetting_summary(V, lens, pp, W, H, rim, in_domain))]
    for k in C.cfg["vignetting"]["compare_models"]
        t = time(); r2 = fit(k, C, lens, dev, reg.rotations, W, H); T["fit_vignetting ($k, compare)"] = time() - t
        s2 = score_seams(lens.forward, dev.frames, reg.rotations, sview; vignetting = VgtkCore.estimate(r2), gains = fit_gains(C, r2), skw...)
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
        println(io, "- Registration: $(reg.pairs) verified pairs, bundle rms $bund$(reg.provenance === nothing ? "" : "; $(reg.provenance)").")
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
        println(io, "\nFrame gains (vignetting.frame_gains = $(C.cfg["vignetting"]["frame_gains"])): " *
                    (C.cfg["vignetting"]["frame_gains"] ? "g_k of the fit of record (g₁ = 1) $(join(f.(rows[1][2].metadata.gains), ", ")); the seam score and the corrected renders divide them out." :
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
