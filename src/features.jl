# =============================================================================
# Features: per frame, the detections and their descriptions; per frame pair, the matches
# =============================================================================
#
# VisualGeometryToolkitFeatures' ASIFT detections (affine frames with their covariances) on
# the tone-mapped or the variance-stabilized green inside the camera's imaged region, about N of the strongest spread over the
# frame (VisualGeometryToolkitFeatures' `spread_strongest`, SSC), described by DeDoDe at them (`LoMaDescriptor`, which resamples
# the frame to its network's fixed input), and matched by the LoMa matcher (`LoMaMatcher`, LightGlue architecture).
# =============================================================================

# The detector's variants (`detect_features`' `selection` and `detection_image`).
const SELECTIONS = (:spread,)
const DETECTION_IMAGES = (:tone, :vst)

# The detector's settings, checked before any feature is detected: with `detection_image = :vst` the threshold `noise_sigmas`
# and, per frame, its sensor model (VisualGeometryToolkitCore's `SensorModel`, its `sensor_model` resolver; `nframes` of them,
# or the one frame's), whose noise must carry a gain (`_check_vst_model`); with `:tone` neither is read, so neither is given (no
# inert setting: a model other than the absent `SensorModel()` is refused).
function _check_detector_settings(what, selection, detection_image, models, noise_sigmas; nframes = nothing)
    selection in SELECTIONS || throw(ArgumentError("$what: selection = :$selection is not :spread (Features' spread_strongest, SSC)"))
    detection_image in DETECTION_IMAGES || throw(ArgumentError("$what: detection_image = :$detection_image is not :tone or :vst"))
    if detection_image === :vst
        nframes === nothing || length(models) == nframes ||
            throw(DimensionMismatch("$what: $nframes frames and $(length(models)) sensor models"))
        foreach(m -> _check_vst_model(what, m), models)
        (noise_sigmas isa Real && isfinite(noise_sigmas) && noise_sigmas > 0) ||
            throw(ArgumentError("$what: detection_image = :vst needs noise_sigmas, the threshold in noise σ (positive), got $noise_sigmas"))
    else
        (all(m -> isequal(m, VgtkCore.SensorModel()), models) && noise_sigmas === nothing) ||
            throw(ArgumentError("$what: sensor models and noise_sigmas are read only by detection_image = :vst, not :$detection_image; leave them out"))
    end
    return nothing
end
# `:vst` stabilises the green cells' Poisson–Gaussian noise: the model's noise must carry a gain (VisualGeometryToolkitCore's
# `PoissonGaussian`); a read-noise-only, relative or absent noise has none, refused naming the model's reason.
_check_vst_model(what, m::VgtkCore.SensorModel) = _check_vst_noise(what, m, m.noise)
_check_vst_noise(what, m, ::VgtkCore.PoissonGaussian) = nothing
_check_vst_noise(what, m, _) =
    throw(ArgumentError("$what: detection_image = :vst needs a gain: a PoissonGaussian characterisation (VisualGeometryToolkitCore's " *
                        "sensor_model), got a $(nameof(typeof(m.noise))) noise (the model's reason: $(m.reason))"))

# The detector's keyword of 0.6 (`(; electrons_per_dn, read_noise, noise_sigmas)`), refused naming its successors.
const _RETIRED_DETECTION_NOISE = "the frame's noise is its `sensor_model` (VisualGeometryToolkitCore's SensorModel, its sensor_model " *
                                 "resolver of the body's characterisation) and the threshold `noise_sigmas`"

# A frame's linear `H × W × 3` image: the frame itself, or a VisualGeometryToolkitCore `LinearRaw`'s (`:bin2`) image.
_frame_image(frame::AbstractArray{<:Real, 3}) = frame
_frame_image(frame::VgtkCore.LinearRaw) = first(_frame_values(frame.image))   # its values (an uncertain image's value plane)

# A frame's clip mask (`true` where a value is clipped: Features' detection `exclusion_mask`): the caller's, else a `LinearRaw`'s own
# saturation, else none.
_clip_mask(frame, saturated::AbstractMatrix{Bool}) = saturated
_clip_mask(frame::VgtkCore.LinearRaw, ::Nothing) = frame.saturated
_clip_mask(frame, ::Nothing) = nothing

# ASIFT on the detection image: the tone-mapped green (`toned`'s plane 2) at Features' default contrast threshold, or the
# variance-stabilized green cells of a `LinearRaw` (Features' `variance_stabilize`, whose saturation Features masks) at
# `noise_sigmas` noise σ; no detection reads a clipped pixel (Features' `exclusion_mask`: within three scales of one, dropped).
_asift(frame, toned, ::Val{:tone}, model, noise_sigmas, tilts, mask) =
    VgtkFeatures.asift_detect(Float32.(view(toned, :, :, 2)); tilts, exclusion_mask = mask)
_asift(frame::VgtkCore.LinearRaw, toned, ::Val{:vst}, model::VgtkCore.SensorModel, noise_sigmas, tilts, mask) =
    VgtkFeatures.asift_detect(VgtkFeatures.variance_stabilize(frame, model); noise_sigmas, tilts, exclusion_mask = mask)
_asift(frame, toned, ::Val{:vst}, model, noise_sigmas, tilts, mask) =
    throw(ArgumentError("detect_features: detection_image = :vst needs the frame's LinearRaw (its green cells' code spans), got a $(typeof(frame))"))

"""
    detect_features(camera::BoundedCamera, frame, descriptor; frame_map, tone_percentile, detector_margin_px, asift_tilts,
                    max_detections, selection = :spread, detection_image = :tone, sensor_model = SensorModel(), noise_sigmas = nothing,
                    saturated = nothing) -> StructArray{DescribedFeature}

The features of one frame (`H × W × 3` linear, or a VisualGeometryToolkitCore
`LinearRaw` of mode `:bin2`; `frame_map` its map onto the camera's pixels,
Core's `raw_frame_map(L)`) of
`camera`, described by `descriptor` (a `LoMaDescriptor`) on the frame's tone
`sqrt(G / p)` clamped to `[0, 1]` (`p` the `tone_percentile`-th percentile of the
green channel inside the camera's `imaged_region` — its sensor ∩ its lens
coverage ∩ its model's valid domain), the detections in the frame's pixels (the
descriptor resamples the image to its network's input).

- ASIFT (`asift_tilts`: a VisualGeometryToolkitFeatures `tilt_schedule` name,
  `:imas25` or `:untilted`, or its `(t, φ)` pairs) on the `detection_image`: `:tone`, that tone's green at Features'
  default contrast threshold; `:vst`, the frame's variance-stabilized green cells
  (Features' `variance_stabilize(frame, sensor_model)`, noise σ = 1 at every level; the frame
  must be its `LinearRaw`, decoded with the same model, `LinearRaw(mosaic; mode, sensor_model)`) at
  `noise_sigmas` noise σ, `sensor_model` the frame's VisualGeometryToolkitCore `SensorModel` (its
  `sensor_model` resolver of the body's characterisation at the frame's ISO), whose noise must
  carry a gain (a `PoissonGaussian`; any other is refused naming the model's reason). With `:tone`
  neither is given (the absent `SensorModel()`, the default); the 0.6 keyword `detection_noise`
  is refused naming them.
- No detection within three of its scales of a clipped pixel (Features'
  detection `exclusion_mask`): `saturated`, the frame's `H × W` saturation mask (a
  `LinearRaw`'s own by default; `nothing` for an array frame: none).
- The detections inside the imaged region less `detector_margin_px` at the lens
  coverage's edge (a camera whose coverage is its whole sensor: no margin).
- `selection`: `:spread` (the one selection), about `max_detections` of the
  strongest spread over the frame (Features' `spread_strongest`, SSC, which the
  descriptor applies).

The result is the descriptor's own: VisualGeometryToolkitFeatures' described
features (`StructArray{DescribedFeature}`), each a `Detection` in the frame's
pixels (the detector's default: no detection covariance) and its DeDoDe description —
the one feature store, in Features' canonical order of the detections (a function of
the image: the same run to run, on any number of threads). `frame_map` takes a
detection's centre to the camera's pixels ([`pair_bearings`](@ref), the bundle of
[`register_rotations`](@ref)).
"""
function detect_features(camera::VgtkCore.BoundedCamera, frame, descriptor; frame_map,
                         tone_percentile::Real, detector_margin_px::Real, asift_tilts, max_detections::Integer,
                         selection::Symbol = :spread, detection_image::Symbol = :tone,
                         sensor_model::VgtkCore.SensorModel = VgtkCore.SensorModel(), noise_sigmas::Union{Nothing, Real} = nothing,
                         saturated::Union{Nothing, AbstractMatrix{Bool}} = nothing, retired...)
    for k in keys(retired)
        k === :detection_noise && throw(ArgumentError("detect_features: the keyword `detection_noise` is retired: $(_RETIRED_DETECTION_NOISE)"))
        throw(ArgumentError("detect_features: unknown keyword `$k`"))
    end
    _check_detector_settings("detect_features", selection, detection_image, (sensor_model,), noise_sigmas)
    img = _frame_image(frame)
    Hf, Wf = size(img, 1), size(img, 2)
    region = VgtkCore.imaged_region(camera)
    inside = VgtkCore.pixel_mask(_in_frame(region, frame_map), Wf, Hf)        # the imaged region in the frame's pixels
    p = quantile(vec(view(img, :, :, 2))[vec(inside)], tone_percentile / 100)
    toned = sqrt.(clamp.(Float64.(img) ./ p, 0, 1))
    # the detections in Features' canonical order (a function of the image), so the kept set, its order and every
    # index-sampled RANSAC downstream are the same run to run
    dets = _asift(frame, toned, Val(detection_image), sensor_model, noise_sigmas, asift_tilts, _clip_mask(frame, saturated))
    kept = _imaged_less_rim(camera, detector_margin_px)
    at(d) = frame_map(SVector{2, Float64}(VgtkCore.center(VgtkCore.feature_frame(d))))         # a detection's centre in the camera's pixels
    dets = filter(d -> VgtkCore.Point2(at(d)...) ∈ kept, dets)
    # described on the toned RGB as a 3 × H × W channelview, the detections in the frame's own pixels (the descriptor resamples
    # the image to its network's input, and keeps about max_detections of them spread over it: the kept set is the described one)
    return descriptor(permutedims(toned, (3, 1, 2)), dets)
end

"""
    pair_bearings(camera, fa, fb, matches; frame_map) -> (cs, kept)

The `matches` of two frames' described features `fa`, `fb` ([`detect_features`](@ref))
as unit-bearing correspondences: VisualGeometryToolkitFeatures' `correspondences` of the
matched features (their `:score` column kept), each detection's centre taken to the
camera's pixels by the frames' `frame_map` and back-projected with its Jacobian as the
`M₁`/`M₂` columns (VisualGeometryToolkitCore's `bearing_correspondences` through the calibrated
`camera`: the whole field, past 90°), plain points, so the pair's residuals are in
camera pixels; and the detection index pairs kept (a pixel the camera cannot
back-project is dropped).
"""
function pair_bearings(camera, fa, fb, matches; frame_map)
    ms = VgtkCore.correspondences(fa, fb, matches)          # Features' method of Core's correspondences: the matched features and their score
    # each detection's centre in the camera's pixels, through the frame map
    ua = [frame_map(SVector{2, Float64}(VgtkCore.center(VgtkCore.feature_frame(f)))) for f in VgtkCore.sources(ms)]
    ub = [frame_map(SVector{2, Float64}(VgtkCore.center(VgtkCore.feature_frame(f)))) for f in VgtkCore.targets(ms)]
    cm = VgtkCore.model(camera)                       # the back-projection and its Jacobian are the model's
    bs = VgtkCore.bearing_correspondences(cm, cm, ua .=> ub)  # Core's: the bearings and their M₁/M₂
    f1 = VgtkCore.sources(bs); f2 = VgtkCore.targets(bs)
    keep = [i for i in eachindex(ua) if all(isfinite, f1[i]) && all(isfinite, f2[i]) && all(isfinite, bs.M₁[i]) && all(isfinite, bs.M₂[i])]
    cs = VgtkCore.withattrs(bs[keep]; score = ms.score[keep])
    return cs, [(matches[i].first, matches[i].second) for i in keep]
end
