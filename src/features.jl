# =============================================================================
# Features: per frame, the keypoints and their descriptions; per frame pair, the matches
# =============================================================================
#
# VisualGeometryToolkitFeatures' ASIFT keypoints (affine frames with their covariances) on
# the tone-mapped green channel inside the image circle, the strongest by |response|,
# described by DeDoDe at them (`LoMaDescriptor`, which resamples the frame to its network's
# fixed input), and matched by the LoMa matcher (`LoMaMatcher`, LightGlue architecture).
# =============================================================================

"""
    FrameFeatures

One frame's features: `pixels` (the kept detections' centres in the camera's
pixel frame, full resolution), `pixel_covariances` (each centre's 2 × 2
position covariance in camera pixels², VisualGeometryToolkitFeatures'
`FeatureCovariance` for unit iid noise of the tone-mapped detection image,
scaled as the centres are; all `NaN` for a detection the detector did not
refine), `features` (their DeDoDe descriptions), `n_detected` (ASIFT's count
before the circle and the strength selection).
"""
struct FrameFeatures{F}
    pixels::Vector{SVector{2, Float64}}
    pixel_covariances::Vector{SMatrix{2, 2, Float64, 4}}
    features::F
    n_detected::Int
end

"""
    detect_features(frame, descriptor; raw_scale, circle, tone_percentile, detector_margin_px, asift_tilts, max_detections)
        -> FrameFeatures

The features of one frame (`H × W × 3` linear, `raw_scale` camera pixels per frame
pixel): ASIFT on `sqrt(G / p)` clamped to `[0, 1]` (`p` the `tone_percentile`-th
percentile of the green channel inside the image `circle` — a Core `Circle` or
`Ellipse` in camera pixels, `WholePlane()` for a lens with none), the detections
inside the circle less `detector_margin_px`, the
`max_detections` strongest by |response| (VisualGeometryToolkitFeatures'
`strongest`), described by `descriptor` (a `LoMaDescriptor`) on the frame's
tone-mapped RGB, the detections in the frame's pixels (the descriptor resamples
the image to its network's input).
`asift_tilts` is VisualGeometryToolkitFeatures' tilt schedule (`nothing`: IMAS-25).
"""
function detect_features(frame::AbstractArray{<:Real, 3}, descriptor; raw_scale::Integer, circle, tone_percentile::Real,
                         detector_margin_px::Real, asift_tilts, max_detections::Integer)
    Hf, Wf = size(frame, 1), size(frame, 2); s = raw_scale
    inside = VgtkCore.pixel_mask(_binned(circle, s), Wf, Hf)                  # the circle in the binned frame's pixels
    p = quantile(vec(view(frame, :, :, 2))[vec(inside)], tone_percentile / 100)
    toned = sqrt.(clamp.(Float64.(frame) ./ p, 0, 1))
    dets = VgtkFeatures.asift_detect(Float32.(view(toned, :, :, 2)); tilts = asift_tilts, uncertainty = VgtkFeatures.FeatureCovariance())
    n = length(dets)
    kept = _rim(circle, detector_margin_px)
    dets = VgtkFeatures.strongest(filter(d -> VgtkCore.Point2(s * d.center[1], s * d.center[2]) ∈ kept, dets), max_detections)
    pixels = [SVector(s * d.center[1], s * d.center[2]) for d in dets]
    # the centres' position covariance (the top-left 2 × 2 of the (x, y, log σ) covariance, detection-image pixels²), in camera
    # pixels² as the centres are scaled: s²·Σ
    covs = [(Σ = Statistics.cov(d); s^2 * SMatrix{2, 2, Float64, 4}(Σ[1, 1], Σ[2, 1], Σ[1, 2], Σ[2, 2])) for d in dets]
    # described on the toned RGB as a 3 × H × W channelview, the detections in the frame's own pixels (the descriptor resamples
    # the image to its network's input)
    return FrameFeatures(pixels, covs, descriptor(permutedims(toned, (3, 1, 2)), dets), n)
end

"""
    pair_bearings(camera, fa, fb, matches) -> (cs, kept)

The matches of two frames as unit-bearing correspondences with their
back-projection Jacobians as the `M₁`/`M₂` columns and the match score as the
`:score` column (VisualGeometryToolkitCore's `backproject_with_jacobian` through
the calibrated `camera`: the whole field, past 90°), and the keypoint index pairs
kept (a pixel the camera cannot back-project is dropped).
"""
function pair_bearings(camera, fa::FrameFeatures, fb::FrameFeatures, matches)
    f1 = SVector{3, Float64}[]; f2 = SVector{3, Float64}[]; M1 = SMatrix{3, 2, Float64, 6}[]; M2 = SMatrix{3, 2, Float64, 6}[]
    score = Float64[]; kept = Tuple{Int, Int}[]
    for m in matches
        a, b = m.first, m.second
        ba = VgtkCore.backproject_with_jacobian(camera, fa.pixels[a]); bb = VgtkCore.backproject_with_jacobian(camera, fb.pixels[b])
        (all(isfinite, ba[1]) && all(isfinite, bb[1]) && all(isfinite, ba[2]) && all(isfinite, bb[2])) || continue
        push!(f1, ba[1]); push!(f2, bb[1]); push!(M1, ba[2]); push!(M2, bb[2]); push!(score, m.score); push!(kept, (a, b))
    end
    return VgtkCore.withattrs(VgtkCore.Csponds(f1, f2); M₁ = M1, M₂ = M2, score), kept
end
