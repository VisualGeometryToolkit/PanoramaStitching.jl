# =============================================================================
# Features: per frame, the keypoints and their descriptions; per frame pair, the matches
# =============================================================================
#
# VisualGeometryToolkitFeatures' ASIFT keypoints (affine frames with their covariances) on
# the tone-mapped green channel inside the image circle, the strongest by |response|,
# described by DeDoDe at them (`LoMaDescriptor`, on the frame resampled to the network's
# fixed 784 × 784 input by Core's `image_warp!`), and matched by the LoMa matcher
# (`LoMaMatcher`, LightGlue architecture).
# =============================================================================

"The DeDoDe/LoMa graphs' fixed input side (VisualGeometryToolkitFeatures' `LoMaDescriptor`)."
const LOMA_SIZE = 784

"""
    FrameFeatures

One frame's features: `pixels` (the kept detections' centres in the camera's
pixel frame, full resolution), `features` (their DeDoDe descriptions),
`n_detected` (ASIFT's count before the circle and the strength selection).
"""
struct FrameFeatures{F}
    pixels::Vector{SVector{2, Float64}}
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
`max_detections` strongest by |response|, described by `descriptor` (a
`LoMaDescriptor`) on the frame's tone-mapped RGB resampled to the network's input.
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
    keep = [i for i in 1:n if VgtkCore.Point2(s * dets[i].center[1], s * dets[i].center[2]) ∈ kept]
    keep = sort!(keep[sortperm(abs.(getproperty.(dets[keep], :response)); rev = true)][1:min(end, max_detections)])
    # the network's input: the frame scaled to LOMA_SIZE², a COLMAP point x ↦ (Wf, Hf)/LOMA_SIZE ⊙ x (a downscale, so every tap
    # lies on the frame and the warp's fill never enters)
    bm = VgtkCore.image_warp(AffineMap(SMatrix{2, 2, Float64, 4}(Wf / LOMA_SIZE, 0.0, 0.0, Hf / LOMA_SIZE), SVector(0.0, 0.0)),
                             VgtkCore.Rect(0, 0, LOMA_SIZE, LOMA_SIZE))
    img = Array{Float32}(undef, 3, LOMA_SIZE, LOMA_SIZE); plane = Matrix{Float32}(undef, LOMA_SIZE, LOMA_SIZE)
    for c in 1:3
        img[c, :, :] .= VgtkCore.image_warp!(plane, bm, view(toned, :, :, c))
    end
    # the detections in the network's coordinates (the descriptor reads their centres); the order is kept
    sx, sy = LOMA_SIZE / Wf, LOMA_SIZE / Hf
    scaled = [VgtkFeatures.Detection(VgtkCore.EllipticFrame(VgtkCore.Point2(sx * dets[i].center[1], sy * dets[i].center[2]),
                                                            SMatrix{2, 2, Float64}(sx^2, 0, 0, sy^2) .* VgtkCore.moment_matrix(dets[i])),
                                     dets[i].metadata) for i in keep]
    pixels = [SVector(s * dets[i].center[1], s * dets[i].center[2]) for i in keep]
    return FrameFeatures(pixels, descriptor(img, scaled), n)
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
