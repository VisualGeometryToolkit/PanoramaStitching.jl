# =============================================================================
# register_rotations — the rotations of a pure-rotation sequence
# =============================================================================
#
#   features   per frame (features.jl): ASIFT keypoints, DeDoDe descriptions
#   pairs      per frame pair: the LoMa matches as unit bearings with their back-projection
#              Jacobians, VisualGeometryToolkitEstimation's `find_rotation` (the RANSAC of the
#              pure rotation on bearings, its scorer and local optimization the caller's)
#   graph      the `ViewGraph` of the verified pairs (a disconnected sequence refused),
#              VisualGeometryToolkitEstimation's `fit_rotation_averaging` over every edge
#   tracks     `chain_tracks`: union-find over the pairs' inlier matches
#   bundle     `refine_rotations`: the rotations and the track directions by NLLSsolver
#              (the pixel reprojection error through the fixed camera)
#
# A task verb on images (naming.md: "images or a session in → a task verb naming the job"):
# `register_*` names the job — image registration — and returns the registration, as
# `find_boards` and `track_boards` do theirs. Not `find_*`, which the stack reserves for a
# sample-and-verify search (the pairs inside it are `find_rotation`s), and not `fit_*`/`refine_*`
# (it composes those estimators and does no estimation of its own).
# =============================================================================

"""
    Registration{T}

The result of [`register_rotations`](@ref): `rotations` (one `RotMatrix{3}` per
frame, `d = R_k·f_k`, the first frame's camera frame the common frame), `graph`
(the [`ViewGraph`](@ref) of the verified pairs), `pairs` (per verified pair
`(i, j)`: matches, inliers, the rms Sampson distance of its inliers in pixels and
its rotation angle in degrees), `averaging` (the `fit_rotation_averaging`
result), `tracks` (the tracks as `(frame, keypoint)` observations), `dropped`
(inconsistent tracks dropped), `bundle` (the `refine_rotations` result; its
metadata's `outcome` and `start_nonfinite` say how the bundle ended and whether
its start was finite, [`RotationBundleMetadata`](@ref)) and `features` (the
frames' [`FrameFeatures`](@ref)).
"""
struct Registration{A, B, F}
    rotations::Vector{RotMatrix{3, Float64, 9}}
    graph::ViewGraph{Int}
    pairs::Dict{Tuple{Int, Int}, NamedTuple{(:matches, :inliers, :rms_px, :rotation_deg), Tuple{Int, Int, Float64, Float64}}}
    averaging::A
    tracks::Vector{Vector{Tuple{Int, Int}}}
    dropped::Int
    bundle::B
    features::Vector{F}
end

"""
    register_rotations(frames, camera; projection = camera, raw_scale, circle, tone_percentile, detector_margin_px, asift_tilts, max_detections,
                       descriptor, matcher, min_score, pairs, max_lag, overlap_deg, min_inliers, seed,
                       scoring = nothing, local_optimization = nothing, max_trials, min_trials,
                       averaging_loss, bundle_loss, bundle_scale_multiple, bundle_iterations, bundle_max_iter, bundle_ftol,
                       bundle_xtol, say = _ -> nothing) -> Registration

The rotations of the pure-rotation sequence `frames` (`H × W × 3` linear images in
capture order, `raw_scale` camera pixels per frame pixel) seen by the calibrated
`camera` (fixed; any camera with `backproject_with_jacobian` and `backproject`),
projected in the bundle through `projection`, its closed-form forward map (the
camera itself when it is a `CameraModel`; for a grid model its `SplineProjection`,
see [`refine_rotations`](@ref)), with the view graph, the tracks and the bundle's
statistics. Every setting is the caller's configuration, none defaulted but the
RANSAC's:

- features: `circle` (a Core `Circle` or `Ellipse` in camera pixels, `WholePlane()`
  for a lens with none), `tone_percentile`,
  `detector_margin_px`, `asift_tilts` (`nothing`: IMAS-25), `max_detections` — see
  [`detect_features`](@ref); `descriptor` and `matcher`, VisualGeometryToolkitFeatures'
  `LoMaDescriptor` and `LoMaMatcher` variants (`:b`, `:b128`); `min_score`.
- pairs: `pairs = :all` (every frame pair) or `:lag_overlap` (pairs at most
  `max_lag` apart, then those whose averaged optical axes are within
  `overlap_deg`); a pair is verified by `find_rotation` (its `scoring` and
  `local_optimization` objects, `nothing` for VisualGeometryToolkitEstimation's
  defaults; `max_trials`, `min_trials`; the sampler seeded from `seed`) and kept
  with at least `min_inliers` inliers.
- graph and bundle: `averaging_loss`, `bundle_loss` (stack loss shapes),
  `bundle_scale_multiple`, `bundle_iterations`, `bundle_max_iter`, `bundle_ftol`,
  `bundle_xtol` — see [`refine_rotations`](@ref).

`say` receives one line per frame, pair and step.
"""
function register_rotations(frames::AbstractVector, camera; projection = camera, raw_scale::Integer, circle, tone_percentile::Real, detector_margin_px::Real,
                            asift_tilts, max_detections::Integer, descriptor::Symbol, matcher::Symbol, min_score::Real,
                            pairs::Symbol, max_lag::Integer, overlap_deg::Real, min_inliers::Integer, seed::Integer,
                            scoring = nothing, local_optimization = nothing, max_trials::Integer, min_trials::Integer,
                            averaging_loss::VgtkCore.AbstractLossShape, bundle_loss::VgtkCore.AbstractLossShape,
                            bundle_scale_multiple::Real, bundle_iterations::Integer, bundle_max_iter::Integer, bundle_ftol::Real,
                            bundle_xtol::Real, say = _ -> nothing)
    pairs in (:all, :lag_overlap) || throw(ArgumentError("register_rotations: pairs = :$pairs is not :all or :lag_overlap"))
    _forward_map(projection)                                                 # refused before the features, not after
    K = length(frames)
    desc = VgtkFeatures.LoMaDescriptor(descriptor; max_detections)
    match = VgtkFeatures.LoMaMatcher(matcher)
    feats = map(1:K) do k
        f = detect_features(frames[k], desc; raw_scale, circle, tone_percentile, detector_margin_px, asift_tilts, max_detections)
        say("  frame $(k - 1): $(f.n_detected) ASIFT detections, $(length(f.pixels)) kept")
        f
    end
    sz = (3, LOMA_SIZE, LOMA_SIZE)
    verified = Dict{Tuple{Int, Int}, NamedTuple{(:rotation, :inliers), Tuple{SMatrix{3, 3, Float64, 9}, Int}}}()
    inlier_matches = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    stats = Dict{Tuple{Int, Int}, NamedTuple{(:matches, :inliers, :rms_px, :rotation_deg), Tuple{Int, Int, Float64, Float64}}}()
    function verify(i, j)
        ms = match(feats[i].features, feats[j].features, sz, sz; min_score)
        if length(ms) < min_inliers
            say("  pair $(i - 1)-$(j - 1): $(length(ms)) matches → dropped"); return nothing
        end
        cs, kept = pair_bearings(camera, feats[i], feats[j], ms)
        kw = (; max_trials, min_trials, rng = Xoshiro(seed + 1000i + j))
        kw = scoring === nothing ? kw : merge(kw, (; scoring))
        kw = local_optimization === nothing ? kw : merge(kw, (; local_optimization))
        res = VgtkEstimation.find_rotation(cs; kw...)
        n = res === nothing ? 0 : count(res.inlier_mask)
        if res === nothing || !VgtkCore.converged(res) || n < min_inliers
            say("  pair $(i - 1)-$(j - 1): $(length(ms)) matches, $n inliers → dropped"); return nothing
        end
        R = SMatrix{3, 3, Float64, 9}(VgtkCore.estimate(res))
        r = res.residuals[res.inlier_mask]
        rms = sqrt(sum(abs2, r) / length(r)); ang = rad2deg(norm(VgtkCore.log_so3(R)))
        verified[(i, j)] = (rotation = R, inliers = n)
        inlier_matches[(i, j)] = kept[res.inlier_mask]
        stats[(i, j)] = (matches = length(ms), inliers = n, rms_px = rms, rotation_deg = ang)
        say("  pair $(i - 1)-$(j - 1): $(length(ms)) matches, $n inliers, rms $(round(rms; digits = 3)) px, rotation $(round(ang; digits = 2))°")
        return nothing
    end
    for i in 1:K, j in (i + 1):K
        (pairs === :all || j - i <= max_lag) && verify(i, j)
    end
    if pairs === :lag_overlap
        lag_averaging = VgtkEstimation.fit_rotation_averaging(Dict(p => e.rotation for (p, e) in verified),
                                                           Dict(p => Float64(e.inliers) for (p, e) in verified); loss = averaging_loss)
        z = SVector(0.0, 0.0, 1.0); A = VgtkCore.estimate(lag_averaging)
        for i in 1:K, j in (i + max_lag + 1):K
            rad2deg(VgtkCore.angle_between(SMatrix{3, 3}(A[i]) * z, SMatrix{3, 3}(A[j]) * z)) < overlap_deg && verify(i, j)
        end
    end
    graph = ViewGraph(collect(1:K), verified)
    avg = VgtkEstimation.fit_rotation_averaging(graph.rotations, Dict(p => Float64(n) for (p, n) in graph.inliers); loss = averaging_loss)
    say("  rotation averaging: $(length(graph.rotations)) edges, σ̂ $(round(rad2deg(avg.sigma); digits = 4))°, $(avg.retcode)")
    ts, dropped = chain_tracks(inlier_matches)
    say("  tracks: $(length(ts)) ($(sum(length, ts; init = 0)) observations), $dropped inconsistent dropped")
    obs = [[k => feats[k].pixels[kp] for (k, kp) in t] for t in ts]
    bundle = refine_rotations(camera, VgtkCore.estimate(avg), obs; projection, loss = bundle_loss, scale_multiple = bundle_scale_multiple,
                              iterations = bundle_iterations, max_iter = bundle_max_iter, ftol = bundle_ftol, xtol = bundle_xtol)
    say("  bundle: $(bundle.n) observations, rms $(round(bundle.rms; digits = 3)) px, loss width $(round(bundle.metadata.loss_width; digits = 3)) px, " *
        "$(bundle.retcode), outcome $(bundle.metadata.outcome) ($(bundle.metadata.start_nonfinite) observations not finite at the start)")
    Rs = [RotMatrix{3, Float64, 9}(VgtkCore.estimate(bundle)[k]) for k in 1:K]
    return Registration(Rs, graph, stats, avg, ts, dropped, bundle, feats)
end
