# =============================================================================
# register_rotations — the rotations of a pure-rotation sequence
# =============================================================================
#
#   features   per frame (features.jl): ASIFT keypoints, DeDoDe descriptions
#   pairs      per frame pair: the LoMa matches as unit bearings with their back-projection
#              Jacobians, VisualGeometryToolkitEstimation's `find_rotation` (the RANSAC of the
#              pure rotation on bearings, its scorer and local optimization the caller's)
#   graph      VisualGeometryToolkitEstimation's `ViewGraph` of the verified pairs, each edge its rotation
#              weighted by its inlier count (Core's `Attributed`), and Estimation's `filtered_rotation_averaging`
#              of it: the cycle filter, the averaging, the orientation filter (a disconnected sequence refused)
#   tracks     `chain_tracks`: the components of the kept pairs' inlier matches
#   bundle     `refine_rotations`: the rotations and the track directions by NLLSsolver
#              (the pixel reprojection error through the fixed camera)
#
# `verify_pairs` is the features and the pairs (a `VerifiedPairs`: everything the graph, the averaging, the tracks and the
# bundle read), and `register_rotations(::VerifiedPairs, camera; …)` the rest, so a caller that keeps the verified pairs
# reruns the averaging and the bundle alone; `register_rotations(frames, camera; …)` composes the two.
#
# A task verb on images (naming.md: "images or a session in → a task verb naming the job"):
# `register_*` names the job — image registration — and returns the registration, as
# `find_boards` and `track_boards` do theirs. Not `find_*`, which the stack reserves for a
# sample-and-verify search (the pairs inside it are `find_rotation`s), and not `fit_*`/`refine_*`
# (it composes those estimators and does no estimation of its own).
# =============================================================================

"""
    Registration{G, A, B, F}

The result of [`register_rotations`](@ref): `rotations` (one `RotMatrix{3}` per
frame, `d = R_k·f_k`, the first frame's camera frame the common frame), `graph`
(VisualGeometryToolkitEstimation's `ViewGraph` of the verified pairs the view-graph
filters kept: per pair `(i, j)` VisualGeometryToolkitCore's `Attributed(R_ij, (;
weight))`, `f_j = R_ij·f_i`, its weight the inlier count), `removed` (the verified
pairs the filters removed, sorted), `pairs` (per verified pair
`(i, j)`, kept or removed: matches, inliers, the rms Sampson distance of its inliers in pixels and
its rotation angle in degrees), `averaging` (the averaging fitted on `graph`,
VisualGeometryToolkitEstimation's `filtered_rotation_averaging`), `tracks` (of the kept pairs) (the tracks as `(frame, keypoint)` observations), `dropped`
(inconsistent tracks dropped), `bundle` (the `refine_rotations` result; its
metadata's `outcome` and `n_dropped_start` say how the bundle ended and how many
observations it left out at its start, [`RotationBundleMetadata`](@ref)), `features` (the
frames' [`FrameFeatures`](@ref)) and `loma` (what ran LoMa: VisualGeometryToolkitFeatures'
`loma_provenance` of the descriptor and of the matcher — the models, the ONNX Runtime
provider used and its threads).
"""
struct Registration{G, A, B, F}
    rotations::Vector{RotMatrix{3, Float64, 9}}
    graph::G
    removed::Vector{Tuple{Int, Int}}
    pairs::Dict{Tuple{Int, Int}, NamedTuple{(:matches, :inliers, :rms_px, :rotation_deg), Tuple{Int, Int, Float64, Float64}}}
    averaging::A
    tracks::Vector{Vector{Tuple{Int, Int}}}
    dropped::Int
    bundle::B
    features::Vector{F}
    loma::NamedTuple    # (; descriptor, matcher): loma_provenance of each
end

"""
    VerifiedPairs{F}

The verified frame pairs of a sequence, [`verify_pairs`](@ref)'s result and the
input of the rotations ([`register_rotations`](@ref)): `features` (per frame its
[`FrameFeatures`](@ref); `F` is `FrameFeatures{Nothing}` for pairs read back from
a record, which keeps no descriptions), `verified` (per verified pair `(i, j)`
its rotation `R_ij`, `f_j = R_ij·f_i`, and its inlier count: what the view graph
and the rotation averaging consume), `inlier_matches` (per verified pair its
inliers as keypoint index pairs `(a, b)`, keypoint `a` of frame `i` ↔ keypoint `b`
of frame `j`: what the tracks consume) and `stats` (per verified pair: matches,
inliers, the rms Sampson distance of its inliers in pixels and its rotation
angle in degrees) and `loma` (what ran LoMa: VisualGeometryToolkitFeatures'
`loma_provenance` of the descriptor and of the matcher, `(; descriptor,
matcher)`). Everything a registration needs after the features, the
matching and the verification, so a caller may keep it and rerun the averaging
and the bundle alone.
"""
struct VerifiedPairs{F}
    features::Vector{F}
    verified::Dict{Tuple{Int, Int}, NamedTuple{(:rotation, :inliers), Tuple{SMatrix{3, 3, Float64, 9}, Int}}}
    inlier_matches::Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}
    stats::Dict{Tuple{Int, Int}, NamedTuple{(:matches, :inliers, :rms_px, :rotation_deg), Tuple{Int, Int, Float64, Float64}}}
    loma::NamedTuple    # (; descriptor, matcher): loma_provenance of each
end

"""
    verify_pairs(frames, camera::BoundedCamera; raw_scale, tone_percentile, detector_margin_px, asift_tilts, max_detections,
                 selection = :spread, detection_image = :tone, detection_noise = nothing, descriptor, matcher, min_score, pairs, max_lag, overlap_deg, min_inliers, seed,
                 scoring = nothing, local_optimization = nothing, max_trials, min_trials, averaging_loss,
                 execution_provider = :auto, say = _ -> nothing) -> VerifiedPairs

The verified frame pairs of the pure-rotation sequence `frames`: the features,
the matching and the pairwise rotations of [`register_rotations`](@ref) (its
keywords of the same names), stopping before the view graph. `averaging_loss` is
read only by `pairs = :lag_overlap`, whose rotation averaging of the verified
pairs aims the pairs past the lag. A `pairs` other than `:all` or
`:lag_overlap`, a `max_lag` below 1 with `:lag_overlap`, a detector setting
[`detect_features`](@ref) refuses (`detection_image = :vst` needs every frame a
`LinearRaw`) and an `execution_provider` other than `:auto`, `:cpu` or `:cuda`
are an `ArgumentError` before any feature is detected.
"""
function verify_pairs(frames::AbstractVector, camera::VgtkCore.BoundedCamera; raw_scale::Integer,
                      tone_percentile::Real, detector_margin_px::Real, asift_tilts, max_detections::Integer,
                      selection::Symbol = :spread, detection_image::Symbol = :tone, detection_noise = nothing, descriptor::Symbol, matcher::Symbol,
                      min_score::Real, pairs::Symbol, max_lag::Integer, overlap_deg::Real, min_inliers::Integer, seed::Integer,
                      scoring = nothing, local_optimization = nothing, max_trials::Integer, min_trials::Integer,
                      averaging_loss::VgtkCore.AbstractLossShape, execution_provider::Symbol = :auto, say = _ -> nothing)
    _check_pair_settings("verify_pairs", pairs, max_lag)
    _check_frames("verify_pairs", frames, selection, detection_image, detection_noise)
    # resolved once (one warning on a fallback to the CPU), refusing an unknown provider, before any feature is detected
    provider = VgtkFeatures.resolve_execution_provider(execution_provider)
    K = length(frames)
    # one ONNX session each for the whole registration, on the resolved provider
    desc = VgtkFeatures.LoMaDescriptor(descriptor; max_detections, execution_provider = provider)
    match = VgtkFeatures.LoMaMatcher(matcher; execution_provider = provider)
    feats = map(1:K) do k
        f = detect_features(camera, frames[k], desc; raw_scale, tone_percentile, detector_margin_px, asift_tilts, max_detections,
                            selection, detection_image, detection_noise)
        say("  frame $(k - 1): $(f.n_detected) ASIFT detections, $(length(f.pixels)) kept")
        f
    end
    sz(k) = (size(_frame_image(frames[k]), 1), size(_frame_image(frames[k]), 2))                         # the image a frame's features are in: the frame
    verified = Dict{Tuple{Int, Int}, NamedTuple{(:rotation, :inliers), Tuple{SMatrix{3, 3, Float64, 9}, Int}}}()
    inlier_matches = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    stats = Dict{Tuple{Int, Int}, NamedTuple{(:matches, :inliers, :rms_px, :rotation_deg), Tuple{Int, Int, Float64, Float64}}}()
    function verify(i, j)
        ms = match(feats[i].features, feats[j].features, sz(i), sz(j); min_score)
        if length(ms) < min_inliers
            say("  pair $(i - 1)-$(j - 1): $(length(ms)) matches → dropped"); return nothing
        end
        cs, kept = pair_bearings(camera, feats[i], feats[j], ms)
        kw = (; max_trials, min_trials, rng = _pair_rng(seed, i, j))
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
    pairs === :lag_overlap && _verify_overlap_pairs!(verify, verified, K, max_lag, overlap_deg, averaging_loss)
    loma = (; descriptor = VgtkFeatures.loma_provenance(desc), matcher = VgtkFeatures.loma_provenance(match))
    return VerifiedPairs(feats, verified, inlier_matches, stats, loma)
end

# The settings every entry checks before any feature is detected.
# (The execution provider: VisualGeometryToolkitFeatures' `resolve_execution_provider`, which refuses an unknown one.)
function _check_pair_settings(what, pairs, max_lag)
    pairs in (:all, :lag_overlap) || throw(ArgumentError("$what: pairs = :$pairs is not :all or :lag_overlap"))
    pairs === :all || max_lag >= 1 || throw(ArgumentError("$what: pairs = :lag_overlap needs max_lag ≥ 1, got $max_lag"))
    return nothing
end
# The detector's settings and, with detection_image = :vst, every frame a LinearRaw: before any feature is detected.
function _check_frames(what, frames, selection, detection_image, detection_noise)
    _check_detector_settings(what, selection, detection_image, detection_noise)
    detection_image === :vst && !all(f -> f isa VgtkFeatures.LinearRaw, frames) &&
        throw(ArgumentError("$what: detection_image = :vst needs every frame a LinearRaw (its green cells' code spans)"))
    return nothing
end
_check_whitening(what, bundle_whitening) = bundle_whitening in (:none, :keypoint_covariance) ||
    throw(ArgumentError("$what: bundle_whitening = :$bundle_whitening is not :none or :keypoint_covariance"))

"""
    register_rotations(frames, camera::BoundedCamera; raw_scale, tone_percentile, detector_margin_px, asift_tilts, max_detections,
                       selection = :spread, detection_image = :tone, detection_noise = nothing,
                       descriptor, matcher, min_score, pairs, max_lag, overlap_deg, min_inliers, seed,
                       scoring = nothing, local_optimization = nothing, max_trials, min_trials,
                       averaging_loss, cycle_tol = deg2rad(2), orientation_tol = deg2rad(5), bundle_loss, bundle_scale_multiple, bundle_iterations, bundle_max_iter, bundle_ftol,
                       bundle_xtol, bundle_whitening = :none, bundle_camera_free = (), execution_provider = :auto,
                       say = _ -> nothing) -> Registration
    register_rotations(pairs::VerifiedPairs, camera::BoundedCamera; averaging_loss, cycle_tol = deg2rad(2),
                       orientation_tol = deg2rad(5), bundle_loss, bundle_scale_multiple,
                       bundle_iterations, bundle_max_iter, bundle_ftol, bundle_xtol, bundle_whitening = :none,
                       bundle_camera_free = (), say = _ -> nothing) -> Registration

The rotations of the pure-rotation sequence `frames` (`H × W × 3` linear images in
capture order, `raw_scale` camera pixels per frame pixel) seen by the calibrated
`camera` (fixed; a VisualGeometryToolkitCore `BoundedCamera`, whose `imaged_region` bounds the features,
projected in the bundle through its model's `imaging_transform` — a grid lens's `SplineLens` through its forward spline; see
[`refine_rotations`](@ref)), with the view graph, the tracks and the bundle's
statistics. Every setting is the caller's configuration, none defaulted but the
RANSAC's:

- features: inside the camera's `imaged_region`; `tone_percentile`,
  `detector_margin_px`, `asift_tilts` (`nothing`: IMAS-25), `max_detections`,
  `selection`, `detection_image`, `detection_noise` — see
  [`detect_features`](@ref) (`detection_image = :vst`: the frames are `LinearRaw`s); `descriptor` and `matcher`, VisualGeometryToolkitFeatures'
  `LoMaDescriptor` and `LoMaMatcher` variants (`:b`, `:b128`), one ONNX session
  each for the whole registration on ONNX Runtime's `execution_provider`
  (`:auto`, the default: CUDA where VisualGeometryToolkitFeatures finds a device and
  the environment holds `CUDA_Runtime_jll` and `CUDNN_jll`, else the CPU with a
  warning naming why; `:cpu`; `:cuda`, an error naming why where it cannot run),
  on `Threads.nthreads()` ONNX Runtime threads; `min_score`. The provider used is
  in the result's `loma`.
- pairs: `pairs = :all` (every frame pair) or `:lag_overlap` (pairs at most
  `max_lag` apart; while the verified pairs leave the frames in more than one
  part, as when a lag pair is dropped, the pair between two parts nearest in
  capture order; then those whose optical axes, aimed by the rotation
  averaging of the verified pairs, are within `overlap_deg`); a pair is verified by `find_rotation` (its `scoring` and
  `local_optimization` objects, `nothing` for VisualGeometryToolkitEstimation's
  defaults; `max_trials`, `min_trials`; the sampler seeded from `seed`) and kept
  with at least `min_inliers` inliers.
- graph: VisualGeometryToolkitEstimation's `filtered_rotation_averaging` of the
  verified pairs' view graph, its loss `averaging_loss`: the pairs off a cycle
  by more than `cycle_tol` removed (`filter_cycles`), the averaging, then the
  pairs off the averaged rotations by more than `orientation_tol` removed
  (`filter_by_orientation`) and the averaging again (radians; `nothing` skips
  that filter). A periodic scene verifies a pair one period off as well as the
  true one, and only the view graph tells them apart. The removed pairs are the
  result's `removed`; their matches make no track.
- bundle: `bundle_loss` (stack loss shapes),
  `bundle_scale_multiple`, `bundle_iterations`, `bundle_max_iter`, `bundle_ftol`,
  `bundle_xtol` — see [`refine_rotations`](@ref). `bundle_whitening = :none`
  (the default) weighs every observation alike, in pixels;
  `:keypoint_covariance` whitens each by its keypoint's position covariance
  ([`FrameFeatures`](@ref)'s `pixel_covariances`; a keypoint the detector did
  not refine takes the isotropic median), the loss acting on the whitened
  residuals. `bundle_camera_free` (by name, the camera parameters the bundle
  refines with the rotations, `refine_rotations`' `camera_free`; `()`, the
  default, holds the camera): the rotating camera's self-calibration, its camera
  the bundle metadata's `camera` (`result.bundle.metadata.camera`) and its
  information `result.bundle.metadata.calibration`. The pairs, the averaging and
  the tracks are those of the given camera.

`say` receives one line per frame, pair and step.

The second form starts from the verified pairs ([`verify_pairs`](@ref), the
first form's first step: everything up to the view graph), so a caller that keeps
them reruns the averaging and the bundle alone; the first form is
`register_rotations(verify_pairs(frames, camera; …), camera; …)`.

A task verb: it composes the estimators above and does none of its own. Each
pair's RANSAC sampler is seeded from `seed` and the pair's frame indices, a
stream of its own per `(seed, i, j)`. A `pairs` other than `:all` or
`:lag_overlap`, a `max_lag` below 1 with `:lag_overlap`, a `bundle_whitening`
other than `:none` or `:keypoint_covariance` and an `execution_provider` other
than `:auto`, `:cpu` or `:cuda` are an `ArgumentError` before any feature is detected. With either `pairs`, a view graph that is not connected once the
pairs with fewer than `min_inliers` inliers are dropped (its frames have no
common frame; VisualGeometryToolkitEstimation's `require_connected`) is an `ArgumentError` too.
"""
function register_rotations(frames::AbstractVector, camera::VgtkCore.BoundedCamera; raw_scale::Integer,
                            tone_percentile::Real, detector_margin_px::Real, asift_tilts, max_detections::Integer,
                            selection::Symbol = :spread, detection_image::Symbol = :tone, detection_noise = nothing,
                            descriptor::Symbol, matcher::Symbol, min_score::Real,
                            pairs::Symbol, max_lag::Integer, overlap_deg::Real, min_inliers::Integer, seed::Integer,
                            scoring = nothing, local_optimization = nothing, max_trials::Integer, min_trials::Integer,
                            averaging_loss::VgtkCore.AbstractLossShape, cycle_tol::Union{Nothing, Real} = deg2rad(2),
                            orientation_tol::Union{Nothing, Real} = deg2rad(5), bundle_loss::VgtkCore.AbstractLossShape,
                            bundle_scale_multiple::Real, bundle_iterations::Integer, bundle_max_iter::Integer, bundle_ftol::Real,
                            bundle_xtol::Real, bundle_whitening::Symbol = :none, bundle_camera_free::Tuple{Vararg{Symbol}} = (),
                            execution_provider::Symbol = :auto, say = _ -> nothing)
    _check_pair_settings("register_rotations", pairs, max_lag)
    _check_whitening("register_rotations", bundle_whitening)
    _check_tolerances("register_rotations", cycle_tol, orientation_tol)
    vp = verify_pairs(frames, camera; raw_scale, tone_percentile, detector_margin_px, asift_tilts, max_detections, selection,
                      detection_image, detection_noise, descriptor, matcher,
                      min_score, pairs, max_lag, overlap_deg, min_inliers, seed, scoring, local_optimization, max_trials, min_trials,
                      averaging_loss, execution_provider, say)
    return register_rotations(vp, camera; averaging_loss, cycle_tol, orientation_tol, bundle_loss, bundle_scale_multiple, bundle_iterations, bundle_max_iter,
                              bundle_ftol, bundle_xtol, bundle_whitening, bundle_camera_free, say)
end

function register_rotations(vp::VerifiedPairs, camera::VgtkCore.BoundedCamera; averaging_loss::VgtkCore.AbstractLossShape,
                            cycle_tol::Union{Nothing, Real} = deg2rad(2), orientation_tol::Union{Nothing, Real} = deg2rad(5),
                            bundle_loss::VgtkCore.AbstractLossShape, bundle_scale_multiple::Real, bundle_iterations::Integer,
                            bundle_max_iter::Integer, bundle_ftol::Real, bundle_xtol::Real, bundle_whitening::Symbol = :none,
                            bundle_camera_free::Tuple{Vararg{Symbol}} = (), say = _ -> nothing)
    _check_whitening("register_rotations", bundle_whitening)
    feats = vp.features; K = length(feats)
    (; fit, graph) = VgtkEstimation.filtered_rotation_averaging(_view_graph(K, vp.verified); cycle_tol, orientation_tol, loss = averaging_loss)
    avg = fit
    removed = sort!([p for p in keys(vp.verified) if !haskey(graph, p...)])
    say("  rotation averaging: $(VgtkEstimation.nedges(graph)) edges ($(length(removed)) removed by the view-graph filters" *
        "$(isempty(removed) ? "" : ": " * join(["$(i - 1)-$(j - 1)" for (i, j) in removed], ", "))), σ̂ $(round(rad2deg(avg.sigma); digits = 4))°, $(avg.retcode)")
    ts, dropped = chain_tracks(Dict(p => m for (p, m) in vp.inlier_matches if haskey(graph, p...)))
    say("  tracks: $(length(ts)) ($(sum(length, ts; init = 0)) observations), $dropped inconsistent dropped")
    obs = bundle_whitening === :none ? [[k => feats[k].pixels[kp] for (k, kp) in t] for t in ts] : _uncertain_observations(feats, ts; say)
    bundle = refine_rotations(camera, VgtkCore.estimate(avg), obs; loss = bundle_loss, scale_multiple = bundle_scale_multiple,
                              iterations = bundle_iterations, max_iter = bundle_max_iter, ftol = bundle_ftol, xtol = bundle_xtol,
                              camera_free = bundle_camera_free)
    say("  bundle: $(bundle.n) observations, rms $(round(bundle.rms; digits = 3)) px, loss width $(round(bundle.metadata.loss_width; digits = 3)) px, " *
        "$(bundle.retcode), outcome $(bundle.metadata.outcome) ($(bundle.metadata.n_dropped_start) observations dropped at the start: not imaged there)")
    Rs = [RotMatrix{3, Float64, 9}(VgtkCore.estimate(bundle)[k]) for k in 1:K]
    return Registration(Rs, graph, removed, vp.stats, avg, ts, dropped, bundle, feats, vp.loma)
end

# The view-graph filters' tolerances (radians): positive, or nothing (that filter skipped). Checked before any feature is detected;
# from the verified pairs, VisualGeometryToolkitEstimation's `filtered_rotation_averaging` refuses them itself.
function _check_tolerances(what, cycle_tol, orientation_tol)
    for (name, t) in (("cycle_tol", cycle_tol), ("orientation_tol", orientation_tol))
        t === nothing || (isfinite(t) && t > 0) || throw(ArgumentError("$what: $name = $t is not positive (radians; nothing skips the filter)"))
    end
    return nothing
end

# The view graph of the frames 1:K and the verified pairs: VisualGeometryToolkitEstimation's `ViewGraph`, each edge the pair's
# rotation `R_ij` (`f_j = R_ij·f_i`) weighted by its inlier count, VisualGeometryToolkitCore's `Attributed(R_ij, (; weight))`.
_view_graph(K::Integer, verified::AbstractDict) =
    VgtkEstimation.ViewGraph(collect(1:K), Dict(p => VgtkCore.Attributed(e.rotation, (; weight = Float64(e.inliers))) for (p, e) in verified))

# The bundle's observations with their keypoints' position covariances (`bundle_whitening = :keypoint_covariance`): each
# `frame => Uncertain(pixel, CovMat(Σ))`, Σ the keypoint's camera-pixel covariance (`FrameFeatures`). A keypoint the detector did
# not refine (Σ NaN), or whose Σ is not positive definite, takes the isotropic fallback σ²·I, σ² the median over the tracks'
# refined keypoints of tr Σ / 2.
function _uncertain_observations(feats, ts; say = _ -> nothing)
    valid(Σ) = all(isfinite, Σ) && Σ[1, 1] > 0 && Σ[1, 1] * Σ[2, 2] - Σ[1, 2] * Σ[2, 1] > 0
    Σs = [feats[k].pixel_covariances[kp] for t in ts for (k, kp) in t]
    good = filter(valid, Σs)
    isempty(good) && throw(ArgumentError("register_rotations: bundle_whitening = :keypoint_covariance, but no tracked keypoint has a covariance"))
    fallback = SMatrix{2, 2, Float64, 4}(Statistics.median(Σ[1, 1] + Σ[2, 2] for Σ in good) / 2 * I)
    say("  bundle whitening: keypoint covariances, $(length(Σs) - length(good)) of $(length(Σs)) observations take the isotropic fallback " *
        "($(round(sqrt(fallback[1, 1]); sigdigits = 3)) px)")
    return [[k => VgtkCore.Uncertain(feats[k].pixels[kp], VgtkCore.CovMat(valid(feats[k].pixel_covariances[kp]) ? feats[k].pixel_covariances[kp] : fallback))
             for (k, kp) in t] for t in ts]
end

# The RANSAC sampler of the pair (i, j): Core's `stable_seed` keyed by the pair's frame indices (a SHA-256 of the integers'
# decimal text, so no two (seed, i, j) share a stream) seeding a StableRNG (its stream fixed across Julia versions).
_pair_rng(seed::Integer, i::Integer, j::Integer) = StableRNG(VgtkCore.stable_seed(seed, i, j))

# The pairs of `pairs = :lag_overlap` past the lag, after the lag pairs: `verify(i, j)` adds a pair it keeps to `verified`.
# While the verified pairs leave the frames in more than one part (a lag pair dropped), the untried pair between two parts
# nearest in capture order is tried, since no verified rotation relates their frames; when no such pair is left the graph stays
# disconnected (the rotation averaging refuses it). Then the rotation averaging of every verified pair aims the rest: a pair whose optical
# axes are within `overlap_deg` is tried, in (i, j) order.
function _verify_overlap_pairs!(verify, verified::AbstractDict, K::Integer, max_lag::Integer, overlap_deg::Real, averaging_loss)
    tried = Set{Tuple{Int, Int}}()                               # pairs past the lag: no lag pair comes up again
    attempt(p) = (push!(tried, p); verify(p...))
    while true
        parts = VgtkEstimation.components(_view_graph(K, verified))
        length(parts) == 1 && break
        part = zeros(Int, K)
        for (c, fs) in enumerate(parts)
            part[fs] .= c
        end
        joins = [(i, j) for i in 1:K for j in (i + max_lag + 1):K if part[i] != part[j] && (i, j) ∉ tried]
        isempty(joins) && return nothing
        attempt(argmin(p -> (p[2] - p[1], p[1]), joins))
    end
    K > 1 || return nothing
    A = VgtkCore.estimate(VgtkEstimation.fit_rotation_averaging(_view_graph(K, verified); loss = averaging_loss))
    axis = [SMatrix{3, 3}(A[k]) * SVector(0.0, 0.0, 1.0) for k in 1:K]
    for i in 1:K, j in (i + max_lag + 1):K
        ((i, j) ∉ tried && rad2deg(VgtkCore.angle_between(axis[i], axis[j])) < overlap_deg) && attempt((i, j))
    end
    return nothing
end
