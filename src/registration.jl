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
VisualGeometryToolkitEstimation's `filtered_rotation_averaging`), `tracks` (of the kept pairs, as `(frame, keypoint)` observations), `dropped`
(inconsistent tracks dropped), `bundle` (the `refine_rotations` result; its
metadata's `outcome` and `n_dropped_start` say how the bundle ended and how many
observations it left out at its start, [`RotationBundleMetadata`](@ref)), `bundle_sigma` (the
pooled noise scale σ̂ of the kept pairs, px, whose `bundle_efficiency_constant` is the bundle's width), `features` (the
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
    bundle_sigma::Float64
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
angle in degrees), `scales` (per verified pair its noise scale: `sigma`, px per
coordinate of its Sampson residual, and `success`, whether SIMFIT++ found one —
see [`register_rotations`](@ref)) and `loma` (what ran LoMa: VisualGeometryToolkitFeatures'
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
    scales::Dict{Tuple{Int, Int}, NamedTuple{(:sigma, :success), Tuple{Float64, Bool}}}
    loma::NamedTuple    # (; descriptor, matcher): loma_provenance of each
end

"""
    verify_pairs(frames, camera::BoundedCamera; frame_map, tone_percentile, detector_margin_px, asift_tilts, max_detections,
                 selection = :spread, detection_image = :tone, detection_noise = nothing, saturated = nothing, descriptor, matcher, min_score, pairs, max_lag, overlap_deg, min_inliers, seed,
                 scoring = nothing, local_optimization_strategy = nothing, max_trials, min_trials, averaging_loss,
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
function verify_pairs(frames::AbstractVector, camera::VgtkCore.BoundedCamera; frame_map,
                      tone_percentile::Real, detector_margin_px::Real, asift_tilts, max_detections::Integer,
                      selection::Symbol = :spread, detection_image::Symbol = :tone, detection_noise = nothing,
                      saturated::Union{Nothing, AbstractVector} = nothing, descriptor::Symbol, matcher::Symbol,
                      min_score::Real, pairs::Symbol, max_lag::Integer, overlap_deg::Real, min_inliers::Integer, seed::Integer,
                      scoring = nothing, local_optimization_strategy = nothing, max_trials::Integer, min_trials::Integer,
                      averaging_loss::VgtkCore.AbstractLossShape, execution_provider::Symbol = :auto, say = _ -> nothing, retired...)
    _refuse_retired_registration_keywords("verify_pairs", retired)
    _check_pair_settings("verify_pairs", pairs, max_lag)
    _check_frames("verify_pairs", frames, selection, detection_image, detection_noise)
    saturated === nothing || length(saturated) == length(frames) ||
        throw(DimensionMismatch("verify_pairs: $(length(frames)) frames and $(length(saturated)) saturation masks"))
    # resolved once (one warning on a fallback to the CPU), refusing an unknown provider, before any feature is detected
    provider = VgtkFeatures.resolve_execution_provider(execution_provider)
    K = length(frames)
    # one ONNX session each for the whole registration, on the resolved provider
    desc = VgtkFeatures.LoMaDescriptor(descriptor; max_detections, execution_provider = provider)
    match = VgtkFeatures.LoMaMatcher(matcher; execution_provider = provider)
    feats = map(1:K) do k
        f = detect_features(camera, frames[k], desc; frame_map, tone_percentile, detector_margin_px, asift_tilts, max_detections,
                            selection, detection_image, detection_noise, saturated = saturated === nothing ? nothing : saturated[k])
        say("  frame $(k - 1): $(f.n_detected) ASIFT detections, $(length(f.pixels)) kept")
        f
    end
    sz(k) = (size(_frame_image(frames[k]), 1), size(_frame_image(frames[k]), 2))                         # the image a frame's features are in: the frame
    verified = Dict{Tuple{Int, Int}, NamedTuple{(:rotation, :inliers), Tuple{SMatrix{3, 3, Float64, 9}, Int}}}()
    inlier_matches = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    stats = Dict{Tuple{Int, Int}, NamedTuple{(:matches, :inliers, :rms_px, :rotation_deg), Tuple{Int, Int, Float64, Float64}}}()
    scales = Dict{Tuple{Int, Int}, NamedTuple{(:sigma, :success), Tuple{Float64, Bool}}}()
    function verify(i, j)
        ms = match(feats[i].features, feats[j].features, sz(i), sz(j); min_score)
        if length(ms) < min_inliers
            say("  pair $(i - 1)-$(j - 1): $(length(ms)) matches → dropped"); return nothing
        end
        cs, kept = pair_bearings(camera, feats[i], feats[j], ms)
        kw = (; max_trials, min_trials, rng = _pair_rng(seed, i, j))
        kw = scoring === nothing ? kw : merge(kw, (; scoring))
        kw = local_optimization_strategy === nothing ? kw : merge(kw, (; local_optimization_strategy))
        res = VgtkEstimation.find_rotation(cs; kw...)
        n = res === nothing ? 0 : count(VgtkCore.inlier_mask(res))
        if res === nothing || !VgtkCore.converged(res) || n < min_inliers
            say("  pair $(i - 1)-$(j - 1): $(length(ms)) matches, $n inliers → dropped"); return nothing
        end
        R = SMatrix{3, 3, Float64, 9}(VgtkCore.estimate(res))
        r = res.residuals[VgtkCore.inlier_mask(res)]
        rms = sqrt(sum(abs2, r) / length(r)); ang = rad2deg(norm(VgtkCore.log_so3(R)))
        verified[(i, j)] = (rotation = R, inliers = n)
        inlier_matches[(i, j)] = kept[VgtkCore.inlier_mask(res)]
        stats[(i, j)] = (matches = length(ms), inliers = n, rms_px = rms, rotation_deg = ang)
        scales[(i, j)] = _pair_scale(cs, res, local_optimization_strategy, _pair_rng(seed, i, j))
        say("  pair $(i - 1)-$(j - 1): $(length(ms)) matches, $n inliers, rms $(round(rms; digits = 3)) px, rotation $(round(ang; digits = 2))°, " *
            "σ̂ $(round(scales[(i, j)].sigma; digits = 3)) px$(scales[(i, j)].success ? "" : " (SIMFIT++ found none)")")
        return nothing
    end
    for i in 1:K, j in (i + 1):K
        (pairs === :all || j - i <= max_lag) && verify(i, j)
    end
    pairs === :lag_overlap && _verify_overlap_pairs!(verify, verified, K, max_lag, overlap_deg, averaging_loss)
    loma = (; descriptor = VgtkFeatures.loma_provenance(desc), matcher = VgtkFeatures.loma_provenance(match))
    return VerifiedPairs(feats, verified, inlier_matches, stats, scales, loma)
end

# SIMFIT++ (Edstedt, CVPRW 2025; VisualGeometryToolkitEstimation's port) on the Sampson residual of a pair's rotation: χ² of its 2
# degrees of freedom (the residual's codimension), the reference's coverage α = 0.99 and its width bounds (px).
const _SIMFIT = VgtkEstimation.SimFitPP(; dg = 2)
_chisq2_quantile(α) = -2 * log1p(-α)          # the χ²₂ quantile, closed form: F(x) = 1 − exp(−x/2)

# A verified pair's noise scale: SIMFIT++'s loss width over its correspondences, the rotation refitted on a random train split
# at the current width (Estimation's `refine_rotation` from the RANSAC estimate, MSAC at that width and the pair's local
# optimization) and the held-out split's squared residuals at it, seeded with the RANSAC's own σ̂ (its consensus Student-t
# scale) as a width; σ̂ = width / √χ²₂(α). `rng` the pair's own stream. `success` false when SIMFIT++ found no width in its
# bounds (the RANSAC σ̂ is then the pair's sigma, and the pair is not pooled).
function _pair_scale(cs, res, local_optimization_strategy, rng)
    q = _chisq2_quantile(_SIMFIT.alpha)
    # the RANSAC's own scale, read through the hand-off accessor (a result that recorded none has no seed width)
    s = VgtkCore.scale_source(res) == VgtkCore.ScaleNone ? NaN : Float64(VgtkCore.noise_scale(res))
    w0 = sqrt(q) * s
    isfinite(w0) && w0 > 0 || return (sigma = s, success = false)
    R0 = VgtkCore.estimate(res)
    lo = local_optimization_strategy === nothing ? (;) : (; optimization_strategy = local_optimization_strategy)   # a refine's keyword
    function fit(w, train)
        r = VgtkEstimation.refine_rotation(R0, cs[collect(train)]; scoring = VgtkCore.MsacScoring(VgtkEstimation.RotationProblem; threshold = w), lo...)
        return VgtkCore.converged(r) ? VgtkCore.estimate(r) : nothing
    end
    residuals_sq(R, val) = abs2.(VgtkEstimation.refine_rotation(R, cs[collect(val)]; optimization_strategy = nothing).residuals)
    w, ok = VgtkEstimation.simfitpp_loss_width(fit, residuals_sq, length(cs), w0, _SIMFIT; rng)
    return (sigma = w / sqrt(q), success = ok)
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

# The registration's retired keywords, refused naming their successor; any other unknown keyword is refused as unknown.
const _RETIRED_REGISTRATION_KEYWORDS = (
    bundle_camera_free = "the bundle holds the camera's calibration (refine_rotations); the autocalibration of a rotating camera is " *
                         "refine_camera_and_rotations",
    bundle_scale_multiple = "the bundle's width multiple is `bundle_efficiency_constant` (the plain multiple of the pooled scale)",
    local_optimization = "the pairs' RANSAC strategy is `local_optimization_strategy`")
function _refuse_retired_registration_keywords(what, kwargs)
    for (k, _) in pairs(kwargs)
        haskey(_RETIRED_REGISTRATION_KEYWORDS, k) && throw(ArgumentError("$what: the keyword `$k` is retired: $(_RETIRED_REGISTRATION_KEYWORDS[k])"))
        throw(ArgumentError("$what: unknown keyword `$k`"))
    end
    return nothing
end

"""
    register_rotations(frames, camera::BoundedCamera; frame_map, tone_percentile, detector_margin_px, asift_tilts, max_detections,
                       selection = :spread, detection_image = :tone, detection_noise = nothing, saturated = nothing,
                       descriptor, matcher, min_score, pairs, max_lag, overlap_deg, min_inliers, seed,
                       scoring = nothing, local_optimization_strategy = nothing, max_trials, min_trials,
                       averaging_loss, cycle_tol = deg2rad(2), orientation_tol = deg2rad(5), bundle_loss, bundle_efficiency_constant, bundle_max_iter, bundle_ftol,
                       bundle_xtol, bundle_whitening = :none, execution_provider = :auto,
                       say = _ -> nothing) -> Registration
    register_rotations(pairs::VerifiedPairs, camera::BoundedCamera; averaging_loss, cycle_tol = deg2rad(2),
                       orientation_tol = deg2rad(5), bundle_loss, bundle_efficiency_constant,
                       bundle_max_iter, bundle_ftol, bundle_xtol, bundle_whitening = :none,
                       say = _ -> nothing) -> Registration

The rotations of the pure-rotation sequence `frames` (`H × W × 3` linear images in
capture order, `frame_map` their map onto the camera's pixels: VisualGeometryToolkitFeatures' `raw_frame_map`) seen by the calibrated
`camera` (fixed; a VisualGeometryToolkitCore `BoundedCamera`, whose `imaged_region` bounds the features,
projected in the bundle through its model's `imaging_transform` — a grid lens's `SplineLens` through its forward spline; see
[`refine_rotations`](@ref)), with the view graph, the tracks and the bundle's
statistics. Every setting is the caller's configuration, none defaulted but the
RANSAC's:

- features: inside the camera's `imaged_region`; `tone_percentile`,
  `detector_margin_px`, `asift_tilts` (a Features `tilt_schedule` name, `:imas25`
  or `:untilted`, or its pairs), `max_detections`, `selection`,
  `detection_image`, `detection_noise`, `saturated` (the frames' clip masks,
  `nothing`: a `LinearRaw`'s own, none for an array) — see
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
  `local_optimization_strategy` objects, `nothing` for VisualGeometryToolkitEstimation's
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
- scale: per verified pair, after its RANSAC, its noise scale by SIMFIT++
  (Edstedt, "Less Biased Noise Scale Estimation for Threshold-Robust RANSAC",
  CVPRW 2025; VisualGeometryToolkitEstimation's `simfitpp_loss_width` with χ²₂, the
  rotation's Sampson residual's codimension, at coverage α = 0.99): the
  rotation refitted on random train splits (`refine_rotation`, MSAC at the
  current width, the pair's `local_optimization_strategy`), the held-out residuals'
  truncation-debiased median the scale; seeded with the RANSAC's own σ̂, the
  sampler the pair's (`seed`, `i`, `j`). The pairs' `scales`
  ([`VerifiedPairs`](@ref)).
- bundle: `bundle_loss` (stack loss shapes) at the width
  `bundle_efficiency_constant`·σ̂, σ̂ the pooled scale of the kept pairs whose
  SIMFIT++ succeeded, `σ̂² = Σ nₚσₚ² / Σ nₚ` (`nₚ` the pair's inliers: their
  pooled variance), the result's `bundle_sigma`; ONE LM run (`bundle_max_iter`,
  `bundle_ftol`, `bundle_xtol`: [`refine_rotations`](@ref)), the width fixed.
  No kept pair with a scale is an `ArgumentError`. `bundle_whitening = :none`
  (the default) weighs every observation alike, in pixels;
  `:keypoint_covariance` whitens each by its keypoint's position covariance
  ([`FrameFeatures`](@ref)'s `pixel_covariances`, divided by their median
  `tr Σ / 2` over the tracked keypoints, so the whitened residual is in pixels
  of the median keypoint and the width applies; a keypoint the detector did
  not refine takes the identity), the loss acting on the whitened
  residuals. The bundle holds the camera's calibration (`refine_rotations`); the
  retired `bundle_camera_free`, `bundle_scale_multiple` and `local_optimization` are an
  `ArgumentError` naming their successors (the autocalibration of a rotating camera is
  [`refine_camera_and_rotations`](@ref), on the result's rotations and tracks).

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
function register_rotations(frames::AbstractVector, camera::VgtkCore.BoundedCamera; frame_map,
                            tone_percentile::Real, detector_margin_px::Real, asift_tilts, max_detections::Integer,
                            selection::Symbol = :spread, detection_image::Symbol = :tone, detection_noise = nothing,
                            saturated::Union{Nothing, AbstractVector} = nothing,
                            descriptor::Symbol, matcher::Symbol, min_score::Real,
                            pairs::Symbol, max_lag::Integer, overlap_deg::Real, min_inliers::Integer, seed::Integer,
                            scoring = nothing, local_optimization_strategy = nothing, max_trials::Integer, min_trials::Integer,
                            averaging_loss::VgtkCore.AbstractLossShape, cycle_tol::Union{Nothing, Real} = deg2rad(2),
                            orientation_tol::Union{Nothing, Real} = deg2rad(5), bundle_loss::VgtkCore.AbstractLossShape,
                            bundle_efficiency_constant::Real, bundle_max_iter::Integer, bundle_ftol::Real,
                            bundle_xtol::Real, bundle_whitening::Symbol = :none,
                            execution_provider::Symbol = :auto, say = _ -> nothing, retired...)
    _refuse_retired_registration_keywords("register_rotations", retired)
    _check_pair_settings("register_rotations", pairs, max_lag)
    _check_whitening("register_rotations", bundle_whitening)
    _check_tolerances("register_rotations", cycle_tol, orientation_tol)
    vp = verify_pairs(frames, camera; frame_map, tone_percentile, detector_margin_px, asift_tilts, max_detections, selection,
                      detection_image, detection_noise, saturated, descriptor, matcher,
                      min_score, pairs, max_lag, overlap_deg, min_inliers, seed, scoring, local_optimization_strategy, max_trials, min_trials,
                      averaging_loss, execution_provider, say)
    return register_rotations(vp, camera; averaging_loss, cycle_tol, orientation_tol, bundle_loss, bundle_efficiency_constant, bundle_max_iter,
                              bundle_ftol, bundle_xtol, bundle_whitening, say)
end

function register_rotations(vp::VerifiedPairs, camera::VgtkCore.BoundedCamera; averaging_loss::VgtkCore.AbstractLossShape,
                            cycle_tol::Union{Nothing, Real} = deg2rad(2), orientation_tol::Union{Nothing, Real} = deg2rad(5),
                            bundle_loss::VgtkCore.AbstractLossShape, bundle_efficiency_constant::Real,
                            bundle_max_iter::Integer, bundle_ftol::Real, bundle_xtol::Real, bundle_whitening::Symbol = :none,
                            say = _ -> nothing, retired...)
    _refuse_retired_registration_keywords("register_rotations", retired)
    _check_whitening("register_rotations", bundle_whitening)
    feats = vp.features; K = length(feats)
    (; fit, graph) = VgtkEstimation.filtered_rotation_averaging(_view_graph(K, vp.verified); cycle_tol, orientation_tol,
                                                                   optimization_strategy = _averaging_irls(averaging_loss))
    avg = fit
    removed = sort!([p for p in keys(vp.verified) if !haskey(graph, p...)])
    say("  rotation averaging: $(VgtkEstimation.nedges(graph)) edges ($(length(removed)) removed by the view-graph filters" *
        "$(isempty(removed) ? "" : ": " * join(["$(i - 1)-$(j - 1)" for (i, j) in removed], ", "))), σ̂ $(round(rad2deg(avg.sigma); digits = 4))°, $(avg.retcode)")
    ts, dropped = chain_tracks(Dict(p => m for (p, m) in vp.inlier_matches if haskey(graph, p...)))
    say("  tracks: $(length(ts)) ($(sum(length, ts; init = 0)) observations), $dropped inconsistent dropped")
    σ = _pooled_scale(vp, graph)
    say("  scale: σ̂ $(round(σ; digits = 3)) px pooled over the kept pairs' SIMFIT++ scales, bundle width $(round(bundle_efficiency_constant * σ; digits = 3))")
    obs = bundle_whitening === :none ? [[k => feats[k].pixels[kp] for (k, kp) in t] for t in ts] : _uncertain_observations(feats, ts; say)
    bundle = refine_rotations(camera, VgtkCore.estimate(avg), obs; loss = bundle_loss, loss_width = bundle_efficiency_constant * σ,
                              max_iter = bundle_max_iter, ftol = bundle_ftol, xtol = bundle_xtol)
    say("  bundle: $(bundle.n) observations, $(bundle.iterations) iterations, rms $(round(bundle.rms; digits = 3)) px, loss width $(round(bundle.metadata.loss_width; digits = 3)) px, " *
        "$(bundle.retcode), outcome $(bundle.metadata.outcome) ($(bundle.metadata.n_dropped_start) observations dropped at the start: not imaged there)")
    Rs = [RotMatrix{3, Float64, 9}(VgtkCore.estimate(bundle)[k]) for k in 1:K]
    return Registration(Rs, graph, removed, vp.stats, avg, ts, dropped, bundle, σ, feats, vp.loma)
end

# The view graph's averaging is VisualGeometryToolkitEstimation's IRLS strategy on `averaging_loss`, at its default width (the
# loss's efficiency constant at codimension 3 times the fit's σ̂) and its `[fit_rotation_averaging]` iteration cap of record.
const _ESTIMATION_DEFAULTS = VgtkCore.parse_defaults(VgtkEstimation.default_config_path())
_averaging_irls(loss::VgtkCore.AbstractLossShape) = VgtkCore.Irls{typeof(loss)}(
    max_iter = VgtkCore.toml_default(_ESTIMATION_DEFAULTS, "fit_rotation_averaging", "optimization_strategy")["max_iter"])

# The one noise scale of the bundle: the kept pairs' SIMFIT++ scales pooled as a variance weighted by their inlier counts,
# σ̂² = Σ nₚσₚ² / Σ nₚ, over the pairs whose estimate succeeded.
function _pooled_scale(vp::VerifiedPairs, graph)
    ps = [p for p in keys(vp.verified) if haskey(graph, p...) && vp.scales[p].success]
    isempty(ps) && throw(ArgumentError("register_rotations: no kept pair has a noise scale (SIMFIT++ found none): the bundle has no width"))
    n = [vp.verified[p].inliers for p in ps]
    return sqrt(sum(n .* [vp.scales[p].sigma^2 for p in ps]) / sum(n))
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
# `frame => Uncertain(pixel, CovMat(Σ / m))`, Σ the keypoint's camera-pixel covariance (`FrameFeatures`) and m the median over
# the tracks' refined keypoints of tr Σ / 2, so the whitened residual is in pixels of the median keypoint (the bundle's width,
# from the pairs' pixel scale, applies to it). A keypoint the detector did not refine (Σ NaN), or whose Σ is not positive
# definite, takes the identity.
function _uncertain_observations(feats, ts; say = _ -> nothing)
    valid(Σ) = all(isfinite, Σ) && Σ[1, 1] > 0 && Σ[1, 1] * Σ[2, 2] - Σ[1, 2] * Σ[2, 1] > 0
    Σs = [feats[k].pixel_covariances[kp] for t in ts for (k, kp) in t]
    good = filter(valid, Σs)
    isempty(good) && throw(ArgumentError("register_rotations: bundle_whitening = :keypoint_covariance, but no tracked keypoint has a covariance"))
    m = Statistics.median(Σ[1, 1] + Σ[2, 2] for Σ in good) / 2
    say("  bundle whitening: keypoint covariances over their median (tr Σ / 2 = $(round(m; sigdigits = 3)) px²), " *
        "$(length(Σs) - length(good)) of $(length(Σs)) observations take the identity")
    one2 = SMatrix{2, 2, Float64, 4}(I)
    return [[k => VgtkCore.Uncertain(feats[k].pixels[kp], VgtkCore.CovMat(valid(feats[k].pixel_covariances[kp]) ? feats[k].pixel_covariances[kp] / m : one2))
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
    A = VgtkCore.estimate(VgtkEstimation.fit_rotation_averaging(_view_graph(K, verified); optimization_strategy = _averaging_irls(averaging_loss)))
    axis = [SMatrix{3, 3}(A[k]) * SVector(0.0, 0.0, 1.0) for k in 1:K]
    for i in 1:K, j in (i + max_lag + 1):K
        ((i, j) ∉ tried && rad2deg(VgtkCore.angle_between(axis[i], axis[j])) < overlap_deg) && attempt((i, j))
    end
    return nothing
end
