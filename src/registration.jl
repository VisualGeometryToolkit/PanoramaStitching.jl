# =============================================================================
# register_rotations — the rotations of a pure-rotation sequence
# =============================================================================
#
#   features   per frame (features.jl): ASIFT keypoints, DeDoDe descriptions
#   pairs      per frame pair: the LoMa matches as unit bearings with their back-projection
#              Jacobians, VisualGeometryToolkitEstimation's `find_rotation` (the RANSAC of the
#              pure rotation on bearings, its scorer and local optimization the caller's)
#   graph      VisualGeometryToolkitEstimation's `ViewGraph` of the verified pairs, each edge its rotation
#              weighted by its inlier count (Core's `Attributed`), and Estimation's `fit_filtered_rotation_averaging`
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
filters kept: per pair `(i, j)` VisualGeometryToolkitCore's `Attributed(Uncertain(R_ij,
CovMat(Σ_R)), (; weight))`, `f_j = R_ij·f_i`, Σ_R its absolute rotation covariance (rad²;
VisualGeometryToolkitEstimation's `uncertain_rotation`, the keypoints' covariance shapes
included), its weight the inlier count), `removed` (the verified pairs the filters removed,
sorted), `no_covariance` (the verified pairs whose rotation covariance is not determined:
not edges, sorted), `pairs` (per verified pair
`(i, j)`, kept or removed: matches, inliers, the rms whitened residual of its inliers — sigmas of the keypoints' unit
noise — and its rotation angle in degrees), `averaging` (the averaging fitted on `graph`,
VisualGeometryToolkitEstimation's `fit_filtered_rotation_averaging`: the edges' covariances carry the scale, so it records scale 1,
`ScaleUpstream`; the edges sharing a keypoint are treated as independent, the first-order approximation), `tracks` (of the kept pairs, as `(frame, keypoint)` observations), `dropped`
(inconsistent tracks dropped), `bundle` (the `refine_rotations` result; its
its `retcode`/`stop_rule` (Core's termination record) and its metadata's `n_dropped_start` say how the bundle ended and how many
observations it left out at its start, [`RotationBundleMetadata`](@ref)), `bundle_sigma` (the
pooled noise scale σ̂ of the kept pairs, the variance factor σ₀ of the keypoints' covariances — the noise σ₀²Σ̃ — and the
bundle's `scale`), `features` (the frames' described features, [`detect_features`](@ref): a track's `(frame, keypoint)` is
keypoint `keypoint` of `features[frame]`, in the frame's pixels) and `loma` (what ran LoMa: VisualGeometryToolkitFeatures'
`loma_provenance` of the descriptor and of the matcher — the models, the ONNX Runtime
provider used and its threads).
"""
struct Registration{G, A, B, F}
    rotations::Vector{RotMatrix{3, Float64, 9}}
    graph::G
    removed::Vector{Tuple{Int, Int}}
    no_covariance::Vector{Tuple{Int, Int}}
    pairs::Dict{Tuple{Int, Int}, NamedTuple{(:matches, :inliers, :rms, :rotation_deg), Tuple{Int, Int, Float64, Float64}}}
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
described features, [`detect_features`](@ref): VisualGeometryToolkitFeatures'
`StructArray{DescribedFeature}`, in the frame's pixels), `verified` (per verified pair `(i, j)`
its rotation `R_ij`, `f_j = R_ij·f_i`, its inlier count and its absolute rotation
`covariance` (rad², the so(3) chart: `uncertain_rotation`; `nothing` when not determined):
what the view graph and the rotation averaging consume), `inlier_matches` (per verified pair its
inliers as keypoint index pairs `(a, b)`, keypoint `a` of frame `i` ↔ keypoint `b`
of frame `j`: what the tracks consume) and `stats` (per verified pair: matches,
inliers, the rms whitened residual of its inliers in sigmas of the keypoints' unit
noise and its rotation angle in degrees), `scales` (per verified pair its noise scale: `sigma`, the variance factor σ₀
of its keypoints' covariances (the noise σ₀²Σ̃) per coordinate of its whitened residual, its RANSAC scorer's own (`NaN` when
the result recorded none), and `dof`, the result's degrees of freedom —
see [`register_rotations`](@ref)) and `loma` (what ran LoMa: VisualGeometryToolkitFeatures'
`loma_provenance` of the descriptor and of the matcher, `(; descriptor,
matcher)`). Everything a registration needs after the features, the
matching and the verification, so a caller may keep it and rerun the averaging
and the bundle alone.
"""
struct VerifiedPairs{F}
    features::Vector{F}
    verified::Dict{Tuple{Int, Int}, NamedTuple{(:rotation, :inliers, :covariance), Tuple{SMatrix{3, 3, Float64, 9}, Int, Union{Nothing, SMatrix{3, 3, Float64, 9}}}}}
    inlier_matches::Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}
    stats::Dict{Tuple{Int, Int}, NamedTuple{(:matches, :inliers, :rms, :rotation_deg), Tuple{Int, Int, Float64, Float64}}}
    scales::Dict{Tuple{Int, Int}, NamedTuple{(:sigma, :dof), Tuple{Float64, Int}}}
    loma::NamedTuple    # (; descriptor, matcher): loma_provenance of each
end
# A verified pair's entry; a `verified` Dict of any compatible entry type (a covariance of every pair) converts to it.
const _VerifiedEntry = NamedTuple{(:rotation, :inliers, :covariance), Tuple{SMatrix{3, 3, Float64, 9}, Int, Union{Nothing, SMatrix{3, 3, Float64, 9}}}}
VerifiedPairs(features::Vector{F}, verified::AbstractDict, args...) where {F} =
    VerifiedPairs{F}(features, Dict{Tuple{Int, Int}, _VerifiedEntry}(verified), args...)

"""
    verify_pairs(frames, camera::BoundedCamera; frame_map, tone_percentile, detector_margin_px, asift_tilts, max_detections,
                 selection = :spread, detection_image = :tone, detection_noise = nothing, saturated = nothing, descriptor, matcher, min_score, pairs, max_lag, overlap_deg, min_inliers, seed,
                 scoring = nothing, local_optimization_strategy = nothing, max_trials, min_trials, averaging_optimization_strategy,
                 execution_provider = :auto, say = _ -> nothing) -> VerifiedPairs

The verified frame pairs of the pure-rotation sequence `frames`: the features,
the matching and the pairwise rotations of [`register_rotations`](@ref) (its
keywords of the same names), stopping before the view graph. `averaging_optimization_strategy` is
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
                      averaging_optimization_strategy = _NotGiven(), execution_provider::Symbol = :auto, say = _ -> nothing, retired...)
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
        say("  frame $(k - 1): $(length(f)) keypoints kept")
        f
    end
    sz(k) = (size(_frame_image(frames[k]), 1), size(_frame_image(frames[k]), 2))                         # the image a frame's features are in: the frame
    verified = Dict{Tuple{Int, Int}, NamedTuple{(:rotation, :inliers, :covariance), Tuple{SMatrix{3, 3, Float64, 9}, Int, Union{Nothing, SMatrix{3, 3, Float64, 9}}}}}()
    inlier_matches = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    stats = Dict{Tuple{Int, Int}, NamedTuple{(:matches, :inliers, :rms, :rotation_deg), Tuple{Int, Int, Float64, Float64}}}()
    scales = Dict{Tuple{Int, Int}, NamedTuple{(:sigma, :dof), Tuple{Float64, Int}}}()
    function verify(i, j)
        ms = match(feats[i], feats[j], sz(i), sz(j); min_score)
        if length(ms) < min_inliers
            say("  pair $(i - 1)-$(j - 1): $(length(ms)) matches → dropped"); return nothing
        end
        cs, kept = pair_bearings(camera, feats[i], feats[j], ms; frame_map)
        kw = (; max_trials, min_trials, rng = _pair_rng(seed, i, j))
        kw = scoring === nothing ? kw : merge(kw, (; scoring))
        kw = local_optimization_strategy === nothing ? kw : merge(kw, (; local_optimization_strategy))
        res = VgtkEstimation.find_rotation(cs; kw...)
        n = res === nothing ? 0 : count(VgtkCore.inlier_mask(res))
        if res === nothing || !VgtkCore.found_model(res) || n < min_inliers
            say("  pair $(i - 1)-$(j - 1): $(length(ms)) matches, $n inliers → dropped"); return nothing
        end
        R = SMatrix{3, 3, Float64, 9}(VgtkCore.estimate(res))
        r = res.residuals[VgtkCore.inlier_mask(res)]
        rms = sqrt(sum(abs2, r) / length(r)); ang = rad2deg(norm(VgtkCore.log_so3(R)))
        verified[(i, j)] = (rotation = R, inliers = n, covariance = _verified_rotation_covariance(cs, res))
        inlier_matches[(i, j)] = kept[VgtkCore.inlier_mask(res)]
        stats[(i, j)] = (matches = length(ms), inliers = n, rms, rotation_deg = ang)
        # the pair's scale: its RANSAC scorer's own ŝ (Core's noise_scale: the consensus set's Student-t posterior scale) and the
        # result's degrees of freedom, what the bundle's pool reads; NaN when the result recorded none
        scales[(i, j)] = (sigma = VgtkCore.scale_source(res) == VgtkCore.ScaleNone ? NaN : Float64(VgtkCore.noise_scale(res)), dof = Int(res.dof))
        say("  pair $(i - 1)-$(j - 1): $(length(ms)) matches, $n inliers, rms $(round(rms; sigdigits = 3)), rotation $(round(ang; digits = 2))°, " *
            "ŝ $(round(scales[(i, j)].sigma; sigdigits = 3)) ($(scales[(i, j)].dof) dof)")
        return nothing
    end
    for i in 1:K, j in (i + 1):K
        (pairs === :all || j - i <= max_lag) && verify(i, j)
    end
    pairs === :lag_overlap && _verify_overlap_pairs!(verify, verified, K, max_lag, overlap_deg, averaging_optimization_strategy)
    loma = (; descriptor = VgtkFeatures.loma_provenance(desc), matcher = VgtkFeatures.loma_provenance(match))
    return VerifiedPairs(feats, verified, inlier_matches, stats, scales, loma)
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

# The registration's retired keywords, refused naming their successor; any other unknown keyword is refused as unknown.
const _RETIRED_REGISTRATION_KEYWORDS = (
    bundle_camera_free = "the bundle holds the camera's calibration (refine_rotations); the autocalibration of a rotating camera is " *
                         "refine_camera_and_rotations",
    bundle_scale_multiple = "the bundle's strategy is `bundle_optimization_strategy` (its efficiency constant × the pooled scale)",
    bundle_efficiency_constant = "the bundle's strategy is `bundle_optimization_strategy` (its efficiency constant × the pooled scale)",
    bundle_loss = "the bundle's robust strategy is `bundle_optimization_strategy = Triggs{Shape}(…)`",
    averaging_loss = "the averaging's robust strategy is `averaging_optimization_strategy = Irls{Shape}(…)`",
    local_optimization = "the pairs' RANSAC strategy is `local_optimization_strategy`",
    bundle_whitening = "the bundle always whitens each observation by its keypoint's covariance, the detector's (Features' " *
                       "FeatureCovariance, through the frames' `frame_map`)")
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
                       averaging_optimization_strategy = <Estimation's>, cycle_tol = deg2rad(2), orientation_tol = deg2rad(5),
                       bundle_optimization_strategy = Triggs{WelschShape}(…), bundle_max_iter, bundle_ftol,
                       bundle_xtol, execution_provider = :auto, say = _ -> nothing) -> Registration
    register_rotations(pairs::VerifiedPairs, camera::BoundedCamera; frame_map, averaging_optimization_strategy = <Estimation's>,
                       cycle_tol = deg2rad(2), orientation_tol = deg2rad(5), bundle_optimization_strategy = Triggs{WelschShape}(…),
                       bundle_max_iter, bundle_ftol, bundle_xtol, say = _ -> nothing) -> Registration

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
  with at least `min_inliers` inliers. Its residuals are whitened by its keypoints'
  covariances ([`pair_bearings`](@ref)): in sigmas of the detection image's unit
  noise, so a scorer's threshold is a number of those sigmas.
- graph: VisualGeometryToolkitEstimation's `fit_filtered_rotation_averaging` of the
  verified pairs' view graph, its strategy `averaging_optimization_strategy` (an `Irls{Shape}`; by default
  VisualGeometryToolkitEstimation's `[fit_rotation_averaging] optimization_strategy`): the pairs off a cycle
  by more than `cycle_tol` removed (`filter_cycles`), the averaging, then the
  pairs off the averaged rotations by more than `orientation_tol` removed
  (`filter_by_orientation`) and the averaging again (radians; `nothing` skips
  that filter). A periodic scene verifies a pair one period off as well as the
  true one, and only the view graph tells them apart. The removed pairs are the
  result's `removed`; their matches make no track.
- scale: per verified pair its RANSAC scorer's own scale ŝ (Core's `noise_scale` of
  its result: the consensus set's Student-t posterior scale) and the result's degrees
  of freedom, the pairs' `scales` ([`VerifiedPairs`](@ref)): the variance factor σ₀ of
  the keypoints' covariances (the noise σ₀²Σ̃, Σ̃ the detector's). No second scale
  estimate.
- bundle: [`refine_rotations`](@ref) with `bundle_optimization_strategy` (a `Triggs{Shape}`, by default VisualGeometryToolkitEstimation's pixel bundle's) at
  `scale` the kept pairs' scales pooled by VisualGeometryToolkitEstimation's
  `pooled_noise_scale` (`σ̂² = Σ νₚŝₚ² / Σ νₚ`, `νₚ` the pair's degrees of freedom;
  the bundle records it as `ScaleUpstream`, its width the strategy's efficiency
  constant × σ̂), σ̂ the result's `bundle_sigma`; ONE LM run (`bundle_max_iter`,
  `bundle_ftol`, `bundle_xtol`: [`refine_rotations`](@ref)), the width fixed.
  No kept pair with a scale is an `ArgumentError`. Every observation is its
  keypoint in the camera's pixels with its covariance, the detector's taken through
  the frames' `frame_map` (VisualGeometryToolkitCore's `propagate`), so the bundle
  whitens each residual by it and the loss acts on the whitened residuals, in the
  pairs' sigmas. The bundle holds the camera's calibration (`refine_rotations`); the
  retired `bundle_camera_free`, `bundle_scale_multiple`, `bundle_whitening` and `local_optimization` are an
  `ArgumentError` naming their successors (the autocalibration of a rotating camera is
  [`refine_camera_and_rotations`](@ref), on the result's rotations and tracks).

`say` receives one line per frame, pair and step.

The second form starts from the verified pairs ([`verify_pairs`](@ref), the
first form's first step: everything up to the view graph) and their frames'
`frame_map`, so a caller that keeps them reruns the averaging and the bundle alone;
the first form is `register_rotations(verify_pairs(frames, camera; frame_map, …), camera; frame_map, …)`.

A task verb: it composes the estimators above and does none of its own. Each
pair's RANSAC sampler is seeded from `seed` and the pair's frame indices, a
stream of its own per `(seed, i, j)`. A `pairs` other than `:all` or
`:lag_overlap`, a `max_lag` below 1 with `:lag_overlap` and an `execution_provider` other
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
                            averaging_optimization_strategy = _NotGiven(), cycle_tol::Union{Nothing, Real} = deg2rad(2),
                            orientation_tol::Union{Nothing, Real} = deg2rad(5), bundle_optimization_strategy = _DEFAULT_BUNDLE_STRATEGY,
                            bundle_max_iter::Integer, bundle_ftol::Real, bundle_xtol::Real,
                            execution_provider::Symbol = :auto, say = _ -> nothing, retired...)
    _refuse_retired_registration_keywords("register_rotations", retired)
    _check_pair_settings("register_rotations", pairs, max_lag)
    _check_tolerances("register_rotations", cycle_tol, orientation_tol)
    vp = verify_pairs(frames, camera; frame_map, tone_percentile, detector_margin_px, asift_tilts, max_detections, selection,
                      detection_image, detection_noise, saturated, descriptor, matcher,
                      min_score, pairs, max_lag, overlap_deg, min_inliers, seed, scoring, local_optimization_strategy, max_trials, min_trials,
                      averaging_optimization_strategy, execution_provider, say)
    return register_rotations(vp, camera; frame_map, averaging_optimization_strategy, cycle_tol, orientation_tol, bundle_optimization_strategy,
                              bundle_max_iter, bundle_ftol, bundle_xtol, say)
end

function register_rotations(vp::VerifiedPairs, camera::VgtkCore.BoundedCamera; frame_map, averaging_optimization_strategy = _NotGiven(),
                            cycle_tol::Union{Nothing, Real} = deg2rad(2), orientation_tol::Union{Nothing, Real} = deg2rad(5),
                            bundle_optimization_strategy = _DEFAULT_BUNDLE_STRATEGY,
                            bundle_max_iter::Integer, bundle_ftol::Real, bundle_xtol::Real,
                            say = _ -> nothing, retired...)
    _refuse_retired_registration_keywords("register_rotations", retired)
    feats = vp.features; K = length(feats)
    no_covariance = _no_covariance_pairs(vp.verified)
    isempty(no_covariance) || say("  rotation averaging: $(length(no_covariance)) verified pair(s) with no determined rotation covariance, " *
                                  "not edges: " * join(["$(i - 1)-$(j - 1)" for (i, j) in no_covariance], ", "))
    (; fit, graph) = VgtkEstimation.fit_filtered_rotation_averaging(_view_graph(K, vp.verified); cycle_tol, orientation_tol,
                                                                   _averaging_kw(averaging_optimization_strategy)...)
    avg = fit
    removed = sort!([p for p in keys(vp.verified) if !haskey(graph, p...)])
    say("  rotation averaging: $(VgtkEstimation.nedges(graph)) edges ($(length(removed)) removed by the view-graph filters" *
        "$(isempty(removed) ? "" : ": " * join(["$(i - 1)-$(j - 1)" for (i, j) in removed], ", "))), edges with their rotation covariances (scale $(VgtkCore.noise_scale(avg)), $(VgtkCore.scale_source(avg))), $(avg.retcode)")
    ts, dropped = chain_tracks(Dict(p => m for (p, m) in vp.inlier_matches if haskey(graph, p...)))
    say("  tracks: $(length(ts)) ($(sum(length, ts; init = 0)) observations), $dropped inconsistent dropped")
    # the bundle's scale, upstream: the kept pairs' RANSAC scales pooled by their degrees of freedom (Estimation's
    # pooled_noise_scale), σ₀ of the keypoints' covariances
    ps = sort!([p for p in keys(vp.verified) if haskey(graph, p...)])
    pooled = VgtkEstimation.pooled_noise_scale([vp.scales[p].sigma for p in ps], [vp.scales[p].dof for p in ps])
    VgtkCore.scale_source(pooled) == VgtkCore.ScaleNone &&
        throw(ArgumentError("register_rotations: no kept pair has a noise scale: the bundle has no width"))
    σ = Float64(VgtkCore.noise_scale(pooled))
    say("  scale: σ̂ $(round(σ; sigdigits = 3)) pooled over $(pooled.metadata.pooled) kept pairs' RANSAC scales (σ₀ of the keypoints' covariances), the bundle's scale")
    # every observation its keypoint in the camera's pixels with its covariance (the bundle whitens by it)
    obs = [[k => _camera_keypoint(frame_map, feats[k][kp].detection) for (k, kp) in t] for t in ts]
    bundle = refine_rotations(camera, VgtkCore.estimate(avg), obs; optimization_strategy = bundle_optimization_strategy, scale = pooled,
                              max_iter = bundle_max_iter, ftol = bundle_ftol, xtol = bundle_xtol)
    say("  bundle: $(bundle.n) observations, $(bundle.iterations) iterations, rms $(round(bundle.rms; digits = 3)) px, loss width $(round(bundle.metadata.loss_width; sigdigits = 3)) (whitened), " *
        "$(VgtkCore.retcode(bundle)) ($(VgtkCore.stop_rule(bundle))), $(bundle.metadata.n_dropped_start) observations dropped at the start (not imaged there)")
    Rs = [RotMatrix{3, Float64, 9}(VgtkCore.estimate(bundle)[k]) for k in 1:K]
    return Registration(Rs, graph, removed, no_covariance, vp.stats, avg, ts, dropped, bundle, σ, feats, vp.loma)
end

# The averaging's strategy: the caller's (`optimization_strategy`, an Irls or nothing), else VisualGeometryToolkitEstimation's
# own default (`[fit_rotation_averaging] optimization_strategy`), by leaving the keyword out.
_averaging_kw(::_NotGiven) = (;)
_averaging_kw(strategy) = (; optimization_strategy = strategy)

# The view-graph filters' tolerances (radians): positive, or nothing (that filter skipped). Checked before any feature is detected;
# from the verified pairs, VisualGeometryToolkitEstimation's `fit_filtered_rotation_averaging` refuses them itself.
function _check_tolerances(what, cycle_tol, orientation_tol)
    for (name, t) in (("cycle_tol", cycle_tol), ("orientation_tol", orientation_tol))
        t === nothing || (isfinite(t) && t > 0) || throw(ArgumentError("$what: $name = $t is not positive (radians; nothing skips the filter)"))
    end
    return nothing
end

# A verified pair's rotation covariance (rad², the so(3) chart of R_ij): VisualGeometryToolkitEstimation's
# `uncertain_rotation` of the pair's own RotationProblem and RANSAC result — its scorer's `noise_scale`² × the
# implicit-function covariance of its estimator over the inliers, the keypoints' covariance shapes included (the problem's
# `Σ₁`/`Σ₂` columns). `nothing` when the inliers do not determine it (a singular covariance): that pair is no edge.
function _verified_rotation_covariance(cs, res)
    u = try
        VgtkEstimation.uncertain_rotation(VgtkEstimation.RotationProblem(cs), res)
    catch e
        e isa ArgumentError || rethrow()
        return nothing
    end
    return SMatrix{3, 3, Float64, 9}(Statistics.cov(u))
end

# The verified pairs whose rotation covariance is not determined: not edges of the averaging (sorted).
_no_covariance_pairs(verified) = sort!([p for (p, e) in verified if e.covariance === nothing])

# The view graph of the frames 1:K and the verified pairs: VisualGeometryToolkitEstimation's `ViewGraph`, each edge the pair's
# rotation `R_ij` (`f_j = R_ij·f_i`) with its absolute covariance, weighted by its inlier count:
# `Attributed(Uncertain(R_ij, CovMat(Σ_R)), (; weight))`. A pair with no determined covariance is no edge; a drop that splits
# the frames is refused naming the dropped pairs (the averaging of a partial graph would hide them).
function _view_graph(K::Integer, verified::AbstractDict)
    edges = Dict(p => VgtkCore.Attributed(VgtkCore.Uncertain(e.rotation, VgtkCore.CovMat(e.covariance)), (; weight = Float64(e.inliers)))
                 for (p, e) in verified if e.covariance !== nothing)
    vg = VgtkEstimation.ViewGraph(collect(1:K), edges)
    dropped = _no_covariance_pairs(verified)
    if !isempty(dropped)
        all_pairs = VgtkEstimation.ViewGraph(collect(1:K), Dict(p => VgtkCore.Attributed(e.rotation, (; weight = 1.0)) for (p, e) in verified))
        length(VgtkEstimation.components(vg)) > length(VgtkEstimation.components(all_pairs)) && throw(ArgumentError(
            "register_rotations: dropping the pair(s) with no determined rotation covariance " *
            join(["$(i - 1)-$(j - 1)" for (i, j) in dropped], ", ") * " disconnects the view graph"))
    end
    return vg
end

# The RANSAC sampler of the pair (i, j): Core's `stable_seed` keyed by the pair's frame indices (a SHA-256 of the integers'
# decimal text, so no two (seed, i, j) share a stream) seeding a StableRNG (its stream fixed across Julia versions).
_pair_rng(seed::Integer, i::Integer, j::Integer) = StableRNG(VgtkCore.stable_seed(seed, i, j))

# The pairs of `pairs = :lag_overlap` past the lag, after the lag pairs: `verify(i, j)` adds a pair it keeps to `verified`.
# While the verified pairs leave the frames in more than one part (a lag pair dropped), the untried pair between two parts
# nearest in capture order is tried, since no verified rotation relates their frames; when no such pair is left the graph stays
# disconnected (the rotation averaging refuses it). Then the rotation averaging of every verified pair aims the rest: a pair whose optical
# axes are within `overlap_deg` is tried, in (i, j) order.
function _verify_overlap_pairs!(verify, verified::AbstractDict, K::Integer, max_lag::Integer, overlap_deg::Real, averaging_strategy)
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
    A = VgtkCore.estimate(VgtkEstimation.fit_rotation_averaging(_view_graph(K, verified); _averaging_kw(averaging_strategy)...))
    axis = [SMatrix{3, 3}(A[k]) * SVector(0.0, 0.0, 1.0) for k in 1:K]
    for i in 1:K, j in (i + max_lag + 1):K
        ((i, j) ∉ tried && rad2deg(VgtkCore.angle_between(axis[i], axis[j])) < overlap_deg) && attempt((i, j))
    end
    return nothing
end
