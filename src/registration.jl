# =============================================================================
# register_rotations — the rotations of a pure-rotation sequence
# =============================================================================
#
#   features   per frame (features.jl): ASIFT detections, DeDoDe descriptions
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
VisualGeometryToolkitEstimation's `uncertain_rotation`), its weight the inlier count), `removed` (the verified pairs the filters removed,
sorted), `no_covariance` (the verified pairs whose rotation covariance is not determined:
not edges, sorted), `pairs` (per verified pair
`(i, j)`, kept or removed: matches, inliers, the rms Sampson residual of its inliers in pixels and its rotation angle in
degrees), `averaging` (the averaging fitted on `graph`,
VisualGeometryToolkitEstimation's `fit_filtered_rotation_averaging`: the edges' covariances carry the scale, so it records scale 1,
`ScaleUpstream`; the edges sharing a detection are treated as independent, the first-order approximation), `near_duplicates`
(the kept pairs whose frames share at least `near_duplicate_overlap` of the smaller footprint, VisualGeometryToolkitCore's
`view_overlap` under the averaged rotations: averaged, not chained; sorted), `tracks` (of the kept pairs but
the near-duplicates, as `(frame, detection)` observations, after the track check), `dropped`
(inconsistent tracks dropped by [`chain_tracks`](@ref)), `track_check` (one entry per solve the track check tested: the
observations it `removed` past the cut and that solve's `rms` and `inlier_rms`, px; empty with `check_tracks = false`), `bundle` (the `refine_rotations` result; its `rms` over every kept observation and its metadata's
`inlier_rms` over those within its loss width; its
`retcode`/`stop_rule` (Core's termination record) and its metadata's `n_dropped_start` say how the bundle ended and how many
observations it left out at its start, [`RotationBundleMetadata`](@ref)), `bundle_sigma` (the
pooled noise scale σ̂ of the kept pairs, px, the bundle's `scale`), `features` (the frames' described features, [`detect_features`](@ref): a track's `(frame, detection)` is
detection `detection` of `features[frame]`, in the frame's pixels) and `loma` (what ran LoMa: VisualGeometryToolkitFeatures'
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
    near_duplicates::Vector{Tuple{Int, Int}}
    tracks::Vector{Vector{Tuple{Int, Int}}}
    dropped::Int
    track_check::Vector{NamedTuple{(:removed, :rms, :inlier_rms), Tuple{Int, Float64, Float64}}}
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
inliers as detection index pairs `(a, b)`, detection `a` of frame `i` ↔ detection `b`
of frame `j`: what the tracks consume) and `stats` (per verified pair: matches,
inliers, the rms Sampson residual of its inliers in pixels and its rotation angle in degrees), `scales` (per verified
pair its noise scale: `sigma`, px per coordinate of its Sampson residual, its RANSAC scorer's own (`NaN` when the result
recorded none), and `dof`, the result's degrees of freedom —
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
                 selection = :spread, detection_image = :tone, sensor_models = nothing, noise_sigmas = nothing, saturated = nothing, descriptor, matcher, min_score, pairs, max_lag = nothing, min_inliers, seed,
                 scoring = nothing, local_optimization_strategy = nothing, max_trials, min_trials, averaging_optimization_strategy = <Estimation's>,
                 execution_provider = :auto, say = _ -> nothing) -> VerifiedPairs

The verified frame pairs of the pure-rotation sequence `frames`: the features,
the matching and the pairwise rotations of [`register_rotations`](@ref) (its
keywords of the same names), stopping before the view graph. `max_lag` and
`averaging_optimization_strategy` are read only by `pairs = :lag_overlap` (its
rotation averaging of the verified pairs aims the pairs past the lag), which
needs `max_lag`; giving either with `pairs = :all` is an `ArgumentError`. The
retired `overlap_deg` is refused by name (an `ArgumentError`: `:lag_overlap`
tries every pair past the lag that shares any view). A `pairs` other than `:all` or
`:lag_overlap`, a `max_lag` below 1 with `:lag_overlap`, a detector setting
[`detect_features`](@ref) refuses (`detection_image = :vst` needs every frame a
`LinearRaw`), an `execution_provider` other than `:auto`, `:cpu` or `:cuda`
and, with `:lag_overlap`, a camera whose footprint VisualGeometryToolkitCore's
`view_overlap` refuses (a model that is not a `CameraModel`, a region the lens
cannot back-project, as a bare sensor's corners past a fisheye's fold, a field
of view within 3° of the backward axis: give the camera its lens coverage)
are an `ArgumentError` before any feature is detected.
"""
function verify_pairs(frames::AbstractVector, camera::VgtkCore.BoundedCamera; frame_map,
                      tone_percentile::Real, detector_margin_px::Real, asift_tilts, max_detections::Integer,
                      selection::Symbol = :spread, detection_image::Symbol = :tone,
                      sensor_models::Union{Nothing, AbstractVector{<:VgtkCore.SensorModel}} = nothing, noise_sigmas::Union{Nothing, Real} = nothing,
                      saturated::Union{Nothing, AbstractVector} = nothing, descriptor::Symbol, matcher::Symbol,
                      min_score::Real, pairs::Symbol, max_lag::Union{Nothing, Integer} = nothing,
                      min_inliers::Integer, seed::Integer,
                      scoring = nothing, local_optimization_strategy = nothing, max_trials::Integer, min_trials::Integer,
                      averaging_optimization_strategy = _NotGiven(), execution_provider::Symbol = :auto, say = _ -> nothing, retired...)
    _refuse_retired_registration_keywords("verify_pairs", retired)
    _check_pair_settings("verify_pairs", pairs, max_lag)
    _check_view_overlap("verify_pairs", camera, pairs, 0.0)
    pairs === :all && !(averaging_optimization_strategy isa _NotGiven) &&
        throw(ArgumentError("verify_pairs: averaging_optimization_strategy is read only by pairs = :lag_overlap (its averaging aims " *
                            "the pairs past the lag); leave it out with pairs = :all"))
    _check_frames("verify_pairs", frames, selection, detection_image, sensor_models, noise_sigmas)
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
                            selection, detection_image, sensor_model = sensor_models === nothing ? VgtkCore.SensorModel() : sensor_models[k], noise_sigmas,
                            saturated = saturated === nothing ? nothing : saturated[k])
        say("  frame $(k - 1): $(length(f)) detections kept")
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
        (pairs === :all || j - i <= max_lag::Integer) && verify(i, j)
    end
    pairs === :lag_overlap && _verify_overlap_pairs!(verify, verified, K, max_lag::Integer, camera, averaging_optimization_strategy)
    loma = (; descriptor = VgtkFeatures.loma_provenance(desc), matcher = VgtkFeatures.loma_provenance(match))
    return VerifiedPairs(feats, verified, inlier_matches, stats, scales, loma)
end

# The settings every entry checks before any feature is detected.
# (The execution provider: VisualGeometryToolkitFeatures' `resolve_execution_provider`, which refuses an unknown one.)
# `max_lag` is read only by `pairs = :lag_overlap`, which needs it; with `:all` giving it is refused (no inert setting).
function _check_pair_settings(what, pairs, max_lag)
    pairs in (:all, :lag_overlap) || throw(ArgumentError("$what: pairs = :$pairs is not :all or :lag_overlap"))
    if pairs === :all
        max_lag === nothing || throw(ArgumentError("$what: max_lag is read only by pairs = :lag_overlap; leave it out with pairs = :all"))
    else
        (max_lag !== nothing && max_lag >= 1) || throw(ArgumentError("$what: pairs = :lag_overlap needs max_lag ≥ 1, got $max_lag"))
    end
    return nothing
end
# The detector's settings and, with detection_image = :vst, every frame a LinearRaw: before any feature is detected.
function _check_frames(what, frames, selection, detection_image, sensor_models, noise_sigmas)
    _check_detector_settings(what, selection, detection_image, something(sensor_models, fill(VgtkCore.SensorModel(), length(frames))),
                             noise_sigmas; nframes = length(frames))
    detection_image === :vst && !all(f -> f isa VgtkCore.LinearRaw, frames) &&
        throw(ArgumentError("$what: detection_image = :vst needs every frame a LinearRaw (its green cells' code spans)"))
    return nothing
end

# The registration's retired keys, each with its successor or migration: ONE list for `register_rotations`' and
# `verify_pairs`' keywords and a config's `[registration]` table (the example's, RC's), refused by `refuse_retired_registration`.
const _SENSOR_NOISE_SUCCESSOR = "the frames' noise is their VisualGeometryToolkitCore sensor model (`sensor_models`), resolved from " *
                                "the body's characterisation (sensor_characterisation)"
const _BUNDLE_WIDTH_SUCCESSOR = "the bundle's strategy is `bundle_optimization_strategy` (a config's table of that name): its " *
                                "efficiency constant × the pooled scale"
const _PAIR_SCORER_SUCCESSOR = "the pairs' scorer is Student-t + uniform (`scoring`; the example's outlier_halfwidth_px), no " *
                               "threshold (the MSAC threshold is retired)"
const _RETIRED_REGISTRATION_KEYS = (;
    # 0.6: the frames' noise constants, the bundle's iteration count
    electrons_per_dn = _SENSOR_NOISE_SUCCESSOR,
    read_noise_dn = _SENSOR_NOISE_SUCCESSOR,
    detection_noise = _RETIRED_DETECTION_NOISE,
    bundle_iterations = "the bundle is ONE LM run of its `bundle_optimization_strategy` at the verified pairs' pooled RANSAC " *
                        "noise scale; remove it",
    bundle_camera_free = "the bundle holds the camera's calibration (refine_rotations); the autocalibration of a rotating camera is " *
                         "refine_camera_and_rotations",
    bundle_scale_multiple = _BUNDLE_WIDTH_SUCCESSOR,
    bundle_efficiency_constant = _BUNDLE_WIDTH_SUCCESSOR,
    bundle_loss = "the bundle's robust strategy is `bundle_optimization_strategy = Triggs{Shape}(…)` (a config's table of that " *
                  "name, its loss)",
    averaging_loss = "the averaging's robust strategy is `averaging_optimization_strategy = Irls{Shape}(…)` (a config's table of " *
                     "that name)",
    local_optimization = "the pairs' RANSAC strategy is `local_optimization_strategy` (a config's table of that name)",
    # the retired names in fragments (check_naming lists them)
    Symbol("bundle_" * "whitening") => "the bundle's observations are the detections' pixels (no detection covariance in this " *
                                       "release)",
    Symbol("scoring_threshold" * "_px") => _PAIR_SCORER_SUCCESSOR,
    scoring_threshold = _PAIR_SCORER_SUCCESSOR,
    # 0.8: the optical-axis angles (the overlap is VisualGeometryToolkitCore's view_overlap, the one estimate)
    overlap_deg = "pairs = :lag_overlap tries every pair past the lag whose frames share any view (VisualGeometryToolkitCore's " *
                  "view_overlap under the averaged rotations), min_inliers deciding; no threshold replaces the optical-axis " *
                  "angle: remove it (PanoramaStitching 0.8)",
    near_duplicate_deg = "a near-duplicate pair shares at least `near_duplicate_overlap` of the smaller footprint " *
                         "(VisualGeometryToolkitCore's view_overlap under the averaged rotations; default 0.90, 0 turns the check " *
                         "off), not an optical-axis angle: replace it with near_duplicate_overlap (PanoramaStitching 0.8)")

"""
    refuse_retired_registration(table, where::AbstractString) -> table

Refuse the registration's retired keys in `table`: a config's `[registration]` table (an
`AbstractDict` with `String` keys, as `TOML.parsefile` gives it) or keywords (a `NamedTuple`
or `pairs(kwargs)`, `Symbol` keys). The first retired key found is an `ArgumentError`,
`"<where>: `<key>` is retired: <its successor, or how to migrate>"`; otherwise `table` is
returned as given (any other key is not this function's: `register_rotations` and
`verify_pairs` refuse an unknown keyword themselves, after this check).

The ONE list of PanoramaStitching's retired registration settings, the keywords of
[`register_rotations`](@ref) and [`verify_pairs`](@ref) and a config's keys alike:
`electrons_per_dn`, `read_noise_dn` and `detection_noise` (the frames' sensor model),
`bundle_iterations`, `bundle_camera_free`, `bundle_scale_multiple`,
`bundle_efficiency_constant` and `bundle_loss` (`bundle_optimization_strategy`;
[`refine_camera_and_rotations`](@ref)), `averaging_loss`
(`averaging_optimization_strategy`), `local_optimization` (`local_optimization_strategy`),
the whitening switch, the pairs' MSAC thresholds (the Student-t + uniform `scoring`), and
0.8's `overlap_deg` (`:lag_overlap` tries every pair that shares any view) and
`near_duplicate_deg` (`near_duplicate_overlap`). A config loader calls it on the table it
reads, before any stage, so a config fails the same way cached or not.

```jldoctest
julia> using PanoramaStitching

julia> r = Dict{String, Any}("pairs" => "all", "near_duplicate_overlap" => 0.9);

julia> PanoramaStitching.refuse_retired_registration(r, "config.toml [registration]") === r
true

julia> try
           PanoramaStitching.refuse_retired_registration(Dict("near_duplicate_deg" => 4.0), "config.toml [registration]")
       catch e
           println(first(e.msg, 59))
       end
config.toml [registration]: `near_duplicate_deg` is retired
```
"""
function refuse_retired_registration(table, where::AbstractString)
    for k in keys(table)
        s = Symbol(k)
        haskey(_RETIRED_REGISTRATION_KEYS, s) && throw(ArgumentError("$where: `$k` is retired: $(_RETIRED_REGISTRATION_KEYS[s])"))
    end
    return table
end

# `register_rotations`' and `verify_pairs`' keywords past their own: a retired one refused naming its successor
# (`refuse_retired_registration`), any other as unknown.
function _refuse_retired_registration_keywords(what, kwargs)
    refuse_retired_registration(kwargs, what)
    for (k, _) in pairs(kwargs)
        throw(ArgumentError("$what: unknown keyword `$k`"))
    end
    return nothing
end

"""
    register_rotations(frames, camera::BoundedCamera; frame_map, tone_percentile, detector_margin_px, asift_tilts, max_detections,
                       selection = :spread, detection_image = :tone, sensor_models = nothing, noise_sigmas = nothing, saturated = nothing,
                       descriptor, matcher, min_score, pairs, max_lag = nothing, min_inliers, seed,
                       scoring = nothing, local_optimization_strategy = nothing, max_trials, min_trials,
                       averaging_optimization_strategy = <Estimation's>, cycle_tol = deg2rad(2), orientation_tol = deg2rad(5),
                       bundle_optimization_strategy = Triggs{WelschShape}(…), bundle_max_iter, bundle_ftol,
                       bundle_xtol, near_duplicate_overlap = 0.90, check_tracks = true, execution_provider = :auto,
                       say = _ -> nothing) -> Registration
    register_rotations(pairs::VerifiedPairs, camera::BoundedCamera; frame_map, averaging_optimization_strategy = <Estimation's>,
                       cycle_tol = deg2rad(2), orientation_tol = deg2rad(5), bundle_optimization_strategy = Triggs{WelschShape}(…),
                       bundle_max_iter, bundle_ftol, bundle_xtol, near_duplicate_overlap = 0.90, check_tracks = true,
                       say = _ -> nothing) -> Registration

The rotations of the pure-rotation sequence `frames` (`H × W × 3` linear images in
capture order, `frame_map` their map onto the camera's pixels: VisualGeometryToolkitCore's `raw_frame_map`) seen by the calibrated
`camera` (fixed; a VisualGeometryToolkitCore `BoundedCamera`, whose `imaged_region` bounds the features,
projected in the bundle through its model's `imaging_transform` — a grid lens's `SplineLens` through its forward spline; see
[`refine_rotations`](@ref)), with the view graph, the tracks and the bundle's
statistics. Every setting is the caller's configuration, none defaulted but the
RANSAC's:

- features: inside the camera's `imaged_region`; `tone_percentile`,
  `detector_margin_px`, `asift_tilts` (a Features `tilt_schedule` name, `:imas25`
  or `:untilted`, or its pairs), `max_detections`, `selection`,
  `detection_image`, `sensor_models` (with `:vst`: one VisualGeometryToolkitCore `SensorModel` per frame, its noise a
  `PoissonGaussian`; `nothing`, the absent model of every frame) and `noise_sigmas`, `saturated` (the frames' clip masks,
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
  capture order; then every pair whose frames share any view under the rotation
  averaging of the verified pairs — VisualGeometryToolkitCore's `view_overlap` of
  the frames posed by those rotations, a solid angle above zero — with
  `min_inliers` deciding which are kept; `max_lag` is given with `:lag_overlap`
  only, refused with `:all`); a pair is verified by `find_rotation` (its `scoring` and
  `local_optimization_strategy` objects, `nothing` for VisualGeometryToolkitEstimation's
  defaults; `max_trials`, `min_trials`; the sampler seeded from `seed`) and kept
  with at least `min_inliers` inliers. Its residuals are in the camera's pixels
  ([`pair_bearings`](@ref): plain detections).
- graph: VisualGeometryToolkitEstimation's `fit_filtered_rotation_averaging` of the
  verified pairs' view graph, its strategy `averaging_optimization_strategy` (an `Irls{Shape}`; by default
  VisualGeometryToolkitEstimation's `[fit_rotation_averaging] optimization_strategy`): the pairs off a cycle
  by more than `cycle_tol` removed (`filter_cycles`), the averaging, then the
  pairs off the averaged rotations by more than `orientation_tol` removed
  (`filter_by_orientation`) and the averaging again (radians; `nothing` skips
  that filter). A periodic scene verifies a pair one period off as well as the
  true one, and only the view graph tells them apart. The pairs the filters removed are the
  result's `removed` (a pair with no determined rotation covariance is never an edge: it is
  the result's `no_covariance`, not `removed`); their matches make no track.
- scale: per verified pair its RANSAC scorer's own scale ŝ (Core's `noise_scale` of
  its result: the consensus set's Student-t posterior scale) and the result's degrees
  of freedom, the pairs' `scales` ([`VerifiedPairs`](@ref)), px. No second scale
  estimate.
- bundle: [`refine_rotations`](@ref) with `bundle_optimization_strategy` (a `Triggs{Shape}`, by default VisualGeometryToolkitEstimation's pixel bundle's) at
  `scale` the kept pairs' scales pooled by VisualGeometryToolkitEstimation's
  `pooled_noise_scale` (`σ̂² = Σ νₚŝₚ² / Σ νₚ`, `νₚ` the pair's degrees of freedom;
  the bundle records it as `ScaleUpstream`, its width the strategy's efficiency
  constant × σ̂), σ̂ the result's `bundle_sigma`; ONE LM run (`bundle_max_iter`,
  `bundle_ftol`, `bundle_xtol`: [`refine_rotations`](@ref)), the width fixed.
  No kept pair with a scale is an `ArgumentError`. Every observation is its
  detection's centre in the camera's pixels (through the frames' `frame_map`), the
  residuals in pixels. The bundle holds the camera's calibration (`refine_rotations`); the
  retired `bundle_camera_free`, `bundle_scale_multiple`, the whitening switch and `local_optimization` are an
  `ArgumentError` naming their successors (the autocalibration of a rotating camera is
  [`refine_camera_and_rotations`](@ref), on the result's rotations and tracks).
- near-duplicates: a kept pair whose frames share at least `near_duplicate_overlap` of the smaller footprint — the solid
  angle both image, VisualGeometryToolkitCore's `view_overlap` under the averaged rotations, over the smaller frame's own
  (`max(fraction_i, fraction_j)`): PanoramaStitching's one overlap estimate, as `:lag_overlap`'s — verifies and is averaged
  but seeds no track: two near-identical views match their repeated structure as well as their scene, and a wrong match
  through them chains far. The result's `near_duplicates`. Always on: `near_duplicate_overlap` defaults to 0.90 (the owner's
  value, owner ruling 2026-11-14); `0` turns the check off (every kept pair chained); a value outside `[0, 1]` is an
  `ArgumentError`. The camera's footprint is its `imaged_region` (its lens coverage), and `view_overlap`'s refusals apply
  (a grid model that is not a `CameraModel`, a field of view within 3° of the backward axis).
- track check (`check_tracks = true`): after the bundle, the observations whose residual exceeds 3.035 σ̂ (σ̂ the pooled
  scale above; √χ²₂(0.99): 1 % of the inliers at the true scale) are removed — a track left in fewer than two frames with
  them — and the bundle re-solved from its rotations, until a solve leaves none to remove or 10 solves have removed some
  (COLMAP's `filter_max_reproj_error` loop, Hugin's cpclean). The result's `track_check` holds each tested solve's
  removals, plain rms and inlier rms.

`say` receives one line per frame, pair and step.

The second form starts from the verified pairs ([`verify_pairs`](@ref), the
first form's first step: everything up to the view graph) and their frames'
`frame_map`, so a caller that keeps them reruns the averaging and the bundle alone;
the first form is `register_rotations(verify_pairs(frames, camera; frame_map, …), camera; frame_map, …)`.

A task verb: it composes the estimators above and does none of its own. Each
pair's RANSAC sampler is seeded from `seed` and the pair's frame indices, a
stream of its own per `(seed, i, j)`. A `pairs` other than `:all` or
`:lag_overlap`, a `max_lag` below 1 or none with `:lag_overlap` or one given with `:all`, a `near_duplicate_overlap` outside
`[0, 1]`, the retired `overlap_deg` and `near_duplicate_deg` (PanoramaStitching 0.7's optical-axis angles, refused by name
naming their successors), and an `execution_provider` other
than `:auto`, `:cpu` or `:cuda`, and a camera whose footprint `view_overlap` refuses wherever the registration reads it
(the near-duplicates, unless `near_duplicate_overlap = 0`; `:lag_overlap`'s pairs past the lag) are an `ArgumentError`
before any feature is detected. With either `pairs`, a view graph that is not connected once the
pairs with fewer than `min_inliers` inliers are dropped (its frames have no
common frame; VisualGeometryToolkitEstimation's `require_connected`) is an `ArgumentError` too.
"""
function register_rotations(frames::AbstractVector, camera::VgtkCore.BoundedCamera; frame_map,
                            tone_percentile::Real, detector_margin_px::Real, asift_tilts, max_detections::Integer,
                            selection::Symbol = :spread, detection_image::Symbol = :tone,
                            sensor_models::Union{Nothing, AbstractVector{<:VgtkCore.SensorModel}} = nothing, noise_sigmas::Union{Nothing, Real} = nothing,
                            saturated::Union{Nothing, AbstractVector} = nothing,
                            descriptor::Symbol, matcher::Symbol, min_score::Real,
                            pairs::Symbol, max_lag::Union{Nothing, Integer} = nothing,
                            min_inliers::Integer, seed::Integer,
                            scoring = nothing, local_optimization_strategy = nothing, max_trials::Integer, min_trials::Integer,
                            averaging_optimization_strategy = _NotGiven(), cycle_tol::Union{Nothing, Real} = deg2rad(2),
                            orientation_tol::Union{Nothing, Real} = deg2rad(5), bundle_optimization_strategy = _DEFAULT_BUNDLE_STRATEGY,
                            bundle_max_iter::Integer, bundle_ftol::Real, bundle_xtol::Real,
                            near_duplicate_overlap::Real = _NEAR_DUPLICATE_OVERLAP, check_tracks::Bool = true,
                            execution_provider::Symbol = :auto, say = _ -> nothing, retired...)
    _refuse_retired_registration_keywords("register_rotations", retired)
    _check_pair_settings("register_rotations", pairs, max_lag)
    _check_tolerances("register_rotations", cycle_tol, orientation_tol)
    _check_near_duplicate("register_rotations", near_duplicate_overlap)
    _check_view_overlap("register_rotations", camera, pairs, near_duplicate_overlap)
    vp = verify_pairs(frames, camera; frame_map, tone_percentile, detector_margin_px, asift_tilts, max_detections, selection,
                      detection_image, sensor_models, noise_sigmas, saturated, descriptor, matcher,
                      min_score, pairs, max_lag, min_inliers, seed, scoring, local_optimization_strategy, max_trials, min_trials,
                      averaging_optimization_strategy = pairs === :lag_overlap ? averaging_optimization_strategy : _NotGiven(),
                      execution_provider, say)
    return register_rotations(vp, camera; frame_map, averaging_optimization_strategy, cycle_tol, orientation_tol, bundle_optimization_strategy,
                              bundle_max_iter, bundle_ftol, bundle_xtol, near_duplicate_overlap, check_tracks, say)
end

function register_rotations(vp::VerifiedPairs, camera::VgtkCore.BoundedCamera; frame_map, averaging_optimization_strategy = _NotGiven(),
                            cycle_tol::Union{Nothing, Real} = deg2rad(2), orientation_tol::Union{Nothing, Real} = deg2rad(5),
                            bundle_optimization_strategy = _DEFAULT_BUNDLE_STRATEGY,
                            bundle_max_iter::Integer, bundle_ftol::Real, bundle_xtol::Real,
                            near_duplicate_overlap::Real = _NEAR_DUPLICATE_OVERLAP, check_tracks::Bool = true,
                            say = _ -> nothing, retired...)
    _refuse_retired_registration_keywords("register_rotations", retired)
    _check_near_duplicate("register_rotations", near_duplicate_overlap)
    _check_view_overlap("register_rotations", camera, :all, near_duplicate_overlap)
    feats = vp.features; K = length(feats)
    no_covariance = _no_covariance_pairs(vp.verified)
    isempty(no_covariance) || say("  rotation averaging: $(length(no_covariance)) verified pair(s) with no determined rotation covariance, " *
                                  "not edges: " * join(["$(i - 1)-$(j - 1)" for (i, j) in no_covariance], ", "))
    (; fit, graph) = VgtkEstimation.fit_filtered_rotation_averaging(_view_graph(K, vp.verified); cycle_tol, orientation_tol,
                                                                   _averaging_kw(averaging_optimization_strategy)...)
    avg = fit
    # removed by the filters: the edges of the averaged graph it no longer holds (a pair with no covariance was never an edge)
    removed = sort!([p for p in keys(vp.verified) if p ∉ no_covariance && !haskey(graph, p...)])
    say("  rotation averaging: $(VgtkEstimation.nedges(graph)) edges ($(length(removed)) removed by the view-graph filters" *
        "$(isempty(removed) ? "" : ": " * join(["$(i - 1)-$(j - 1)" for (i, j) in removed], ", "))), edges with their rotation covariances (scale $(VgtkCore.noise_scale(avg)), $(VgtkCore.scale_source(avg))), $(avg.retcode)")
    # the near-duplicate pairs (sharing ≥ near_duplicate_overlap of the smaller footprint under the averaged rotations, the one
    # overlap estimate, Core's view_overlap): still edges of the averaging, their matches seed no track
    A = VgtkCore.estimate(avg)
    near = _near_duplicate_pairs(camera, [A[k] for k in 1:K], [p for p in keys(vp.verified) if haskey(graph, p...)], near_duplicate_overlap)
    isempty(near) || say("  near-duplicate pairs (sharing ≥ $(near_duplicate_overlap) of the smaller footprint): " *
                         join(["$(i - 1)-$(j - 1)" for (i, j) in near], ", ") * ", verified and averaged, not chained")
    ts, dropped = chain_tracks(Dict(p => m for (p, m) in vp.inlier_matches if haskey(graph, p...) && p ∉ near))
    say("  tracks: $(length(ts)) ($(sum(length, ts; init = 0)) observations), $dropped inconsistent dropped")
    # the bundle's scale, upstream: the kept pairs' RANSAC scales pooled by their degrees of freedom (Estimation's
    # pooled_noise_scale), px
    ps = sort!([p for p in keys(vp.verified) if haskey(graph, p...)])
    pooled = VgtkEstimation.pooled_noise_scale([vp.scales[p].sigma for p in ps], [vp.scales[p].dof for p in ps])
    VgtkCore.scale_source(pooled) == VgtkCore.ScaleNone &&
        throw(ArgumentError("register_rotations: no kept pair has a noise scale: the bundle has no width"))
    σ = Float64(VgtkCore.noise_scale(pooled))
    say("  scale: σ̂ $(round(σ; sigdigits = 3)) pooled over $(pooled.metadata.pooled) kept pairs' RANSAC scales, px, the bundle's scale")
    # every observation its detection's centre in the camera's pixels, through the frame map
    pixel(k, kp) = frame_map(SVector{2, Float64}(VgtkCore.center(VgtkCore.feature_frame(feats[k][kp]))))
    bkw = (; optimization_strategy = bundle_optimization_strategy, scale = pooled, ftol = bundle_ftol, xtol = bundle_xtol)
    observations(ts) = [[k => pixel(k, kp) for (k, kp) in t] for t in ts]
    bundle = refine_rotations(camera, A, observations(ts); max_iter = bundle_max_iter, bkw...)
    # the track check: remove the observations past k·σ̂ and re-solve from the last rotations, until none is removed (or the cap)
    history = NamedTuple{(:removed, :rms, :inlier_rms), Tuple{Int, Float64, Float64}}[]
    if check_tracks
        cut = _TRACK_CHECK_K * σ
        for _ in 1:_TRACK_CHECK_CAP
            kept = [[o for (o, r) in zip(t, rt) if !(r > cut)] for (t, rt) in zip(ts, bundle.metadata.residuals)]   # NaN: left out, kept
            kept = filter(t -> length(unique(first.(t))) >= 2, kept)
            nrem = sum(length, ts; init = 0) - sum(length, kept; init = 0)
            push!(history, (; removed = nrem, rms = bundle.rms, inlier_rms = bundle.metadata.inlier_rms))
            say("  track check $(length(history)): $nrem observations past $(round(cut; digits = 3)) px ($(_TRACK_CHECK_K) σ̂) removed; " *
                "bundle rms $(round(bundle.rms; digits = 3)) px, inlier rms $(round(bundle.metadata.inlier_rms; digits = 3)) px")
            nrem == 0 && break
            ts = kept
            bundle = refine_rotations(camera, VgtkCore.estimate(bundle), observations(ts); max_iter = bundle_max_iter, bkw...)
        end
    end
    say("  bundle: $(bundle.n) observations, $(bundle.iterations) iterations, rms $(round(bundle.rms; digits = 3)) px (inliers $(round(bundle.metadata.inlier_rms; digits = 3)) px), loss width $(round(bundle.metadata.loss_width; digits = 3)) px, " *
        "$(VgtkCore.retcode(bundle)) ($(VgtkCore.stop_rule(bundle))), $(bundle.metadata.n_dropped_start) observations dropped at the start (not imaged there)")
    Rs = [RotMatrix{3, Float64, 9}(VgtkCore.estimate(bundle)[k]) for k in 1:K]
    return Registration(Rs, graph, removed, no_covariance, vp.stats, avg, near, ts, dropped, history, bundle, σ, feats, vp.loma)
end

# The averaging's strategy: the caller's (`optimization_strategy`, an Irls or nothing), else VisualGeometryToolkitEstimation's
# own default (`[fit_rotation_averaging] optimization_strategy`), by leaving the keyword out: `_NotGiven()`, the keyword's
# default, is "not given" (Estimation's default applies), never a refusal.
struct _NotGiven end
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

# The near-duplicate threshold of record: a kept pair whose frames share at least this fraction of the smaller footprint
# (VisualGeometryToolkitCore's view_overlap) seeds no track. The owner's value (owner ruling 2026-11-14); the default of
# `near_duplicate_overlap`, whose 0 turns the check off.
const _NEAR_DUPLICATE_OVERLAP = 0.90

# The camera's footprint, checked before any feature is detected: the registration reads VisualGeometryToolkitCore's
# view_overlap through `camera` (the near-duplicates, unless `near_duplicate_overlap` is 0; `:lag_overlap`'s pairs past the
# lag), so its refusals of the footprint (a model that is not a CameraModel, a region the lens cannot back-project, a field
# of view within 3° of the backward axis) come at the entry, not after every pair was matched. Where nothing reads it,
# nothing is refused.
function _check_view_overlap(what, camera, pairs, near_duplicate_overlap)
    readers = String[]
    pairs === :lag_overlap && push!(readers, "pairs = :lag_overlap (its pairs past the lag)")
    near_duplicate_overlap > 0 && push!(readers, "near_duplicate_overlap = $near_duplicate_overlap (the near-duplicate pairs)")
    isempty(readers) && return nothing
    # one frame's footprint (Core's view_overlap builds it, and refuses it, before any pair): a stopgap until Core exposes the
    # footprint check itself
    try
        _shared_views(camera, [one(SMatrix{3, 3, Float64, 9})])
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError("$what: " * join(readers, " and ") * " read VisualGeometryToolkitCore's view_overlap through the " *
                            "camera, which refuses its footprint: " * e.msg))
    end
    return nothing
end

# `near_duplicate_overlap`: a fraction in [0, 1] (0: every kept pair chained). Checked before any feature is detected.
_check_near_duplicate(what, f) = (isfinite(f) && 0 <= f <= 1) ||
    throw(ArgumentError("$what: near_duplicate_overlap = $f is not a fraction in [0, 1] (of the smaller footprint; 0 turns the " *
                        "check off)"))

# The track check's outlier cut and its cap (`register_rotations`, `check_tracks`): an observation whose bundle residual exceeds
# _TRACK_CHECK_K × σ̂ (σ̂ the kept pairs' pooled robust scale, the bundle's) is removed and the bundle re-solved, until none is
# removed or _TRACK_CHECK_CAP solves have removed some. The cut: a residual is 2-D, ‖r‖²/σ² ~ χ²₂ for an inlier, so
# P(‖r‖ > kσ) = exp(−k²/2); k = √χ²₂(0.99) = √9.21 = 3.035 removes 1 % of the inliers at the true σ (COLMAP's
# filter_max_reproj_error and Hugin's cpclean loop, at a stated quantile). The cap: each solve after the first only re-fits
# the observations that remain, so the removals shrink geometrically; 10 bounds a sweep whose removals do not.
const _TRACK_CHECK_K = 3.035
const _TRACK_CHECK_CAP = 10

# A verified pair's rotation covariance (rad², the so(3) chart of R_ij): VisualGeometryToolkitEstimation's
# `uncertain_rotation` of the pair's own RotationProblem and RANSAC result — its scorer's `noise_scale`² × the
# implicit-function covariance of its estimator over the inliers. `nothing` when the inliers do not determine it (a singular covariance): that pair is no edge.
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
# the frames is refused naming the dropped pairs (the averaging of a partial graph would hide them); `check = false` skips that
# refusal (`:lag_overlap`'s joining loop, which may yet bridge the split with a pair past the lag).
function _view_graph(K::Integer, verified::AbstractDict; check::Bool = true)
    edges = Dict(p => VgtkCore.Attributed(VgtkCore.Uncertain(e.rotation, VgtkCore.CovMat(e.covariance)), (; weight = Float64(e.inliers)))
                 for (p, e) in verified if e.covariance !== nothing)
    vg = VgtkEstimation.ViewGraph(collect(1:K), edges)
    dropped = _no_covariance_pairs(verified)
    if check && !isempty(dropped)
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
# disconnected (the rotation averaging refuses it). Then the rotation averaging of every verified pair aims the rest: a pair whose
# frames share any view through `camera` (`_shared_views`: a solid angle above zero) is tried, in (i, j) order; `verify` keeps it
# with `min_inliers` inliers.
function _verify_overlap_pairs!(verify, verified::AbstractDict, K::Integer, max_lag::Integer, camera::VgtkCore.BoundedCamera, averaging_strategy)
    tried = Set{Tuple{Int, Int}}()                               # pairs past the lag: no lag pair comes up again
    attempt(p) = (push!(tried, p); verify(p...))
    while true
        parts = VgtkEstimation.components(_view_graph(K, verified; check = false))
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
    K > max_lag + 1 || return nothing                            # no pair past the lag
    A = VgtkCore.estimate(VgtkEstimation.fit_rotation_averaging(_view_graph(K, verified; check = false); _averaging_kw(averaging_strategy)...))
    shared = _shared_views(camera, [A[k] for k in 1:K])
    for i in 1:K, j in (i + max_lag + 1):K
        ((i, j) ∉ tried && shared[(i, j)] > 0) && attempt((i, j))
    end
    return nothing
end

# PanoramaStitching's ONE spherical view-overlap estimate: VisualGeometryToolkitCore's `view_overlap` of the frames of `camera`
# posed by the rotations `A` (camera → common frame, `d = R_k·f_k`), per pair `(i, j)`, i < j, the solid angle both frames image
# as a fraction of the smaller footprint (`max(fraction_i, fraction_j)`). `:lag_overlap` tries the pairs past the lag that share
# any view; the registration's chaining leaves out the pairs that share at least `near_duplicate_overlap`.
function _shared_views(camera::VgtkCore.BoundedCamera, A::AbstractVector)
    rows = VgtkCore.view_overlap([_oriented_frame(camera, R) for R in A])
    return Dict((r.i, r.j) => max(r.fraction_i, r.fraction_j) for r in rows)
end

# A frame of `camera` turning about its centre, posed by its camera → common-frame rotation `R`: Core's oriented camera, its
# extent and lens coverage the camera's (what `view_overlap` takes).
_oriented_frame(camera::VgtkCore.BoundedCamera, R) =
    VgtkCore.BoundedCamera(VgtkCore.PosedCamera(VgtkCore.model(camera), VgtkCore.relabel(VgtkCore.CameraFrame, VgtkCore.WorldFrame,
                                                                                         VgtkCore.RotationMap(SMatrix{3, 3, Float64, 9}(R)))),
                           VgtkCore.extent(camera); lens_coverage = VgtkCore.lens_coverage(camera))

# The near-duplicate pairs among `candidates` (sorted): those whose frames share at least `threshold` of the smaller footprint
# under the rotations `A` (`_shared_views`); none with `threshold` 0 (the check off).
function _near_duplicate_pairs(camera::VgtkCore.BoundedCamera, A::AbstractVector, candidates, threshold::Real)
    (threshold > 0 && !isempty(candidates)) || return Tuple{Int, Int}[]
    shared = _shared_views(camera, A)
    return sort!([p for p in candidates if shared[p] >= threshold])
end
