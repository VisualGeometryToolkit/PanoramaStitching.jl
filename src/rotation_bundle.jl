# =============================================================================
# The rotation bundle — the frames' rotations and the tracks' directions
# =============================================================================
#
# A camera that only rotated sees a distant point at one direction d_j of the
# common frame from every frame k: x_jk = π(R_kᵀ·d_j), π the calibrated camera's
# projection (fixed), R_k frame k's rotation (d = R_k·f_k). `refine_rotations`
# minimizes the robust sum of the pixel reprojection errors of every observation
# of every track over the rotations of every frame but the first (the gauge)
# and the directions of the tracks: an NLLSsolver problem with one
# VisualGeometryToolkitCore `SO3Manifold` variable per frame and one `SphereManifold`
# (2-dof) direction per track, the directions ELIMINABLE (their Schur complement,
# NLLSsolver's #60 on the stack's branch; the registered NLLSsolver solves the
# same problem by sparse LDL), one 2-row residual block per observation with its
# closed-form Jacobian, the stack's loss shape as NLLSsolver's robustifier
# (VisualGeometryToolkitCore `loss_robustifier`) — the pattern of Estimation's
# relative-pose bundle.
#
# The camera optionally free (`camera_free`, the self-calibration of a rotating
# camera, Hartley 1994; Agapito, Hayman and Reid 2001): one more variable,
# VisualGeometryToolkitCore's `ParameterManifold` of the camera's model
# restricted to the named parameters (the camera block of Estimation's
# `refine_camera_and_poses`), on which every observation then also depends. With
# the first frame's rotation the gauge, a rotating sequence leaves nothing else
# free; the bundle's metadata carries the reduced (Schur) information of the
# rotations and the camera with which to check it.
# =============================================================================

const _FrameChart = VgtkCore.SO3Manifold{Float64}
const _SphereChart = typeof(VgtkCore.SphereManifold(SVector(0.0, 0.0, 1.0)))

# A track's direction is a 2-dof sphere chart held as Core's `Eliminable{E}`: `E` says whether NLLSsolver may eliminate it
# (only where the installed NLLSsolver has the Schur system).
const _SCHUR = VgtkCore.nlls_features().schur

# The camera the bundle reads (`_BundleCamera`): its model (the back-projection and its Jacobian) and where that model's
# projection is defined, its `valid_domain` (a grid lens's lattice; the whole plane for a lens without a limit), computed once.
# Not the lens coverage: the coverage says where features and samples are taken, not where a residual exists — so a parametric
# camera's residual is never masked (as in 0.5.1) and a grid lens's is NaN off its lattice.
struct _BundleCamera{M, D}
    model::M
    domain::D
end
_BundleCamera(camera) = (m = VgtkCore.model(camera); _BundleCamera(m, VgtkCore.valid_domain(m)))

# One observation: √w (π(R_kᵀ d_j) − x_jk), pixels, through the fixed camera: `projection` its forward map π (`_forward_map`),
# `camera` its `_BundleCamera` (the Jacobian's back-projection and the domain). `whitening` is `nothing` (the pixel residual) or L⁻¹ of the observation's
# pixel covariance Σ = LLᵀ (the residual L⁻¹(π − x) and its Jacobian L⁻¹J, in units of the covariance's unit noise).
struct _Observation{E, C, P, L, W} <: NLLSsolver.AbstractResidual
    camera::C
    projection::P
    x::SVector{2, Float64}
    varind::SVector{2, Int}
    kernel::L
    whitening::W
end
NLLSsolver.ndeps(::_Observation) = static(2)
NLLSsolver.nres(::_Observation) = static(2)
NLLSsolver.varindices(r::_Observation) = r.varind
NLLSsolver.getvars(r::_Observation{E}, vars::Vector) where {E} = (vars[r.varind[1]]::_FrameChart, vars[r.varind[2]]::VgtkCore.Eliminable{E, _SphereChart})
NLLSsolver.robustkernel(r::_Observation) = r.kernel
Base.eltype(::_Observation) = Float64

@inline _camera_direction(ch::VgtkCore.RotationChart, d::VgtkCore.Eliminable) = SMatrix{3, 3}(ch.R)' * VgtkCore.model(d.value)

# π(f): the forward map in closed form, NaN where it does not image AND off the model's valid domain (a grid lens's lattice),
# where its back-projection Jacobian is NaN. A forward map can image past that domain (a finite pixel), so without the second
# test the residual would be finite where its Jacobian
# is not: NLLSsolver would accept such an iterate, linearize there and loop on its NaN system for ever (measured 2026-11-04,
# the synthetic Tamron sequence from its true rotations). With it the residual and its Jacobian are defined on the same set,
# an iterate past it is a NaN cost (a rejected step), and an observation whose start prediction is past it is dropped
# (`refine_rotations`, counted in `n_dropped_start`).
@inline function _project(r::_Observation, f)
    u = r.projection(f); p = SVector(u[1], u[2])
    return VgtkCore.Point2(p[1], p[2]) ∈ r.camera.domain ? p : SVector(NaN, NaN)
end

# An observation's pixel and whitening: a bare pixel has none; an `Uncertain` pixel's is Core's `whitening_factor` of the
# residual π − x (its measurement Jacobian −I, so L⁻¹ of the covariance Σ = LLᵀ), refused unless Σ is finite and positive
# definite (Core's factor is NaN or zero then).
_observed_pixel(x) = x
_observed_pixel(x::VgtkCore.Uncertain) = VgtkCore.geom(x)
_observation_whitening(x) = nothing
function _observation_whitening(x::VgtkCore.Uncertain)
    Σ = SMatrix{2, 2, Float64, 4}(Statistics.cov(x))
    Li = VgtkCore.whitening_factor(-SMatrix{2, 2, Float64, 4}(I), Σ)
    (all(isfinite, Li) && !iszero(Li)) || throw(ArgumentError("refine_rotations: an observation's covariance $Σ is not finite and positive definite"))
    return Li
end

# The whitening of a residual or its Jacobian: none, or L⁻¹ (the observation's pixel covariance Σ = LLᵀ).
@inline _whiten(::Nothing, x) = x
@inline _whiten(Li::SMatrix{2, 2, Float64, 4}, x) = Li * x

# The pixel residual π(R_kᵀ d_j) − x_jk (unwhitened: the residual norms the metadata reports).
@inline _pixel_residual(r::_Observation, ch::VgtkCore.RotationChart, d::VgtkCore.Eliminable) = _project(r, _camera_direction(ch, d)) - r.x

NLLSsolver.computeresidual(r::_Observation, ch::VgtkCore.RotationChart, d::VgtkCore.Eliminable) = _whiten(r.whitening, _pixel_residual(r, ch, d))

# One observation of the self-calibrating bundle: as `_Observation`, its camera the third variable (`varind` = frame,
# direction, camera), the model's `ParameterManifold` restricted to the free parameters, its projection and its valid domain
# read from that variable's model.
struct _CameraObservation{E, L, W} <: NLLSsolver.AbstractResidual
    x::SVector{2, Float64}
    varind::SVector{3, Int}
    kernel::L
    whitening::W
end
NLLSsolver.ndeps(::_CameraObservation) = static(3)
NLLSsolver.nres(::_CameraObservation) = static(2)
NLLSsolver.varindices(r::_CameraObservation) = r.varind
NLLSsolver.getvars(r::_CameraObservation{E}, vars::Vector) where {E} =
    (vars[r.varind[1]]::_FrameChart, vars[r.varind[2]]::VgtkCore.Eliminable{E, _SphereChart}, vars[r.varind[3]])
NLLSsolver.robustkernel(r::_CameraObservation) = r.kernel
Base.eltype(::_CameraObservation) = Float64

# π(f) of the camera model `m`, NaN off its valid domain (as `_project`); `_image` without the test (the camera block's
# derivative, taken where the test holds).
@inline _image(m, f) = (u = VgtkCore.imaging_transform(m)(f); SVector(u[1], u[2]))
@inline function _project_model(m, f)
    p = _image(m, f)
    return VgtkCore.Point2(p[1], p[2]) ∈ VgtkCore.valid_domain(m) ? p : SVector(NaN, NaN)
end
@inline _pixel_residual(r::_CameraObservation, ch::VgtkCore.RotationChart, d::VgtkCore.Eliminable, c) =
    _project_model(VgtkCore.model(c), _camera_direction(ch, d)) - r.x
NLLSsolver.computeresidual(r::_CameraObservation, ch::VgtkCore.RotationChart, d::VgtkCore.Eliminable, c) =
    _whiten(r.whitening, _pixel_residual(r, ch, d, c))

# The block's cost as the LM reads it: non-finite (an iterate past the model's valid domain) is +Inf (Core's `finite_cost`, the
# stack's rule for every residual block): a rejected step, also where the step is below `xtol` (NLLSsolver accepts any step
# below it, and a NaN cost accepted there fails its callback's cost check).
NLLSsolver.computecost(vars::AbstractVector, r::Union{_Observation, _CameraObservation}) =
    VgtkCore.finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))

# Its closed-form frame and direction blocks (those of `_Observation`, through the variable's model) and its camera block,
# ForwardDiff of π along the camera chart's coordinates at its anchor. `varflags`: bit 1 the frame, bit 2 the direction, bit 3
# the camera; only the free variables' columns are returned, in that order.
function NLLSsolver.computeresjac(varflags, r::_CameraObservation, ch::_FrameChart, d::VgtkCore.Eliminable, c)
    m = VgtkCore.model(c)
    R = SMatrix{3, 3, Float64, 9}(ch.R); dir = VgtkCore.model(d.value); f = R' * dir
    u = _project_model(m, f)
    _, M = VgtkCore.backproject_with_jacobian(m, u)
    Mp = _whiten(r.whitening, (M' * M) \ M')
    Jω = Mp * (R' * VgtkCore.skew(dir))
    Jv = Mp * (R' * VgtkCore.retract_jacobian(d.value))
    z0 = zero(SVector{VgtkCore.tangent_size(c), Float64})
    Jc = _whiten(r.whitening, ForwardDiff.jacobian(z -> _image(VgtkCore.model(VgtkCore.retract(c, z)), f), z0))
    e = _whiten(r.whitening, u - r.x)
    varflags == 7 && return e, hcat(Jω, Jv, Jc)
    varflags == 6 && return e, hcat(Jv, Jc)
    varflags == 5 && return e, hcat(Jω, Jc)
    varflags == 4 && return e, Jc
    varflags == 3 && return e, hcat(Jω, Jv)
    return varflags == 2 ? (e, Jv) : (e, Jω)
end

# The closed-form 2 × 5 Jacobian at the variables' anchors: ∂π/∂f on the tangent plane is the pseudo-inverse of the
# camera's back-projection Jacobian M at π(f) (`backproject_with_jacobian`), f = R_kᵀ·exp(−ω)·d moves by R_kᵀ[d]×·dω
# and by R_kᵀ·∂d/∂v along the direction's chart.
# `varflags` (NLLSsolver's): bit 1 the frame, bit 2 the direction — only the free variables' columns are returned (the
# gauge frame is fixed). A whitened observation returns L⁻¹e and L⁻¹J (L fixed: the residual's exact Jacobian).
function NLLSsolver.computeresjac(varflags, r::_Observation, ch::_FrameChart, d::VgtkCore.Eliminable)
    R = SMatrix{3, 3, Float64, 9}(ch.R); dir = VgtkCore.model(d.value); f = R' * dir
    u = _project(r, f)
    _, M = VgtkCore.backproject_with_jacobian(r.camera.model, u)
    Mp = _whiten(r.whitening, (M' * M) \ M')
    Jω = Mp * (R' * VgtkCore.skew(dir))
    Jv = Mp * (R' * VgtkCore.retract_jacobian(d.value))
    e = _whiten(r.whitening, u - r.x)
    varflags == 3 && return e, hcat(Jω, Jv)
    return varflags == 2 ? (e, Jv) : (e, Jω)
end

"""
    RotationBundleMetadata

`FitResult` metadata of [`refine_rotations`](@ref): `directions` (one unit
direction per track, the common frame), `residuals` (per track, the pixel
reprojection error norm of each of its observations at the result, unwhitened),
`loss_width` (the given width: px, or in whitened units when the
observations carry covariances), `n_observations` (given), `n_dropped_start`
(the observations left out of the solve: those whose residual or Jacobian is
not finite at the start — a start prediction off the camera model's valid
domain, where it has no projection and no back-projection Jacobian, as for a
grid lens calibrated over the sensor whose frames' observations reach its
corners from a start a fraction of a degree off, the Tamron 20 mm of
2026-10-29 — and the rest of a track left in fewer than two frames; a
dropped observation's residual is NaN, and a track left out keeps its start
direction), `frames_unconstrained` (the frames other than the gauge left with
no kept observation: each is held at its start rotation), `outcome`, the
bundle's outcome:

- `:not_run` — `max_iter = 0` (the start's rotations are the result);
- `:converged` — the retcode is `:converged`;
- otherwise the retcode itself (`:max_iterations`, `:stalled`, `:solve_failed`, …);

`camera` (the camera at the result: the one given when it is held, else the
refined one, of the given camera's type and, for a `BoundedCamera`, its sensor
and lens coverage) and `calibration`, `nothing` when the camera is held, else
the self-calibration's information at the result (`camera_free`):

- `free` — the names of the camera parameters refined;
- `parameters`, `sigma`, `covariance` — their values, standard deviations and
  covariance (parameter units: pixels for `f`, `cx`, `cy`; the lens's own for its
  coefficients) — `s²·H⁻¹`, `H` the Gauss–Newton information of the robust
  bundle (each observation weighted by its loss's IRLS weight at the result)
  reduced to the rotations and the camera by the Schur complement of the track
  directions, the gauge frame removed, mapped from the camera chart to the
  parameters; `s² = Σ wᵢ‖rᵢ‖² / ν`, `ν = 2n − p` (`n` observations kept, `p`
  unknowns: `3(K − 1)` rotations, 2 per kept track, the free parameters);
- `s2`, `nu` — `s²` and `ν`;
- `reduced_eigenvalues` — the eigenvalues (ascending) of that reduced
  information with the gauge frame INCLUDED, its rows and columns scaled to a
  unit diagonal: three are zero to rounding (a common rotation of every frame
  and every direction is the problem's one symmetry) and the next is the
  smallest of `reduced_eigenvalues_gauge_fixed`, the same with the gauge frame
  removed — positive when nothing else is free;
- `direction_min_eigenvalue` — the least eigenvalue over the kept tracks of
  their 2 × 2 direction blocks (each positive: every kept direction observed).
"""
struct RotationBundleMetadata{K, C, I}
    directions::Vector{SVector{3, Float64}}
    residuals::Vector{Vector{Float64}}
    loss_width::Float64
    n_observations::Int
    n_dropped_start::Int
    frames_unconstrained::Vector{K}
    outcome::Symbol
    camera::C
    calibration::I
end

"""
    refine_rotations(camera, rotations, tracks; loss, loss_width, max_iter, ftol, xtol,
                     eliminate = true, camera_free = ()) -> FitResult{Dict{K, RotMatrix{3, Float64}}, RotationBundleMetadata}

The rotation bundle of a pure-rotation sequence seen by the calibrated `camera`
(fixed; its `backproject` and `backproject_with_jacobian` give the start and
the Jacobian; a `CameraModel` or a `BoundedCamera`), projected through its
closed-form forward map π, its model's `imaging_transform` (a grid lens's
`SplineLens` through its forward spline, VisualGeometryToolkitEstimation's
`fit_camera(SplineLens, camera, imaged_region(camera); tolerance)`). A residual is NaN off
the camera model's `valid_domain` (a grid lens's lattice), where its
back-projection Jacobian is undefined; the lens coverage does not enter the
bundle. The
rotations `R_k` (`d = R_k·f_k`) of every frame but the first of
the sorted keys of `rotations` (the gauge, held fixed) and one direction `d_j`
per track minimizing the robust sum of the pixel reprojection errors
`π(R_kᵀ d_j) − x_jk` of every observation. A refine: it starts from
`rotations` (a rotation averaging of the view graph,
`VisualGeometryToolkitEstimation.fit_rotation_averaging`) and each track's mean
bearing in the common frame.

An observation whose start prediction is not imaged (its residual or Jacobian
not finite at the start: the prediction off the model's valid domain) is
dropped, a track left in fewer than two frames is dropped whole (it keeps its
start direction, held), and a partly kept track restarts from the mean bearing
of the observations it keeps, its observations tested again until none is
dropped: the solve is that of the kept observations alone. A frame other than
the gauge left with no kept observation is held at its start rotation. The
result's `n` is the number of observations kept; its metadata's
`n_observations` the number given and `n_dropped_start` their difference
([`RotationBundleMetadata`](@ref)).

- `tracks` — `AbstractVector` of tracks, each an `AbstractVector` of `frame =>
  pixel` observations (pixels in the camera's frame), two frames at least. A
  pixel may be VisualGeometryToolkitCore's `Uncertain(pixel, CovMat(Σ))`, its
  2 × 2 covariance `Σ = LLᵀ` (finite, positive definite, up to one global
  scale): the residual is then whitened, `L⁻¹(π(R_kᵀ d_j) − x_jk)`, and the
  loss acts on the whitened residual. Every observation carries a covariance
  or none does.
- `loss`, `loss_width` — the stack loss shape (NLLSsolver's robustifier,
  `loss_robustifier`) and its width (pixels, or whitened units; positive and
  finite), fixed for the bundle's ONE NLLSsolver run: the caller's, from the
  data's noise scale (`register_rotations`: the verified pairs' pooled
  SIMFIT++ scale). `max_iter`, `ftol` (the relative cost decrease) and `xtol`
  (the largest step in chart units: radians for a rotation or a direction)
  are its stops (`nlls_options`); the result's `iterations` the LM iterations
  it took.
- `eliminate` — whether NLLSsolver may eliminate the directions (the Schur
  complement, where the loaded NLLSsolver has it); `false` solves every unknown
  together. Both reach the same optimum.
- `camera_free` — the camera parameters refined with the rotations and the
  directions, by name (VisualGeometryToolkitCore's `parameter_names` of the
  camera's model: `:f, :aspect, :skew, :cx, :cy`, then the lens's
  coefficients); `()` (the default) holds the camera fixed, the bundle above.
  Free, the camera is one more variable, Core's `ParameterManifold` of its model
  restricted to those parameters (the camera block of Estimation's
  `refine_camera_and_poses`), its Jacobian ForwardDiff through the chart; the
  rotating camera's self-calibration. The first frame's rotation stays the one
  gauge (the metadata's `calibration` carries the reduced information that
  shows nothing else is free, and the parameters' covariance). A parametric
  camera only: a SplineLens camera, or a name its model does not have, is an
  `ArgumentError`. The observations kept at the start are those the GIVEN
  camera images; the residuals, the loss width and the result are the
  refined camera's.

A refine ends in the local minimum its start leads to; the result's `retcode`
and its metadata's `outcome` ([`RotationBundleMetadata`](@ref)) say how the
bundle ended. A track observed in fewer than two frames, or naming a frame
without a start rotation, some observations with a covariance and some
without, a covariance that is not finite and positive definite, and a start
that keeps no observation (every track left in fewer than two imaged frames)
are an `ArgumentError`.
"""
function refine_rotations(camera, rotations::AbstractDict{K}, tracks::AbstractVector;
                          loss::VgtkCore.AbstractLossShape, loss_width::Real,
                          max_iter::Integer, ftol::Real, xtol::Real, eliminate::Bool = true,
                          camera_free::Tuple{Vararg{Symbol}} = ()) where {K}
    isfinite(loss_width) && loss_width > 0 ||
        throw(ArgumentError("refine_rotations: the loss width must be positive and finite, got $loss_width"))
    chc = _camera_chart(camera, camera_free)                     # nothing: the camera held
    fwd = _forward_map(camera); bc = _BundleCamera(camera); C = typeof(bc); P = typeof(fwd)
    frames = sort!(collect(keys(rotations)))
    fid = Dict(k => v for (v, k) in enumerate(frames))
    E = eliminate && _SCHUR
    V = chc === nothing ? Union{_FrameChart, VgtkCore.Eliminable{E, _SphereChart}} :
        Union{_FrameChart, VgtkCore.Eliminable{E, _SphereChart}, typeof(chc)}
    nobs = 0; nuncertain = 0
    for t in tracks
        length(unique(first.(t))) >= 2 || throw(ArgumentError("refine_rotations: a track needs observations in two frames"))
        all(o -> haskey(fid, first(o)), t) || throw(ArgumentError("refine_rotations: a track names a frame with no start rotation"))
        nobs += length(t); nuncertain += count(o -> last(o) isa VgtkCore.Uncertain, t)
    end
    0 < nuncertain < nobs && throw(ArgumentError("refine_rotations: $nuncertain of $nobs observations carry a covariance; " *
                                                 "every observation or none must"))
    # per observation: its pixel and its whitening (nothing, or L⁻¹ of its covariance)
    pixel(o) = SVector{2, Float64}(_observed_pixel(last(o)))
    Wt = nuncertain > 0 ? SMatrix{2, 2, Float64, 4} : Nothing
    whitenings = [Wt[_observation_whitening(last(o)) for o in t] for t in tracks]
    # the start: the frames' rotations, each track's mean bearing in the common frame over the observations it keeps
    nf = length(frames)
    bearing(o) = SMatrix{3, 3, Float64, 9}(rotations[first(o)]) * VgtkCore.backproject(camera, pixel(o))
    # (a track none of whose kept pixels back-projects has no start direction: NaN, and its observations are then dropped)
    function start_direction(t, keep)
        rays = [r for (i, o) in enumerate(t) if keep[i] for r in (bearing(o),) if all(isfinite, r)]
        return VgtkCore.Eliminable{E}(VgtkCore.SphereManifold(isempty(rays) ? SVector(NaN, NaN, NaN) : sum(rays)))
    end
    vars = V[[VgtkCore.SO3Manifold(SMatrix{3, 3, Float64, 9}(rotations[k])) for k in frames];
             [start_direction(t, trues(length(t))) for t in tracks]]
    chc === nothing || push!(vars, chc)                                     # the camera, the last variable
    ic = nf + length(tracks) + 1
    width = Float64(loss_width)
    observation(kernel, j, i, o) = chc === nothing ?
        _Observation{E, C, P, typeof(kernel), Wt}(bc, fwd, pixel(o), SVector(fid[first(o)], nf + j), kernel, whitenings[j][i]) :
        _CameraObservation{E, typeof(kernel), Wt}(pixel(o), SVector(fid[first(o)], nf + j, ic), kernel, whitenings[j][i])
    residual_type(kernel) = chc === nothing ? _Observation{E, C, P, typeof(kernel), Wt} : _CameraObservation{E, typeof(kernel), Wt}
    all_free = chc === nothing ? 3 : 7
    # the observations kept, a fixed point: an observation whose residual or Jacobian is not finite at its track's start (a
    # start prediction off the model's valid domain has neither) is dropped, a track left in fewer than two frames is dropped
    # whole, and a partly kept track restarts from the mean bearing of the observations it keeps — which moves its other
    # predictions, so its kept observations are tested again, until no track loses one (each round only drops, so at most one
    # round per observation). The solve is then the solve of the kept observations alone. A track left out keeps its start
    # direction, held.
    finite_at_start(j, i, o) = (ob = observation(nothing, j, i, o); r = NLLSsolver.computeresjac(all_free, ob, NLLSsolver.getvars(ob, vars)...);
                                all(isfinite, r[1]) && all(isfinite, r[2]))
    active = [trues(length(t)) for t in tracks]
    for (j, t) in enumerate(tracks)
        while any(active[j])
            keep = [active[j][i] && finite_at_start(j, i, o) for (i, o) in enumerate(t)]
            length(unique(first(o) for (i, o) in enumerate(t) if keep[i])) >= 2 || (keep = falses(length(t)))
            keep == active[j] && break
            active[j] = keep
            vars[nf + j] = start_direction(t, any(keep) ? keep : trues(length(t)))   # left out: back to its start direction
        end
    end
    nkept = sum(count, active; init = 0)
    nkept > 0 || throw(ArgumentError("refine_rotations: no observation is imaged at the start (every track keeps fewer than two " *
                                     "frames whose start prediction lies in the camera model's valid domain)"))
    unfixed = trues(length(vars)); unfixed[1] = false                       # the gauge frame
    for j in eachindex(tracks)
        any(active[j]) || (unfixed[nf + j] = false)                          # a track left out keeps its start direction
    end
    # a frame left with no kept observation is held at its start rotation (no cost would constrain it), and listed
    observed = falses(nf)
    for (j, t) in enumerate(tracks), (i, o) in enumerate(t)
        active[j][i] && (observed[fid[first(o)]] = true)
    end
    unconstrained = K[frames[i] for i in 2:nf if !observed[i]]
    foreach(k -> (unfixed[fid[k]] = false), unconstrained)
    residual_norms(f) = [[active[j][i] ? (ob = observation(nothing, j, i, o); norm(f(ob, NLLSsolver.getvars(ob, vars)...))) : NaN
                          for (i, o) in enumerate(t)] for (j, t) in enumerate(tracks)]
    result = nothing
    if max_iter > 0                                                          # ONE LM run at the given width
        kernel = VgtkCore.loss_robustifier(loss, width)
        problem = NLLSsolver.NLLSProblem(V, residual_type(kernel))
        for v in vars
            NLLSsolver.addvariable!(problem, v)
        end
        for (j, t) in enumerate(tracks), (i, o) in enumerate(t)
            active[j][i] && NLLSsolver.addcost!(problem, observation(kernel, j, i, o))
        end
        result = NLLSsolver.optimize!(problem, VgtkCore.nlls_options(; max_iter, ftol, xtol), unfixed)
        vars = problem.variables
    end
    code = result === nothing ? :max_iterations : VgtkCore.lm_retcode(VgtkCore.nlls_stop_code(result.termination))   # NLLSsolver's stop, decoded (no stall test here)
    out = Dict(k => RotMatrix{3, Float64, 9}(SMatrix{3, 3, Float64, 9}((vars[fid[k]]::_FrameChart).R)) for k in frames)
    dirs = [SVector{3, Float64}(VgtkCore.model((vars[nf + j]::VgtkCore.Eliminable{E}).value)) for j in eachindex(tracks)]
    rn = residual_norms(_pixel_residual); flat = filter(isfinite, reduce(vcat, rn; init = Float64[]))
    outcome = max_iter == 0 ? :not_run : code
    niter = result === nothing ? 0 : result.niterations
    cam_out = chc === nothing ? camera : _with_camera_model(camera, VgtkCore.model(vars[ic]))
    calib = chc === nothing ? nothing :
        _self_calibration(vars[ic], camera_free, [(j, i, o) for (j, t) in enumerate(tracks) for (i, o) in enumerate(t) if active[j][i]],
                          (j, i, o) -> observation(max_iter == 0 ? NLLSsolver.NoRobust() : VgtkCore.loss_robustifier(loss, width), j, i, o),
                          vars, fid, nf, unconstrained)
    return VgtkCore.FitResult(out, code, niter, sqrt(sum(abs2, flat) / max(length(flat), 1)), nkept,
                              RotationBundleMetadata(dirs, rn, width, nobs, nobs - nkept, unconstrained, outcome, cam_out, calib))
end

# The camera's variable: nothing when held (`camera_free = ()`), else Core's ParameterManifold of its model restricted to the
# named parameters. A SplineLens camera (a gauge K: its parameters are its lattice's controls, Estimation's own spline route)
# is refused.
_camera_chart(camera, free::Tuple{}) = nothing
function _camera_chart(camera, free::Tuple{Vararg{Symbol}})
    m = camera isa VgtkCore.BoundedCamera ? VgtkCore.model(camera) : camera
    (m isa VgtkCore.CameraModel && !(m.lens isa VgtkCore.SplineLens)) ||
        throw(ArgumentError("refine_rotations: camera_free needs a parametric camera model (its parameters by name); this camera's lens is a $(nameof(typeof(m.lens)))"))
    names = VgtkCore.parameter_names(m)
    for p in free
        p in names || throw(ArgumentError("refine_rotations: the camera has no parameter $(repr(p)); its parameters are $(join(String.(names), ", "))"))
    end
    allunique(free) || throw(ArgumentError("refine_rotations: camera_free names a parameter twice: $free"))
    return VgtkCore.restrict(VgtkCore.ParameterManifold(m), free)
end
_with_camera_model(camera::VgtkCore.BoundedCamera, m) = VgtkCore.with_model(camera, m)
_with_camera_model(::VgtkCore.CameraModel, m) = m

# The self-calibration's information at the result (`RotationBundleMetadata`'s `calibration`): every kept observation's
# Jacobian (its loss's IRLS weight at the result folded in) over the frames (the gauge frame included), its track's direction
# and the camera chart; the directions eliminated track by track (the Schur complement of their 2 × 2 blocks); the reduced
# information's scaled spectrum with and without the gauge frame; the covariance s²·H⁻¹ of the gauge-fixed reduced system,
# its camera block mapped from the chart to the parameters.
function _self_calibration(cc, free, kept, observation, vars, fid, nf, unconstrained)
    nd = VgtkCore.tangent_size(cc); P = 3nf + nd
    A = zeros(P, P); S2 = 0.0; n = length(kept)
    Dmin = Inf
    bytrack = Dict{Int, Vector{Tuple{Int, Any}}}()
    for (j, i, o) in kept
        push!(get!(bytrack, j, Tuple{Int, Any}[]), (i, o))
    end
    for (j, obs) in bytrack
        Dj = zeros(2, 2); Bj = zeros(2, P)
        for (i, o) in obs
            ob = observation(j, i, o)
            e, J = NLLSsolver.computeresjac(7, ob, NLLSsolver.getvars(ob, vars)...)
            _, w, _ = NLLSsolver.robustifydcost(NLLSsolver.robustkernel(ob), sum(abs2, e))
            S2 += w * sum(abs2, e)
            k = fid[first(o)]
            Jc = zeros(2, P); Jc[:, (3k - 2):(3k)] = J[:, 1:3]; Jc[:, (3nf + 1):P] = J[:, 6:end]
            Jv = Matrix(J[:, 4:5])
            A .+= w .* (Jc' * Jc); Bj .+= w .* (Jv' * Jc); Dj .+= w .* (Jv' * Jv)
        end
        Dmin = min(Dmin, LinearAlgebra.eigmin(LinearAlgebra.Symmetric(Dj)))
        A .-= Bj' * (Dj \ Bj)
    end
    H = LinearAlgebra.Symmetric((A + A') / 2)
    scaled(M) = (d = 1 ./ sqrt.(LinearAlgebra.diag(M)); LinearAlgebra.Symmetric(d .* M .* d'))
    held = vcat([(3k - 2):(3k) for k in [1; [fid[u] for u in unconstrained]]]...)      # the gauge frame and the frames held
    keep = setdiff(1:P, held)
    keep_gauge = setdiff(1:P, held[4:end])
    Hf = H[keep, keep]
    p = length(keep) + 2 * length(bytrack); ν = 2n - p
    s2 = S2 / ν
    Σz = s2 .* inv(LinearAlgebra.Symmetric(Matrix(Hf)))
    Σc = Σz[(end - nd + 1):end, (end - nd + 1):end]
    names = VgtkCore.parameter_names(VgtkCore.model(cc)); idx = [findfirst(==(q), names) for q in free]
    θ(z) = VgtkCore.parameter_vector(VgtkCore.model(VgtkCore.retract(cc, z)))[idx]
    z0 = zeros(nd); T = ForwardDiff.jacobian(θ, z0)
    Σθ = T * Σc * T'
    return (; free, parameters = NamedTuple{free}(Tuple(θ(z0))), sigma = NamedTuple{free}(Tuple(sqrt.(LinearAlgebra.diag(Σθ)))),
            covariance = Σθ, s2, nu = ν,
            reduced_eigenvalues = LinearAlgebra.eigvals(scaled(LinearAlgebra.Symmetric(Matrix(H[keep_gauge, keep_gauge])))),
            reduced_eigenvalues_gauge_fixed = LinearAlgebra.eigvals(scaled(LinearAlgebra.Symmetric(Matrix(Hf)))),
            direction_min_eigenvalue = Dmin)
end
