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
# `refine_rotations` holds the camera's calibration. Its one companion,
# `refine_camera_and_rotations`, frees it: the AUTOCALIBRATION of a rotating
# camera (no target, a pure-rotation sequence; Hartley 1994; Agapito, Hayman and
# Reid 2001) — one more variable, VisualGeometryToolkitCore's `ParameterManifold`
# of the camera's model restricted to its free parameters (the camera block of
# Estimation's `refine_camera_and_poses`), on which every observation then also
# depends. With the first frame's rotation the gauge, a rotating sequence leaves
# nothing else free; the bundle's metadata carries the reduced (Schur) information
# of the rotations and the camera with which to check it. Both are the one
# bundle below (`_rotation_bundle`), the camera variable present or absent.
# =============================================================================

const _FrameChart = VgtkCore.SO3Manifold{Float64}
const _SphereChart = typeof(VgtkCore.SphereManifold(SVector(0.0, 0.0, 1.0)))

# A track's direction is a 2-dof sphere chart held as Core's `Eliminable{E}`: `E` says whether NLLSsolver eliminates it, which
# needs the installed NLLSsolver's Schur system (`MultiVariateLSschur`, Core's `nlls_features().schur`); `eliminate`
# defaults to it, and `eliminate = true` without it is refused (no inert setting).
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
# residual π − x (its observation Jacobian −I, so L⁻¹ of the covariance Σ = LLᵀ), refused unless Σ is finite and positive
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

# One observation of the autocalibration bundle: as `_Observation`, its camera the third variable (`varind` = frame,
# direction, camera), a `_CameraVariable`: its projection read from that variable's model, its valid domain the variable's.
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

# The autocalibration's camera variable: its chart, the model's `ParameterManifold` restricted to the free parameters (Core's
# `restrict`, a `SubsetManifold`), and that model's `valid_domain`, computed ONCE per value of the variable (at construction
# and at each LM update), not per residual: every observation shares the camera, and a lens with its own fold finds its domain
# by a search of its view-angle limit (Core's image_circle.jl). NLLSsolver reads it through `nvars`/`update`, the chart's.
# The domain of a parametric `CameraModel` (the only camera the autocalibration takes) is Core's whole plane or, inside a
# fold, its pixel disc, an `Ellipse{Float64}`: the field holds either, so a fold appearing or vanishing at an LM update keeps
# the variable's one type.
# The model itself is held too (built once per value, not per residual). An update takes plain Float64 steps only (the
# bundle's residuals have closed-form Jacobians; Core evaluates a domain on plain numbers): a dual step is a MethodError.
struct _CameraVariable{S, M}
    chart::S
    model::M
    domain::Union{VgtkCore.WholePlane, VgtkCore.Ellipse{Float64}}
end
function _CameraVariable(c::VgtkCore.SubsetManifold)
    m = VgtkCore.model(c)
    return _CameraVariable(c, m, VgtkCore.valid_domain(m))
end
NLLSsolver.nvars(v::_CameraVariable) = NLLSsolver.nvars(v.chart)
NLLSsolver.update(v::_CameraVariable, δ::AbstractVector{Float64}, start = 1) = _CameraVariable(NLLSsolver.update(v.chart, δ, start))

# π(f) of the camera model `m`, NaN off its valid domain `domain` (as `_project`); `_image` without the test (the camera
# block's derivative, taken where the test holds).
@inline _image(m, f) = (u = VgtkCore.imaging_transform(m)(f); SVector(u[1], u[2]))
@inline function _project_model(m, domain, f)
    p = _image(m, f)
    return VgtkCore.Point2(p[1], p[2]) ∈ domain ? p : SVector(NaN, NaN)
end
# The camera of the camera variable: the model of its chart. Another variable in that slot is a construction error, refused
# by name; the method also keeps inference from reading a frame or direction variable as the camera (NLLSsolver's cost loop
# is generic over the blocks).
_camera_of_chart(c::_CameraVariable) = c.model
_camera_of_chart(c) = throw(ArgumentError("refine_camera_and_rotations: the camera variable must be its _CameraVariable, got a $(typeof(c))"))
@inline _pixel_residual(r::_CameraObservation, ch::VgtkCore.RotationChart, d::VgtkCore.Eliminable, c) =
    _project_model(_camera_of_chart(c), c.domain, _camera_direction(ch, d)) - r.x
NLLSsolver.computeresidual(r::_CameraObservation, ch::VgtkCore.RotationChart, d::VgtkCore.Eliminable, c) =
    _whiten(r.whitening, _pixel_residual(r, ch, d, c))

# The block's cost as the LM reads it: non-finite (an iterate past the model's valid domain) is +Inf (Core's `finite_cost`, the
# stack's rule for every residual block): a rejected step, also where the step is below `xtol` (NLLSsolver accepts any step
# below it, and a NaN cost accepted there fails its callback's cost check).
NLLSsolver.computecost(vars::AbstractVector, r::Union{_Observation, _CameraObservation}) =
    VgtkCore.finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))

# An observation of a HELD frame (the gauge: Core's LmProblem solves every variable it is given, so a held one is no
# variable and the block carries its value, as Estimation's bundle blocks do): `obs`'s residual and Jacobian with the frame's
# rotation `frame` fixed, its variables the direction (and the camera) only, `varind` their indices in the solve's variables.
# NLLSsolver's variable flags count from the direction: shifted by one they are `obs`'s (bit 1 the frame, never set here).
struct _HeldFrame{N, O} <: NLLSsolver.AbstractResidual
    obs::O
    frame::_FrameChart
    varind::SVector{N, Int}
end
NLLSsolver.ndeps(::_HeldFrame{N}) where {N} = static(N)
NLLSsolver.nres(::_HeldFrame) = static(2)
NLLSsolver.varindices(r::_HeldFrame) = r.varind
NLLSsolver.robustkernel(r::_HeldFrame) = NLLSsolver.robustkernel(r.obs)
Base.eltype(::_HeldFrame) = Float64
# NLLSsolver reads only the block's variables (the direction, and the camera); the held frame's rotation is prepended for `obs`.
NLLSsolver.getvars(r::_HeldFrame{1, <:_Observation{E}}, vars::Vector) where {E} = (vars[r.varind[1]]::VgtkCore.Eliminable{E, _SphereChart},)
NLLSsolver.getvars(r::_HeldFrame{2, <:_CameraObservation{E}}, vars::Vector) where {E} =
    (vars[r.varind[1]]::VgtkCore.Eliminable{E, _SphereChart}, vars[r.varind[2]])
NLLSsolver.computeresidual(r::_HeldFrame, args...) = NLLSsolver.computeresidual(r.obs, r.frame, args...)
NLLSsolver.computeresjac(varflags, r::_HeldFrame, args...) = NLLSsolver.computeresjac(Int(varflags) << 1, r.obs, r.frame, args...)
NLLSsolver.computecost(vars::AbstractVector, r::_HeldFrame) =
    VgtkCore.finite_cost(NLLSsolver.computecost(r.obs, r.frame, NLLSsolver.getvars(r, vars)...))
# The same observation with its variables' indices in the solve's variables (the held ones left out).
_with_varind(o::_Observation{E, C, P, L, W}, varind) where {E, C, P, L, W} =
    _Observation{E, C, P, L, W}(o.camera, o.projection, o.x, varind, o.kernel, o.whitening)
_with_varind(o::_CameraObservation{E, L, W}, varind) where {E, L, W} = _CameraObservation{E, L, W}(o.x, varind, o.kernel, o.whitening)

# Its closed-form frame and direction blocks (those of `_Observation`, through the variable's model) and its camera block,
# ForwardDiff of π along the camera chart's coordinates at its anchor. `varflags`: bit 1 the frame, bit 2 the direction, bit 3
# the camera; only the free variables' columns are returned, in that order.
function NLLSsolver.computeresjac(varflags, r::_CameraObservation, ch::_FrameChart, d::VgtkCore.Eliminable, c)
    m = _camera_of_chart(c)
    R = SMatrix{3, 3, Float64, 9}(ch.R); dir = VgtkCore.model(d.value); f = R' * dir
    u = _project_model(m, c.domain, f)
    _, M = VgtkCore.backproject_with_jacobian(m, u)
    Mp = _whiten(r.whitening, (M' * M) \ M')
    Jω = Mp * (R' * VgtkCore.skew(dir))
    Jv = Mp * (R' * VgtkCore.retract_jacobian(d.value))
    z0 = zero(SVector{VgtkCore.tangent_size(c.chart), Float64})
    Jc = _whiten(r.whitening, ForwardDiff.jacobian(z -> _image(VgtkCore.model(VgtkCore.retract(c.chart, z)), f), z0))
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

`FitResult` metadata of [`refine_rotations`](@ref) and
[`refine_camera_and_rotations`](@ref): `directions` (one unit
direction per track, the common frame), `residuals` (per track, the pixel
reprojection error norm of each of its observations at the result, unwhitened),
`loss_width` (the width the strategy ran at: px, or in whitened units when the
observations carry covariances; `NaN` for the plain least squares), `inlier_rms`
(the rms of the `residuals` within `loss_width` — the loss's own inliers, the
result's `rms` taken over them alone, beside the plain `rms` over every kept
observation; `NaN` for the plain least squares or with no residual within it), `scale` and
`scale_source` (the scale record of the robust-estimation contract: the given
`scale`, or `NaN` with `ScaleNone` for a `width` or the least squares; read
through VisualGeometryToolkitCore's `noise_scale`/`scale_source`), `n_observations` (given), `n_dropped_start`
(the observations left out of the solve: those whose residual or Jacobian is
not finite at the start — a start prediction off the camera model's valid
domain, where it has no projection and no back-projection Jacobian, as for a
grid lens calibrated over the sensor whose frames' observations reach its
corners from a start a fraction of a degree off, the Tamron 20 mm of
2026-10-29 — and the rest of a track left in fewer than two frames; a
dropped observation's residual is NaN, and a track left out keeps its start
direction), `frames_unconstrained` (the frames other than the gauge left with
no kept observation: each is held at its start rotation),

`camera` (the camera at the result: the one given by `refine_rotations`, which
holds it, else the refined one, of the given camera's type and, for a
`BoundedCamera`, its sensor and lens coverage) and `calibration`, `nothing` when
the camera is held, else the autocalibration's information at the result
([`refine_camera_and_rotations`](@ref)):

- `free_parameters` — the names of the camera parameters refined;
- `parameters`, `sigma`, `covariance` — their values, standard deviations and
  covariance (parameter units: pixels for `f`, `cx`, `cy`; the lens's own for its
  coefficients) — `s²` × the camera block of the robust bundle's Gauss–Newton
  `(JᵀJ)⁻¹` (each observation weighted by its loss's IRLS weight at the result,
  over the rotations, the track directions and the camera, the gauge frame
  removed; the directions eliminated by their Schur complement, the information
  accumulated block by block, never dense), its camera block mapped from the camera
  chart to the parameters; `s² = Σ wᵢ‖rᵢ‖² / ν`, `ν = 2n − p` (`n` observations
  kept, `p` unknowns: `3(K − 1)` rotations, 2 per kept track, the free parameters);
- `s2`, `nu` — `s²` and `ν`;
- `reduced_eigenvalues_gauge_fixed` — the eigenvalues (ascending) of the
  information reduced to the rotations and the camera (the inverse of their
  marginal covariance: the Schur complement of the directions), the gauge frame
  removed, its rows and columns scaled to a unit diagonal — positive when nothing
  but the gauge is free;
- `direction_min_eigenvalue` — the least eigenvalue over the kept tracks of
  their 2 × 2 direction blocks (each positive: every kept direction observed).
"""
struct RotationBundleMetadata{K, C, I}
    directions::Vector{SVector{3, Float64}}
    residuals::Vector{Vector{Float64}}
    loss_width::Float64
    inlier_rms::Float64
    scale::Float64
    scale_source::VgtkCore.ScaleSource
    n_observations::Int
    n_dropped_start::Int
    frames_unconstrained::Vector{K}
    camera::C
    calibration::I
end

"""
    refine_rotations(camera, rotations, tracks; optimization_strategy = Triggs{WelschShape}(…), scale | width, max_iter,
                     ftol, xtol, eliminate = <NLLSsolver's Schur system>) -> FitResult{Dict{K, RotMatrix{3, Float64}}, RotationBundleMetadata}

The rotation bundle of a pure-rotation sequence seen by the calibrated `camera`
(fixed: the calibration is held, a rotation-only bundle with no parameter
keyword — [`refine_camera_and_rotations`](@ref) is the autocalibration; its `backproject` and `backproject_with_jacobian` give the start and
the Jacobian; a `CameraModel` or a `BoundedCamera`), projected through its
closed-form forward map π, its model's `imaging_transform` (a grid lens's
`SplineLens` through its forward spline: give a SplineLens camera one with VisualGeometryToolkitEstimation's
`fit_forward_spline(camera; tolerance)`, which keeps its backward, K and frame, while `fit_camera(SplineLens, …)` turns
a gauge-K camera onto its optical axis, a new frame). A residual is NaN off
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
- `optimization_strategy` — the robust strategy of the robust-estimation contract: a `Triggs{K}`
  (VisualGeometryToolkitCore's; the default VisualGeometryToolkitEstimation's pixel bundle's, configs/default.toml
  `[refine_camera_and_poses]`: `Triggs{WelschShape}` at the shape's efficiency constant for a 2-D residual), its loss `K` as NLLSsolver's robustifier (`loss_robustifier`) at ONE width fixed for the bundle's one
  NLLSsolver run; or `nothing`, the plain least squares (no scale or width).
- exactly one of `scale = σ` (the observations' noise scale, pixels or whitened units, or an upstream robust result:
  `register_rotations` hands the verified pairs' RANSAC scales pooled by Estimation's `pooled_noise_scale`) — the width is the strategy's efficiency
  constant × σ — or `width = w` (the loss width). Neither, or both, is an `ArgumentError`. `max_iter`, `ftol` (the relative cost decrease) and `xtol`
  (the largest step in chart units: radians for a rotation or a direction)
  are its stops (Core's `LmStops`, its one LM adapter `lmsolve_chart!` on an `LmProblem`); the result's `iterations` the LM iterations
  it took.
- `eliminate` — whether NLLSsolver eliminates the directions (their Schur
  complement), which needs the loaded NLLSsolver's Schur system
  (`MultiVariateLSschur`; Core's `nlls_features().schur`): the default where it
  has one; `false` solves every unknown together, the default otherwise.
  `eliminate = true` without the Schur system is an `ArgumentError` naming it.

A refine ends in the local minimum its start leads to; the result's termination
record (Core's `retcode` and `stop_rule`: a `ReturnCode.T` and a `StopRule`; `converged(result)` its success) says how
the bundle ended: NLLSsolver's stop, a step-size or rejection stop `Success` only where the robust cost is stationary at the
result (Core's one success rule, at its `[lm] stationarity_tol`), `Stalled` otherwise; `max_iter = 0` (no
solve: the start's rotations are the result) `MaxIters` with `IterationCap`. A track observed in fewer than two frames, or naming a frame
without a start rotation, some observations with a covariance and some
without, a covariance that is not finite and positive definite, and a start
that keeps no observation (every track left in fewer than two imaged frames)
are an `ArgumentError`, and so are the retired `loss`/`loss_width` (the strategy and its scale input replace them) and
`camera_free` (the bundle holds the calibration: [`refine_camera_and_rotations`](@ref) frees it).
"""
function refine_rotations(camera, rotations::AbstractDict, tracks::AbstractVector;
                          optimization_strategy = _DEFAULT_BUNDLE_STRATEGY, scale = nothing, width = nothing,
                          max_iter::Integer, ftol::Real, xtol::Real, eliminate::Bool = _SCHUR, retired...)
    for k in keys(retired)
        k === :camera_free && throw(ArgumentError(
            "refine_rotations: the rotation bundle holds the camera's calibration (no parameter keyword); to refine the camera " *
            "with the rotations — the autocalibration of a rotating camera — call refine_camera_and_rotations(…; fixed)"))
        k in (:loss, :loss_width) && throw(ArgumentError(
            "refine_rotations: the keyword `$k` is retired: give the robust strategy as `optimization_strategy = Triggs{Shape}(…)` " *
            "and exactly one of `scale` or `width` (the robust-estimation contract)"))
        throw(ArgumentError("refine_rotations: unknown keyword `$k`"))
    end
    bundle = VgtkEstimation.bundle_loss(optimization_strategy; scale, width, caller = "refine_rotations")
    return _rotation_bundle(camera, rotations, tracks, nothing, (); bundle, max_iter, ftol, xtol, eliminate,
                            what = "refine_rotations")
end

# VisualGeometryToolkitEstimation's configuration of record (its configs/default.toml), read once through its public reader:
# a strategy default the bundles share with Estimation is looked up there, never restated here.
const _ESTIMATION_DEFAULTS = VgtkCore.parse_defaults(VgtkEstimation.default_config_path())
_estimation_strategy(table::AbstractString, key::AbstractString, class::Symbol) =
    VgtkEstimation.optimization_strategy_from_toml(VgtkCore.toml_default(_ESTIMATION_DEFAULTS, table, key);
                                                   class, section = "$table.$key")

# The default strategy of the rotation bundles: Estimation's pixel bundle's, [refine_camera_and_poses]
# `optimization_strategy` (Triggs{Welsch} stating no efficiency constant: its width is Core's efficiency constant for a 2-D
# residual, a pixel reprojection error, times the caller's `scale`).
const _DEFAULT_BUNDLE_STRATEGY = _estimation_strategy("refine_camera_and_poses", "optimization_strategy", :refine_camera_and_poses)

"""
    refine_camera_and_rotations(camera, rotations, tracks; optimization_strategy = Triggs{WelschShape}(…), scale | width,
                                fixed = (), max_iter, ftol, xtol, eliminate = <NLLSsolver's Schur system>)
        -> FitResult{Dict{K, RotMatrix{3, Float64}}, RotationBundleMetadata}

The AUTOCALIBRATION of a rotating camera: no target, a pure-rotation sequence (Hartley 1994; Agapito, Hayman and Reid 2001).
The rotation bundle of [`refine_rotations`](@ref) — the same observations, start, kept set, gauge and result — with the
camera's free parameters one more variable: VisualGeometryToolkitCore's `ParameterManifold` of the camera's model restricted
to them (the camera block of Estimation's `refine_camera_and_poses`), its Jacobian ForwardDiff through the chart. The only
gauge is the first frame's rotation, held exactly; with it a rotating sequence leaves nothing else free, which the metadata's
`calibration` shows (the reduced information, and the parameters' covariance: [`RotationBundleMetadata`](@ref)). The
observations kept at the start are those the GIVEN camera images; the residuals, the loss width and the result are the
refined camera's.

A contract refine:
- `optimization_strategy` — the robust strategy, a `Triggs{K}` (VisualGeometryToolkitCore's; the default
  VisualGeometryToolkitEstimation's pixel bundle's, configs/default.toml `[refine_camera_and_poses]`: `Triggs{WelschShape}`
  at the shape's efficiency constant for a 2-D residual): the loss `K` as NLLSsolver's
  robustifier at ONE fixed width.
- exactly one of `scale = σ` (the observations' noise scale, px or whitened units; or an upstream robust result) — the width
  is then the strategy's efficiency constant × σ — or `width = w` (the loss width itself).
- `fixed` — the camera parameters HELD, by name or by a coefficient family's group (`:k` for `k1…`, `:d`, `:p`, `:sx`,
  `:sy`): VisualGeometryToolkitCore's `free_parameter_names(model, fixed)`. The default `()` frees every parameter but the
  skew, which the stack holds by construction (named in `fixed` it is a no-op). An unknown or repeated entry, a camera with
  nothing free, a SplineLens camera (its controls have no static chart), and the retired `camera_free`, `free`, `fix`,
  `fixed_params` and `hold` are an `ArgumentError`.
- `max_iter`, `ftol`, `xtol`, `eliminate` — as [`refine_rotations`](@ref)'s.
"""
function refine_camera_and_rotations(camera, rotations::AbstractDict, tracks::AbstractVector;
                                     optimization_strategy = _DEFAULT_BUNDLE_STRATEGY,
                                     scale = nothing, width = nothing, fixed::Tuple{Vararg{Symbol}} = (),
                                     max_iter::Integer, ftol::Real, xtol::Real, eliminate::Bool = _SCHUR,
                                     retired...)
    for k in keys(retired)                                       # camera_free and the other spellings of a held-parameter list
        k in _RETIRED_HOLD_KEYWORDS && throw(ArgumentError(
            "refine_camera_and_rotations: the keyword `$k` is not the stack's; give the HELD camera parameters as `fixed = (…)` " *
            "(every other parameter but the skew is free)"))
        throw(ArgumentError("refine_camera_and_rotations: unknown keyword `$k`"))
    end
    bundle = VgtkEstimation.bundle_loss(optimization_strategy; scale, width, caller = "refine_camera_and_rotations")
    m = camera isa VgtkCore.BoundedCamera ? VgtkCore.model(camera) : camera
    names = VgtkCore.free_parameter_names(m, fixed)
    isempty(names) && throw(ArgumentError(
        "refine_camera_and_rotations: `fixed` holds every camera parameter: nothing to calibrate (refine_rotations is the bundle " *
        "with the calibration held)"))
    # the camera's variable: Core's ParameterManifold of its model restricted to the free parameters (names checked by
    # `free_parameter_names`); a SplineLens camera (a gauge K: its parameters are its lattice's controls, which the stack
    # refines only by back-projection, or holds) is refused
    (m isa VgtkCore.CameraModel && !(m.lens isa VgtkCore.SplineLens)) ||
        throw(ArgumentError("refine_camera_and_rotations: the autocalibration needs a parametric camera model (its parameters by name); this camera's lens is a $(nameof(typeof(m.lens)))"))
    chc = VgtkCore.restrict(VgtkCore.ParameterManifold(m), names)
    return _rotation_bundle(camera, rotations, tracks, chc, names; bundle, max_iter, ftol, xtol, eliminate,
                            what = "refine_camera_and_rotations")
end

# The retired names of the held-parameter keyword (`fixed` is the stack's one): refused naming it.
const _RETIRED_HOLD_KEYWORDS = (:camera_free, :free, :fix, :fixed_params, :hold)

# The one rotation bundle of `refine_rotations` (`chc === nothing`: the camera held) and `refine_camera_and_rotations` (`chc`
# the camera's chart over its free parameters `free`), `what` the caller named in a message; `bundle` the loss of the
# robust-estimation contract, VisualGeometryToolkitEstimation's `bundle_loss` of the caller's keywords (its strategy, scale
# input and deployed loss). Its strategy is a Triggs{K} — the loss K as NLLSsolver's robustifier at the deployed width,
# `sqrt(sq_loss_width(loss))` as every bundle of the stack — or `nothing` (plain least squares); an Irls{K} is refused by Estimation's bundle_loss: this
# bundle runs one LM. The scale record is the input's (`scale_source`, and `noise_scale` unless ScaleNone: NaN).
function _rotation_bundle(camera, rotations::AbstractDict{K}, tracks::AbstractVector, chc, free_names::Tuple{Vararg{Symbol}};
                          bundle::NamedTuple, max_iter::Integer, ftol::Real, xtol::Real,
                          eliminate::Bool, what::AbstractString) where {K}
    loss = bundle.strategy === nothing ? nothing : VgtkCore.strategy_loss(bundle.strategy)()
    width = bundle.loss === nothing ? NaN : sqrt(VgtkCore.sq_loss_width(bundle.loss))
    scale_source = VgtkCore.scale_source(bundle.input)
    scale = scale_source == VgtkCore.ScaleNone ? NaN : Float64(VgtkCore.noise_scale(bundle.input))
    loss === nothing || isfinite(width) && width > 0 ||
        throw(ArgumentError("$what: the loss width must be positive and finite, got $width"))
    robustifier() = loss === nothing ? NLLSsolver.NoRobust() : VgtkCore.loss_robustifier(loss, width)
    fwd = _forward_map(camera); bc = _BundleCamera(camera); C = typeof(bc); P = typeof(fwd)
    frames = sort!(collect(keys(rotations)))
    fid = Dict(k => v for (v, k) in enumerate(frames))
    eliminate && !_SCHUR && throw(ArgumentError(
        "$what: eliminate = true needs NLLSsolver's Schur system (MultiVariateLSschur), which the loaded NLLSsolver does not " *
        "have; leave `eliminate` out (every unknown solved together)"))
    E = eliminate                                                 # the directions' Eliminable{E}
    cv = chc === nothing ? nothing : _CameraVariable(chc)
    V = chc === nothing ? Union{_FrameChart, VgtkCore.Eliminable{E, _SphereChart}} :
        Union{_FrameChart, VgtkCore.Eliminable{E, _SphereChart}, typeof(cv)}
    nobs = 0; nuncertain = 0
    for t in tracks
        length(unique(first.(t))) >= 2 || throw(ArgumentError("$what: a track needs observations in two frames"))
        all(o -> haskey(fid, first(o)), t) || throw(ArgumentError("$what: a track names a frame with no start rotation"))
        nobs += length(t); nuncertain += count(o -> last(o) isa VgtkCore.Uncertain, t)
    end
    0 < nuncertain < nobs && throw(ArgumentError("$what: $nuncertain of $nobs observations carry a covariance; " *
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
    chc === nothing || push!(vars, cv)                                      # the camera, the last variable
    ic = nf + length(tracks) + 1
    observation(kernel, j, i, o) = chc === nothing ?
        _Observation{E, C, P, typeof(kernel), Wt}(bc, fwd, pixel(o), SVector(fid[first(o)], nf + j), kernel, whitenings[j][i]) :
        _CameraObservation{E, typeof(kernel), Wt}(pixel(o), SVector(fid[first(o)], nf + j, ic), kernel, whitenings[j][i])
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
    nkept > 0 || throw(ArgumentError("$what: no observation is imaged at the start (every track keeps fewer than two " *
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
    sol = nothing
    if max_iter > 0                                                          # ONE LM run at the given width
        # Core's one multi-block LM (`LmProblem` + `lmsolve_chart!`, its one success rule: the √ρ′-weighted stationarity
        # postcondition at Core's [lm] stationarity_tol), on the FREE variables only: the held ones (the gauge frame, the
        # tracks left out, the unconstrained frames) are no variables, and a block of the held gauge frame carries its
        # rotation (`_HeldFrame`)
        kernel = robustifier()
        newidx = cumsum(unfixed) .* unfixed                                 # a free variable's index in the solve, 0 held
        blocks = NLLSsolver.AbstractResidual[]
        for (j, t) in enumerate(tracks), (i, o) in enumerate(t)
            active[j][i] || continue
            ob = observation(kernel, j, i, o); vi = ob.varind
            if unfixed[vi[1]]
                push!(blocks, _with_varind(ob, SVector(Tuple(newidx[v] for v in vi))))
            else
                rest = SVector(Tuple(newidx[v] for v in vi[2:end]))
                push!(blocks, _HeldFrame{length(rest), typeof(ob)}(ob, vars[vi[1]]::_FrameChart, rest))
            end
        end
        lp = VgtkCore.LmProblem(V[vars[v] for v in eachindex(vars) if unfixed[v]], blocks)
        sol = VgtkCore.lmsolve_chart!(lp, VgtkCore.LmStops(; max_iter, ftol, xtol))
        for v in eachindex(vars)                                            # the solution into the full variable vector
            unfixed[v] && (vars[v] = lp.variables[newidx[v]])
        end
    end
    # The termination record (§14): Core's adapter's (a stop other than a failure or the cap a success only where the
    # stationarity postcondition holds); `max_iter = 0` runs no solve: the iteration cap.
    rc, rule = sol === nothing ? (VgtkCore.ReturnCode.MaxIters, VgtkCore.IterationCap) : (sol.code, sol.rule)
    out = Dict(k => RotMatrix{3, Float64, 9}(SMatrix{3, 3, Float64, 9}((vars[fid[k]]::_FrameChart).R)) for k in frames)
    dirs = [SVector{3, Float64}(VgtkCore.model((vars[nf + j]::VgtkCore.Eliminable{E}).value)) for j in eachindex(tracks)]
    rn = residual_norms(_pixel_residual); flat = filter(isfinite, reduce(vcat, rn; init = Float64[]))
    niter = sol === nothing ? 0 : sol.iterations
    cam_out = chc === nothing ? camera : VgtkCore.with_model(camera, _camera_of_chart(vars[ic]))
    calib = chc === nothing ? nothing :
        _autocalibration((vars[ic]::typeof(cv)).chart, free_names, [(j, i, o) for (j, t) in enumerate(tracks) for (i, o) in enumerate(t) if active[j][i]],
                          (j, i, o) -> observation(max_iter == 0 ? NLLSsolver.NoRobust() : robustifier(), j, i, o),
                          vars, fid, nf, unconstrained)
    return VgtkCore.FitResult(out, rc, rule, niter, _plain_rms(flat), nkept,
                              RotationBundleMetadata(dirs, rn, width, _rms(filter(≤(width), flat)), scale, scale_source, nobs, nobs - nkept, unconstrained, cam_out, calib))
end

# The rms of residual norms: the result's `rms` over every kept observation (0 with none, as before), and its inliers' (those
# within the loss width; NaN with none)
_plain_rms(r) = sqrt(sum(abs2, r; init = 0.0) / max(length(r), 1))
_rms(r) = isempty(r) ? NaN : _plain_rms(r)

# The autocalibration's information at the result (`RotationBundleMetadata`'s `calibration`): the Gauss–Newton information
# H = Σ w JᵀJ of every kept observation (its loss's IRLS weight w at the result; the rows already whitened) over the frames,
# the tracks' directions and the camera chart, accumulated in its block structure — H_aa over the frames and the camera (`a`,
# 3K + nd coordinates), per track its 2 × 2 direction block D_j and its coupling H_aj (each observation touches one frame,
# one direction and the camera) — never as dense rows. The directions are eliminated: the reduced information S = H_aa −
# Σ_j H_aj D_j⁻¹ H_ajᵀ (their Schur complement) over the free frames and the camera, the gauge frame and the held frames left
# out, is the inverse of their marginal covariance, the same as the joint (JᵀWJ)⁻¹'s block; times s² = Σ wᵢ‖rᵢ‖² / ν; the
# camera block mapped from the chart to the parameters. Each kept direction's D_j gives `direction_min_eigenvalue`.
function _autocalibration(cc, free_names, kept, observation, vars, fid, nf, unconstrained)
    nd = VgtkCore.tangent_size(cc)
    na = 3nf + nd; cam = (3nf + 1):na                                        # the frames' and the camera's coordinates
    held = vcat([(3k - 2):(3k) for k in [1; [fid[u] for u in unconstrained]]]...)      # the gauge frame and the frames held
    free = setdiff(1:na, held)
    S2 = 0.0; n = length(kept)
    Haa = zeros(na, na)
    D = Dict{Int, SMatrix{2, 2, Float64, 4}}()                               # per kept track: its direction block
    Hab = Dict{Int, Matrix{Float64}}()                                       # its coupling to the frames and the camera, na × 2
    seen = Dict{Int, Set{Int}}()                                             # the frames that observe it
    for (j, i, o) in kept
        ob = observation(j, i, o)
        e, J = NLLSsolver.computeresjac(7, ob, NLLSsolver.getvars(ob, vars)...)
        _, w, _ = NLLSsolver.robustifydcost(NLLSsolver.robustkernel(ob), sum(abs2, e))
        S2 += w * sum(abs2, e)
        k = fid[first(o)]; ia = [(3k - 2):(3k); cam]
        Ja = Matrix(J[:, [1:3; 6:size(J, 2)]]); Jv = SMatrix{2, 2, Float64, 4}(J[:, 4:5])
        Haa[ia, ia] .+= w .* (Ja' * Ja)
        get!(() -> zeros(na, 2), Hab, j)[ia, :] .+= w .* (Ja' * Jv)
        push!(get!(Set{Int}, seen, j), k)
        D[j] = get(D, j, zero(SMatrix{2, 2, Float64, 4})) + w * (Jv' * Jv)
    end
    tracks = sort!(collect(keys(D))); nt = length(tracks)
    Dmin = minimum(j -> LinearAlgebra.eigmin(LinearAlgebra.Symmetric(D[j])), tracks)
    singular() = throw(ArgumentError("refine_camera_and_rotations: the autocalibration's information is singular at the result"))
    Dmin > 0 || singular()
    S = Haa
    for j in tracks
        rows = [(3k - 2):(3k) for k in sort!(collect(seen[j]))]; rows = [reduce(vcat, rows); cam]   # its nonzero rows only
        Hj = Hab[j][rows, :]
        S[rows, rows] .-= Hj * inv(D[j]) * Hj'                                 # the direction's Schur complement
    end
    Sf = LinearAlgebra.Symmetric(S[free, free])
    C = LinearAlgebra.cholesky(Sf; check = false)
    LinearAlgebra.issuccess(C) || singular()
    ν = 2n - (length(free) + 2nt)
    s2 = S2 / ν
    nc = length(free) - nd                                                  # the camera's coordinates are the last nd free ones
    E = zeros(length(free), nd); E[(nc + 1):end, :] = LinearAlgebra.I(nd)
    Σc = s2 .* (C \ E)[(nc + 1):end, :]                                    # the camera's block of S⁻¹ alone (nd right-hand sides)
    scaled(M) = (d = 1 ./ sqrt.(LinearAlgebra.diag(M)); LinearAlgebra.Symmetric(d .* M .* d'))
    names = VgtkCore.parameter_names(VgtkCore.model(cc)); idx = [findfirst(==(q), names) for q in free_names]
    θ(z) = VgtkCore.parameter_vector(VgtkCore.model(VgtkCore.retract(cc, z)))[idx]
    z0 = zeros(nd); T = ForwardDiff.jacobian(θ, z0)
    Σθ = VgtkCore.propagate(T, Σc)                     # Core's propagation through the chart → parameters Jacobian
    return (; free_parameters = free_names, parameters = NamedTuple{free_names}(Tuple(θ(z0))), sigma = NamedTuple{free_names}(Tuple(sqrt.(LinearAlgebra.diag(Σθ)))),
            covariance = Σθ, s2, nu = ν,
            reduced_eigenvalues_gauge_fixed = LinearAlgebra.eigvals(scaled(Sf)),
            direction_min_eigenvalue = Dmin)
end
