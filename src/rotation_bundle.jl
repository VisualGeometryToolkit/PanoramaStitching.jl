# =============================================================================
# The rotation bundle — the frames' rotations and the tracks' directions
# =============================================================================
#
# A camera that only rotated sees a distant point at one direction d_j of the
# common frame from every frame k: x_jk = π(R_kᵀ·d_j), π the calibrated camera's
# projection (fixed), R_k frame k's rotation (d = R_k·f_k). The rotation bundle
# (`RotationBundleProblem`, refined by Core's `LevenbergMarquardt` through the block contract)
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
# With the camera in the problem the bundle holds its calibration. With the camera in
# the guess, `(; camera, rotations)`, it frees it: the AUTOCALIBRATION of a rotating
# camera (no target, a pure-rotation sequence; Hartley 1994; Agapito, Hayman and
# Reid 2001) — one more variable, VisualGeometryToolkitCore's `ParameterManifold`
# of the camera's model restricted to its free parameters (the camera block of
# Estimation's `CameraPosesProblem`), on which every observation then also
# depends. With the first frame's rotation the gauge, a rotating sequence leaves
# nothing else free; the bundle's metadata carries the reduced (Schur) information
# of the rotations and the camera with which to check it. Both are the one
# bundle below (`_rotation_bundle_statement`, `_rotation_bundle_blocks` and `_rotation_bundle_result`), the camera variable present or absent.
# =============================================================================

# A frame's variable is a `VgtkCore.SO3Manifold{Float64}`, a track direction's a 2-dof
# `VgtkCore.SphereManifold{3, 2, Float64, 6, typeof(identity)}` (the type of `SphereManifold(d)` for a 3-vector `d`), each spelled
# where a block asserts it (check_naming R20: no constant names one concrete type).

# A track's direction is a 2-dof sphere chart held as Core's `Eliminable{E}`: `E` says whether NLLSsolver eliminates it, which
# needs the loaded NLLSsolver's Schur system (`MultiVariateLSschur`, upstream pull request #60: Core's `has_nlls_schur()`, the
# stack's one detection); `eliminate` defaults to it, and `eliminate = true` without it is refused (no inert setting).

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
NLLSsolver.getvars(r::_Observation{E}, vars::Vector) where {E} = (vars[r.varind[1]]::VgtkCore.SO3Manifold{Float64}, vars[r.varind[2]]::VgtkCore.Eliminable{E, VgtkCore.SphereManifold{3, 2, Float64, 6, typeof(identity)}})
NLLSsolver.robustkernel(r::_Observation) = r.kernel
Base.eltype(::_Observation) = Float64

@inline _camera_direction(ch::VgtkCore.RotationChart, d::VgtkCore.Eliminable) = SMatrix{3, 3}(ch.R)' * VgtkCore.model(d.value)

# π(f): the forward map in closed form, NaN where it does not image AND off the model's valid domain (a grid lens's lattice),
# where its back-projection Jacobian is NaN. A forward map can image past that domain (a finite pixel), so without the second
# test the residual would be finite where its Jacobian
# is not: NLLSsolver would accept such an iterate, linearize there and loop on its NaN system for ever (measured 2026-11-04,
# the synthetic Tamron sequence from its true rotations). With it the residual and its Jacobian are defined on the same set,
# an iterate past it is a NaN cost (a rejected step), and an observation whose start prediction is past it is dropped
# (the bundle's start, counted in `n_dropped_start`).
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
    (all(isfinite, Li) && !iszero(Li)) || throw(ArgumentError("RotationBundleProblem: an observation's covariance $Σ is not finite and positive definite"))
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
    (vars[r.varind[1]]::VgtkCore.SO3Manifold{Float64}, vars[r.varind[2]]::VgtkCore.Eliminable{E, VgtkCore.SphereManifold{3, 2, Float64, 6, typeof(identity)}}, vars[r.varind[3]])
NLLSsolver.robustkernel(r::_CameraObservation) = r.kernel
Base.eltype(::_CameraObservation) = Float64

# The autocalibration's camera variable: its chart, the model's `ParameterManifold` restricted to the free parameters (Core's
# `restrict`, a `SubsetManifold`), and that model's `valid_domain`, computed ONCE per value of the variable (at construction
# and at each LM update), not per residual: every observation shares the camera, and a lens with its own fold finds its domain
# by a search of its view-angle limit (Core's image_circle.jl). NLLSsolver reads it through `nvars`/`update`, the chart's.
# The domain of a parametric `CameraModel` (the only camera the autocalibration takes) is Core's whole plane or, inside a
# fold, its pixel disc (`Ellipse{Float64}`) or, for a decentred lens, the star polygon through each azimuth's fold
# (`PolygonRegion`, about 10 ms to build): the field holds any of them, so a fold appearing or vanishing at an LM update
# keeps the variable's one type. An LM step whose domain Core refuses (`ArgumentError`: a decentred lens's fold curve that is
# not star-shaped about its centre, or an azimuth that images no ray) holds `_NoDomain`, which no point is in: every residual
# is NaN, the block's cost +Inf (`finite_cost`) and the LM rejects the step, as it does any iterate off the domain. The start
# is not an LM step: a start camera whose domain is refused is refused.
# The model itself is held too (built once per value, not per residual). An update takes plain Float64 steps only (the
# bundle's residuals have closed-form Jacobians; Core evaluates a domain on plain numbers): a dual step is a MethodError.
struct _NoDomain end
Base.in(_, ::_NoDomain) = false
struct _CameraVariable{S, M}
    chart::S
    model::M
    domain::Union{VgtkCore.WholePlane, VgtkCore.Ellipse{Float64}, VgtkCore.PolygonRegion{Vector{VgtkCore.Point2{Float64}}}, _NoDomain}
end
function _CameraVariable(c::VgtkCore.SubsetManifold)
    m = VgtkCore.model(c)
    return _CameraVariable(c, m, VgtkCore.valid_domain(m))
end
NLLSsolver.nvars(v::_CameraVariable) = NLLSsolver.nvars(v.chart)
function NLLSsolver.update(v::_CameraVariable, δ::AbstractVector{Float64}, start = 1)
    c = NLLSsolver.update(v.chart, δ, start)
    m = VgtkCore.model(c)
    domain = try
        VgtkCore.valid_domain(m)
    catch e
        e isa ArgumentError || rethrow()
        _NoDomain()
    end
    return _CameraVariable(c, m, domain)
end

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
_camera_of_chart(c) = throw(ArgumentError("RotationBundleProblem: the camera variable must be its _CameraVariable, got a $(typeof(c))"))
@inline _pixel_residual(r::_CameraObservation, ch::VgtkCore.RotationChart, d::VgtkCore.Eliminable, c) =
    _project_model(_camera_of_chart(c), c.domain, _camera_direction(ch, d)) - r.x
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
function NLLSsolver.computeresjac(varflags, r::_CameraObservation, ch::VgtkCore.SO3Manifold{Float64}, d::VgtkCore.Eliminable, c)
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
function NLLSsolver.computeresjac(varflags, r::_Observation, ch::VgtkCore.SO3Manifold{Float64}, d::VgtkCore.Eliminable)
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

`FitResult` metadata of the rotation bundle ([`RotationBundleProblem`](@ref)): `directions` (one unit
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

and `calibration`, `nothing` when the camera is held, else the autocalibration's information at the result (the
camera refined: the guess `(; camera, rotations)`):

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
struct RotationBundleMetadata{K, I}
    directions::Vector{SVector{3, Float64}}
    residuals::Vector{Vector{Float64}}
    loss_width::Float64
    inlier_rms::Float64
    scale::Float64
    scale_source::VgtkCore.ScaleSource
    n_observations::Int
    n_dropped_start::Int
    frames_unconstrained::Vector{K}
    calibration::I
end

"""
    RotationBundleProblem(camera, tracks; eliminate = <NLLSsolver's Schur system>)
    RotationBundleProblem(tracks; fixed = (), eliminate = <NLLSsolver's Schur system>)

The `f` of the rotation bundle of a pure-rotation sequence (VisualGeometryToolkitCore's block contract,
`AbstractBlockProblem`), solved from a guess by Core's `LevenbergMarquardt` (its default):

    solve(RefineProblem(RotationBundleProblem(camera, tracks), rotations), LevenbergMarquardt(; optimization_strategy);
          scale | width, stops)                                       -> FitResult{Dict{K, RotMatrix{3, Float64}}, RotationBundleMetadata}
    solve(RefineProblem(RotationBundleProblem(tracks; fixed), (; camera, rotations)), LevenbergMarquardt(; optimization_strategy);
          scale | width, stops)                                       -> FitResult{@NamedTuple{camera, rotations}, RotationBundleMetadata}

What the guess holds is refined. With the camera in the problem (the first form) its calibration is HELD: a rotation-only
bundle of the calibrated `camera` (a `CameraModel` or a `BoundedCamera`; its `backproject` and
`backproject_with_jacobian` give the start and the Jacobian), projected through its closed-form forward map π, its
model's `imaging_transform` (a grid lens's `SplineLens` through its forward spline: give a SplineLens camera one with
VisualGeometryToolkitEstimation's `ForwardSpline(; tolerance)` on `CameraRegressionProblem(camera)`, which keeps its
backward, K and frame, while `CameraRegression` of a SplineLens target turns a gauge-K camera onto its optical axis, a new
frame). With the camera in the guess (the second form) the bundle is the AUTOCALIBRATION of a rotating camera: no target,
a pure-rotation sequence (Hartley 1994; Agapito, Hayman and Reid 2001) — the same observations, start, kept set, gauge
and result, with the camera's free parameters one more variable: VisualGeometryToolkitCore's `ParameterManifold` of the
camera's model restricted to them (the camera block of Estimation's `CameraPosesProblem`), its Jacobian ForwardDiff
through the chart. The observations kept at the start are those the GIVEN camera images; the residuals, the loss width
and the result are the refined camera's; the result's `u` is `(; camera, rotations)`, the camera of the guess's type
(for a `BoundedCamera`, its sensor and lens coverage kept), and the metadata's `calibration` the parameters' information
([`RotationBundleMetadata`](@ref)).

A residual is NaN off the camera model's `valid_domain` (a grid lens's lattice), where its back-projection Jacobian is
undefined; the lens coverage does not enter the bundle. The rotations `R_k` (`d = R_k·f_k`) of every frame but the first
of the sorted keys of `rotations` (the gauge, held fixed) and one direction `d_j` per track minimize the robust sum of the
pixel reprojection errors `π(R_kᵀ d_j) − x_jk` of every observation. A refine: it starts from `rotations` (a rotation
averaging of the view graph, VisualGeometryToolkitEstimation's `RotationAveraging`) and each track's mean bearing in
the common frame.

An observation whose start prediction is not imaged (its residual or Jacobian not finite at the start: the prediction
off the model's valid domain) is dropped, a track left in fewer than two frames is dropped whole (it keeps its start
direction, held), and a partly kept track restarts from the mean bearing of the observations it keeps, its observations
tested again until none is dropped: the solve is that of the kept observations alone. A frame other than the gauge left
with no kept observation is held at its start rotation. The result's `n` is the number of observations kept; its
metadata's `n_observations` the number given and `n_dropped_start` their difference.

- `tracks` — `AbstractVector` of tracks, each an `AbstractVector` of `frame => pixel` observations (pixels in the
  camera's frame), two frames at least. A pixel may be VisualGeometryToolkitCore's `Uncertain(pixel, CovMat(Σ))`, its
  2 × 2 covariance `Σ = LLᵀ` (finite, positive definite, up to one global scale): the residual is then whitened,
  `L⁻¹(π(R_kᵀ d_j) − x_jk)`, and the loss acts on the whitened residual. Every observation carries a covariance or none
  does.
- `fixed` (the autocalibration) — the camera parameters HELD, by name or by a coefficient family's group (`:k` for
  `k1…`, `:d`, `:p`, `:sx`, `:sy`): VisualGeometryToolkitCore's `free_parameter_names(model, fixed)`. The default `()`
  frees every parameter but the skew, which the stack holds by construction (named in `fixed` it is a no-op). An
  unknown or repeated entry, a camera with nothing free and a SplineLens camera (its controls have no static chart) are
  an `ArgumentError` at the solve.
- `eliminate` — whether NLLSsolver eliminates the directions (their Schur complement), which needs the loaded
  NLLSsolver's Schur system (`MultiVariateLSschur`): the default where it has one;
  `false` solves every unknown together, the default otherwise. `eliminate = true` without the Schur system is an
  `ArgumentError` naming it.

The solve: `LevenbergMarquardt`'s `optimization_strategy` is the robust strategy of the robust-estimation contract, a
`Triggs{K}` (VisualGeometryToolkitCore's; `ClassDefault()`, VisualGeometryToolkitEstimation's pixel bundle's,
configs/default.toml `[camera_poses_problem]`: `Triggs{WelschShape}` at the shape's efficiency constant for a 2-D
residual), its loss `K` as NLLSsolver's robustifier (`loss_robustifier`) at ONE width fixed for the bundle's one
NLLSsolver run, or `nothing`, the plain least squares (no scale or width). Exactly one of `scale = σ` (the observations'
noise scale, pixels or whitened units, or an upstream robust result: [`register_rotations`](@ref) hands the verified
pairs' RANSAC scales pooled by Estimation's `PooledScale`) — the width is the strategy's efficiency constant × σ — or
`width = w` (the loss width). `stops` (Core's `LmStops`: `max_iter`, `ftol` the relative cost decrease, `xtol` the
largest step in chart units, radians for a rotation or a direction) are the LM's, Core's multi-block LM of the block
contract (`block_structure`, `block_result`); the result's `stats.nsteps` the LM iterations it took.

A refine descends towards the local minimum its start leads to, until a stop; the result's termination record (`retcode`
and `stop_rule`: a `ReturnCode.T` and a `StopRule`; `successful_retcode(result)` its success) says how the bundle ended:
Core's LM verdict, returned as it is (an `ftol` or `xtol` stop is `Success` with no stationarity test, so
a loose tolerance converges short of the minimum; a step-size stop at a non-finite step is `Stalled` with `NonFinite`;
the iteration cap `MaxIters`); `max_iter = 0` (no solve: the start's rotations are the result) `MaxIters` with
`IterationCap`. A track observed in fewer than two frames, or naming a frame without a start rotation, some
observations with a covariance and some without, a covariance that is not finite and positive definite, and a start
that keeps no observation (every track left in fewer than two imaged frames) are an `ArgumentError`.
"""
struct RotationBundleProblem{C, T <: AbstractVector, F} <: VgtkCore.AbstractBlockProblem
    camera::C                    # the held camera, or nothing (the autocalibration: the camera is the guess's)
    tracks::T
    fixed::F                     # the autocalibration's held parameters; nothing with a held camera
    eliminate::Bool
end
RotationBundleProblem(camera::Union{VgtkCore.CameraModel, VgtkCore.BoundedCamera}, tracks::AbstractVector; eliminate::Bool = VgtkCore.has_nlls_schur()) =
    RotationBundleProblem(camera, tracks, nothing, eliminate)
RotationBundleProblem(tracks::AbstractVector; fixed::Tuple{Vararg{Symbol}} = (), eliminate::Bool = VgtkCore.has_nlls_schur()) =
    RotationBundleProblem(nothing, tracks, fixed, eliminate)

# VisualGeometryToolkitEstimation's configuration of record (its configs/default.toml, read through its version converter):
# a strategy default the bundles share with Estimation is looked up there, never restated here.
const _ESTIMATION_DEFAULTS = VgtkCore.parse_defaults(VgtkEstimation.default_config_path(); versions = VgtkEstimation.ESTIMATION_CONFIG_VERSIONS)
_estimation_strategy(table::AbstractString, key::AbstractString, class::Symbol) =
    VgtkEstimation.optimization_strategy_from_toml(VgtkCore.toml_default(_ESTIMATION_DEFAULTS, table, key);
                                                   class, section = "$table.$key")

# The default strategy of the rotation bundle (`ClassDefault()`): Estimation's pixel bundle's, [camera_poses_problem]
# `optimization_strategy` (Triggs{Welsch} stating no efficiency constant: its width is Core's efficiency constant for a 2-D
# residual, a pixel reprojection error, times the caller's `scale`).
const _DEFAULT_BUNDLE_STRATEGY = _estimation_strategy("camera_poses_problem", "optimization_strategy", :camera_poses_problem)
VgtkCore.default_optimization_strategy(::RotationBundleProblem) = _DEFAULT_BUNDLE_STRATEGY

# The block contract (Core's `block_structure`, `fixed_variables` and `block_result`): the bundle stated once at the guess —
# EVERY variable, its blocks of the observations kept, and the rest of the statement (`stated`: which are held, the kept
# observations) — the held variables declared at the guess (`fixed_variables(f, u0)`: the gauge frame, the tracks left out,
# the frames left with no kept observation; they depend on the start), and the result assembled from the solved variables
# and the summary's parsed refine (`summary.loss`). The held camera: the guess is the rotations.
VgtkCore.block_structure(f::RotationBundleProblem, u0) = _rotation_bundle_blocks(_rotation_bundle_statement(f, u0))
VgtkCore.fixed_variables(f::RotationBundleProblem, u0) = findall(!, _rotation_bundle_statement(f, u0).unfixed)
_rotation_bundle_statement(f::RotationBundleProblem, rotations::AbstractDict) =
    _rotation_bundle_statement(f.camera, rotations, f.tracks, nothing, (); eliminate = f.eliminate)
_rotation_bundle_statement(::RotationBundleProblem{Nothing}, ::AbstractDict) =
    throw(ArgumentError("RotationBundleProblem: the autocalibration (no camera in the problem) refines the guess " *
                        "(; camera, rotations); give the camera to the problem to hold it"))
# The autocalibration: the camera is the guess's, refined over its free parameters.
function _rotation_bundle_statement(f::RotationBundleProblem{Nothing}, u0::NamedTuple{(:camera, :rotations)})
    camera = u0.camera
    m = camera isa VgtkCore.BoundedCamera ? VgtkCore.model(camera) : camera
    # the camera's variable: Core's ParameterManifold of its model restricted to the free parameters (names checked by
    # `free_parameter_names`); a SplineLens camera (a gauge K: its parameters are its lattice's controls, which the stack
    # refines only by back-projection, or holds) is refused
    (m isa VgtkCore.CameraModel && !(m.lens isa VgtkCore.SplineLens)) ||
        throw(ArgumentError("RotationBundleProblem: the autocalibration needs a parametric camera model (its parameters by name); this camera's lens is a $(nameof(typeof(m.lens)))"))
    names = VgtkCore.free_parameter_names(m, f.fixed)
    isempty(names) && throw(ArgumentError(
        "RotationBundleProblem: `fixed` holds every camera parameter: nothing to calibrate (give the camera to the problem " *
        "to hold its calibration)"))
    chc = VgtkCore.restrict(VgtkCore.ParameterManifold(m), names)
    return _rotation_bundle_statement(camera, u0.rotations, f.tracks, chc, names; eliminate = f.eliminate)
end
VgtkCore.block_result(::RotationBundleProblem, _u0, r::VgtkCore.FitResult, summary) = _rotation_bundle_result(summary.stated, r, summary.loss)

# A block robustified for the LM (Core's `with_robustifier`): its robust kernel replaced. Its rows carry their whitening
# (the observation's covariance), never a weight: a √weight other than one is refused.
function _unit_sqrt_weight(sqrt_weight)
    sqrt_weight == 1 || throw(ArgumentError("RotationBundleProblem: a block's rows are whitened by its observation's covariance, " *
                                            "never weighted; got √weight $sqrt_weight"))
    return nothing
end
VgtkCore.with_robustifier(o::_Observation{E, C, P, L, W}, sqrt_weight, kernel::K) where {E, C, P, L, W, K} =
    (_unit_sqrt_weight(sqrt_weight); _Observation{E, C, P, K, W}(o.camera, o.projection, o.x, o.varind, kernel, o.whitening))
VgtkCore.with_robustifier(o::_CameraObservation{E, L, W}, sqrt_weight, kernel::K) where {E, L, W, K} =
    (_unit_sqrt_weight(sqrt_weight); _CameraObservation{E, K, W}(o.x, o.varind, kernel, o.whitening))

# The one rotation bundle's statement (`chc === nothing`: the camera held; else `chc` the camera's chart over its free
# parameters `free_names`): the observations kept at the start (a fixed point, below), the variables (every frame's rotation,
# every track's direction, the camera last) and the held ones (the gauge frame, the tracks left out, the frames left with no
# kept observation: `unfixed` false), which the LM holds (Core's `fixed_variables`, NLLSsolver's unfixed mask).
function _rotation_bundle_statement(camera, rotations::AbstractDict{K}, tracks::AbstractVector, chc, free_names::Tuple{Vararg{Symbol}};
                                    eliminate::Bool) where {K}
    what = "RotationBundleProblem"
    fwd = _forward_map(camera); bc = _BundleCamera(camera); C = typeof(bc); P = typeof(fwd)
    frames = sort!(collect(keys(rotations)))
    fid = Dict(k => v for (v, k) in enumerate(frames))
    eliminate && !VgtkCore.has_nlls_schur() && throw(ArgumentError(
        "$what: eliminate = true needs NLLSsolver's Schur system (MultiVariateLSschur), which the loaded NLLSsolver does not " *
        "have; leave `eliminate` out (every unknown solved together)"))
    E = eliminate && VgtkCore.has_nlls_schur()   # = eliminate after the refusal; a constant false without Schur
    cv = chc === nothing ? nothing : _CameraVariable(chc)
    V = chc === nothing ? Union{VgtkCore.SO3Manifold{Float64}, VgtkCore.Eliminable{E, VgtkCore.SphereManifold{3, 2, Float64, 6, typeof(identity)}}} :
        Union{VgtkCore.SO3Manifold{Float64}, VgtkCore.Eliminable{E, VgtkCore.SphereManifold{3, 2, Float64, 6, typeof(identity)}}, typeof(cv)}
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
    return (; camera, tracks, chc, cv, free_names, E, frames, fid, nf, ic, vars, unfixed, active, nkept, nobs,
            unconstrained, observation)
end
# The bundle's variables (every one: Core holds the held ones), its blocks of the kept observations (each kernel NLLSsolver's
# `NoRobust()`; Core robustifies them at the solve's loss, `with_robustifier`) and the statement, from `st`.
function _rotation_bundle_blocks(st)
    (; tracks, active, observation) = st
    blocks = NLLSsolver.AbstractResidual[]
    for (j, t) in enumerate(tracks), (i, o) in enumerate(t)
        active[j][i] && push!(blocks, observation(NLLSsolver.NoRobust(), j, i, o))
    end
    return copy(st.vars), blocks, st
end

# The one rotation bundle's result from its statement `st`, the LM's result `r` (`r.u` its free variables; no solve at
# `max_iter = 0`: Core's `(MaxIters, IterationCap)`, no step) and the parsed refine of the robust-estimation contract,
# Core's `bundle_loss` of the solve's strategy and scale input (`(; strategy, input, loss)`). Its strategy is a Triggs{K} —
# the loss K as NLLSsolver's robustifier at the deployed width, `sqrt(sq_loss_width(loss))` as every bundle of the stack —
# or `nothing` (plain least squares); an Irls{K} is refused by Core's bundle_loss: this bundle runs one LM. The scale record
# is Core's `scale_record` of it (the input's scale and source, the loss's width; NaN for least squares).
function _rotation_bundle_result(st, r::VgtkCore.FitResult, bundle::NamedTuple)
    (; camera, tracks, chc, cv, free_names, E, frames, fid, nf, ic, active, nkept, nobs, unconstrained, observation) = st
    loss = bundle.strategy === nothing ? nothing : VgtkCore.strategy_loss(bundle.strategy)()
    (; scale, scale_source, loss_width) = VgtkCore.scale_record(bundle)   # Core's record of the parsed refine
    width = loss_width
    robustifier() = loss === nothing ? NLLSsolver.NoRobust() : VgtkCore.loss_robustifier(loss, width)
    vars = r.u                                                              # every variable, the held ones as stated
    solved = !(r.stats.nsteps == 0 && r.stop_rule == VgtkCore.IterationCap)   # `max_iter = 0` runs no solve
    out = Dict(k => RotMatrix{3, Float64, 9}(SMatrix{3, 3, Float64, 9}((vars[fid[k]]::VgtkCore.SO3Manifold{Float64}).R)) for k in frames)
    dirs = [SVector{3, Float64}(VgtkCore.model((vars[nf + j]::VgtkCore.Eliminable{E}).value)) for j in eachindex(tracks)]
    rn = [[active[j][i] ? (ob = observation(nothing, j, i, o); norm(_pixel_residual(ob, NLLSsolver.getvars(ob, vars)...))) : NaN
           for (i, o) in enumerate(t)] for (j, t) in enumerate(tracks)]
    flat = filter(isfinite, reduce(vcat, rn; init = Float64[]))
    calib = chc === nothing ? nothing :
        _autocalibration((vars[ic]::typeof(cv)).chart, free_names, [(j, i, o) for (j, t) in enumerate(tracks) for (i, o) in enumerate(t) if active[j][i]],
                          (j, i, o) -> observation(solved ? robustifier() : NLLSsolver.NoRobust(), j, i, o),
                          vars, fid, nf, unconstrained)
    u = chc === nothing ? out : (; camera = VgtkCore.with_model(camera, _camera_of_chart(vars[ic])), rotations = out)
    return VgtkCore.FitResult(u, r.retcode, r.stop_rule, VgtkCore.SolveStats(; nsteps = r.stats.nsteps),
                              VgtkCore.rms(flat, max(length(flat), 1)), nkept,   # 0 with none kept, as before
                              RotationBundleMetadata(dirs, rn, width, _inlier_rms(filter(≤(width), flat)), scale, scale_source, nobs, nobs - nkept, unconstrained, calib))
end

# =============================================================================
# The track check — the bundle re-solved after its far observations are removed
# =============================================================================

"""
    TrackCheck(refine = LevenbergMarquardt(); cut_multiple = 3.035, max_outer_iterations = 10)

A VisualGeometryToolkitCore `RefineKind`: the rotation bundle refined by `refine` (Core's `LevenbergMarquardt`), then its
observations whose residual exceeds `cut_multiple` × σ̂ removed — a track left in fewer than two frames with them — and the
bundle refined again from its result, until a solve leaves none past the cut or `max_outer_iterations` solves have removed
some (COLMAP's `filter_max_reproj_error` loop, Hugin's cpclean):

    solve(RefineProblem(RotationBundleProblem(camera, tracks), rotations), TrackCheck(LevenbergMarquardt(; optimization_strategy));
          scale, stops) -> FitResult

σ̂ is the solve's `scale` (the bundle's noise scale, its `noise_scale`; a `width` alone gives none: an `ArgumentError`), `stops`
the bundle's `LmStops`. The cut: a residual is 2-D, ‖r‖²/σ² ~ χ²₂ for an inlier, so P(‖r‖ > kσ) = exp(−k²/2); the default
k = √χ²₂(0.99) = √9.21 = 3.035 removes 1 % of the inliers at the true σ. The cap: each solve after the first refits only the
observations that remain, so the removals shrink geometrically; 10 bounds a sweep whose removals do not.

The result is the last bundle's (`u`, termination record, `stats`, `rms`, `n`, `resid`) with `metadata = (; bundle, history,
kept)`: `bundle` that last bundle's own result (its [`RotationBundleMetadata`](@ref)), `history` per solve the check tested
its `removed` count, `rms` and `inlier_rms` (px), and `kept` the observations the last bundle solved, `j => indices` per
track `j` of the problem's `tracks` it kept (the tracks left out absent). `cut_multiple` must be positive and
`max_outer_iterations` at least 1 (an `ArgumentError`). The observations must be plain pixels: an `Uncertain` one (whitened
by its covariance) is refused (an `ArgumentError`), since the cut compares the bundle's pixel residuals with its noise scale,
which a whitened bundle states in units of the covariances, until the check whitens its residuals.
"""
struct TrackCheck{A <: VgtkCore.LevenbergMarquardt} <: VgtkCore.AbstractEstimationAlgorithm
    refine::A
    cut_multiple::Float64
    max_outer_iterations::Int
    function TrackCheck(refine::A, cut_multiple::Float64, max_outer_iterations::Int) where {A <: VgtkCore.LevenbergMarquardt}
        (isfinite(cut_multiple) && cut_multiple > 0) ||
            throw(ArgumentError("TrackCheck: cut_multiple = $cut_multiple is not positive (a multiple of the bundle's noise scale)"))
        max_outer_iterations >= 1 || throw(ArgumentError("TrackCheck: max_outer_iterations = $max_outer_iterations is not at least 1"))
        return new{A}(refine, cut_multiple, max_outer_iterations)
    end
end
TrackCheck(refine::VgtkCore.LevenbergMarquardt = VgtkCore.LevenbergMarquardt(); cut_multiple::Real = 3.035,
           max_outer_iterations::Integer = 10) = TrackCheck(refine, Float64(cut_multiple), Int(max_outer_iterations))
VgtkCore.estimator_kind(::TrackCheck) = VgtkCore.RefineKind()

function VgtkCore.__solve(prob::VgtkCore.RefineProblem{<:RotationBundleProblem}, alg::TrackCheck; kwargs...)
    f = prob.f
    tracks = f.tracks
    # the cut compares each observation's pixel residual (unwhitened: the metadata's `residuals`) with a multiple of the
    # bundle's noise scale; a whitened observation's scale is in units of its covariance, not pixels: refused until the
    # check whitens its residuals
    any(t -> any(o -> last(o) isa VgtkCore.Uncertain, t), tracks) &&
        throw(ArgumentError("TrackCheck: an observation is Uncertain (whitened by its covariance): the check compares its " *
                            "unwhitened pixel residual with the bundle's whitened noise scale; give plain pixels"))
    sub(kept) = RotationBundleProblem(f.camera, [tracks[j][idx] for (j, idx) in kept], f.fixed, f.eliminate)
    kept = [j => collect(eachindex(t)) for (j, t) in enumerate(tracks)]
    bundle = VgtkCore.solve(VgtkCore.RefineProblem(f, prob.u0, prob.p), alg.refine; kwargs...)
    σ = VgtkCore.scale_source(bundle) == VgtkCore.ScaleNone ? NaN : Float64(VgtkCore.noise_scale(bundle))
    isfinite(σ) || throw(ArgumentError("TrackCheck: the cut is a multiple of the bundle's noise scale: give the solve its `scale`"))
    cut = alg.cut_multiple * σ
    history = NamedTuple{(:removed, :rms, :inlier_rms), Tuple{Int, Float64, Float64}}[]
    for _ in 1:alg.max_outer_iterations
        # NaN (an observation left out at the bundle's start) is not past the cut: kept
        next = [j => [i for (i, r) in zip(idx, rt) if !(r > cut)] for ((j, idx), rt) in zip(kept, bundle.metadata.residuals)]
        next = filter(((j, idx),) -> length(unique(first.(tracks[j][idx]))) >= 2, next)
        nrem = sum(p -> length(p.second), kept; init = 0) - sum(p -> length(p.second), next; init = 0)
        push!(history, (; removed = nrem, rms = bundle.rms, inlier_rms = bundle.metadata.inlier_rms))
        nrem == 0 && break
        kept = next
        bundle = VgtkCore.solve(VgtkCore.RefineProblem(sub(kept), bundle.u, prob.p), alg.refine; kwargs...)
    end
    return VgtkCore.FitResult(bundle.u, bundle.retcode, bundle.stop_rule, bundle.stats, bundle.rms, bundle.n,
                              (; bundle, history, kept); resid = bundle.resid)
end

# The inliers' rms (the residual norms within the loss width; NaN with none), Core's `rms`
_inlier_rms(r) = isempty(r) ? NaN : VgtkCore.rms(r)

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
    singular() = throw(ArgumentError("RotationBundleProblem: the autocalibration's information is singular at the result"))
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
