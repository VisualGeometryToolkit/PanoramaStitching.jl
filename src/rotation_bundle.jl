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
# an iterate past it is a NaN cost (a rejected step) and a start past it is a non-finite start (`:nonfinite_start`).
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
`loss_width` (of the last NLLSsolver run: px, or in whitened units when the
observations carry covariances), `n_observations`,
`start_nonfinite` (the observations whose residual or Jacobian is not finite
at the start: a start prediction outside the lens's domain, where the camera
has no back-projection Jacobian) and `outcome`, the bundle's outcome:

- `:not_run` — `iterations = 0` (the start's rotations are the result);
- `:converged` — the retcode is `:converged`;
- `:nonfinite_start` — NLLSsolver stopped at its start point because its start
  is not finite: the retcode is `:solve_failed`, the rotations are the start's
  bit for bit (no step was tried) and `start_nonfinite > 0`. A rectilinear
  frame whose observations reach its corners, a start (the rotation
  averaging's) off by a fraction of a degree and a grid lens calibrated over
  the sensor at most give this (the Tamron 20 mm of 2026-10-29);
- otherwise the retcode itself (`:max_iterations`, `:stalled`, `:solve_failed`, …).
"""
struct RotationBundleMetadata
    directions::Vector{SVector{3, Float64}}
    residuals::Vector{Vector{Float64}}
    loss_width::Float64
    n_observations::Int
    start_nonfinite::Int
    outcome::Symbol
end

"""
    refine_rotations(camera, rotations, tracks; projection = camera, loss, scale_multiple, iterations, max_iter, ftol, xtol,
                     eliminate = true) -> FitResult{Dict{K, RotMatrix{3, Float64}}, RotationBundleMetadata}

The rotation bundle of a pure-rotation sequence seen by the calibrated `camera`
(fixed; its `backproject` and `backproject_with_jacobian` give the start and
the Jacobian), projected through `projection`, the camera's closed-form forward
map π: the `camera` itself (its model's `imaging_transform`; a grid lens's
`SplineLens` through its forward spline, VisualGeometryToolkitEstimation's
`fit_camera(SplineLens, camera, imaged_region(camera); tolerance)`); a
`projection` that is no camera is an `ArgumentError`. A residual is NaN off
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

- `tracks` — `AbstractVector` of tracks, each an `AbstractVector` of `frame =>
  pixel` observations (pixels in the camera's frame), two frames at least. A
  pixel may be VisualGeometryToolkitCore's `Uncertain(pixel, CovMat(Σ))`, its
  2 × 2 covariance `Σ = LLᵀ` (finite, positive definite, up to one global
  scale): the residual is then whitened, `L⁻¹(π(R_kᵀ d_j) − x_jk)`, and the
  loss acts on the whitened residual. Every observation carries a covariance
  or none does.
- `loss`, `scale_multiple` — the stack loss shape (NLLSsolver's robustifier,
  `loss_robustifier`) and its width as a multiple of the Rayleigh scale of the
  residual norms (pixels, or whitened: the one global scale that absorbs the
  noise level and the model error the covariances leave out) at the start of
  each of the `iterations` NLLSsolver runs (`max_iter`, `ftol`, `xtol` each:
  `nlls_options`).
- `eliminate` — whether NLLSsolver may eliminate the directions (the Schur
  complement, where the loaded NLLSsolver has it); `false` solves every unknown
  together. Both reach the same optimum.

A refine ends in the local minimum its start leads to; the result's `retcode`
and its metadata's `outcome` ([`RotationBundleMetadata`](@ref)) say how the
bundle ended. A track observed in fewer than two frames, or naming a frame
without a start rotation, some observations with a covariance and some
without, and a covariance that is not finite and positive definite are an
`ArgumentError`.
"""
function refine_rotations(camera, rotations::AbstractDict{K}, tracks::AbstractVector; projection = camera,
                          loss::VgtkCore.AbstractLossShape, scale_multiple::Real, iterations::Integer,
                          max_iter::Integer, ftol::Real, xtol::Real, eliminate::Bool = true) where {K}
    fwd = _forward_map(projection); bc = _BundleCamera(camera); C = typeof(bc); P = typeof(fwd)
    frames = sort!(collect(keys(rotations)))
    fid = Dict(k => v for (v, k) in enumerate(frames))
    E = eliminate && _SCHUR
    V = Union{_FrameChart, VgtkCore.Eliminable{E, _SphereChart}}
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
    # the start: the frames' rotations, each track's mean bearing in the common frame
    vars = V[VgtkCore.SO3Manifold(SMatrix{3, 3, Float64, 9}(rotations[k])) for k in frames]
    for t in tracks
        m = sum(SMatrix{3, 3, Float64, 9}(rotations[first(o)]) * VgtkCore.backproject(camera, pixel(o)) for o in t)
        push!(vars, VgtkCore.Eliminable{E}(VgtkCore.SphereManifold(m)))
    end
    unfixed = trues(length(vars)); unfixed[1] = false                       # the gauge frame
    width = NaN
    observation(kernel, j, i, o) = _Observation{E, C, P, typeof(kernel), Wt}(bc, fwd, pixel(o), SVector(fid[first(o)], length(frames) + j),
                                                                              kernel, whitenings[j][i])
    residual_norms(f) = [[norm(f(observation(nothing, j, i, o), vars[fid[first(o)]]::_FrameChart, vars[length(frames) + j]::VgtkCore.Eliminable{E}))
                          for (i, o) in enumerate(t)] for (j, t) in enumerate(tracks)]
    # the start's finiteness: the observations whose residual or Jacobian is not finite there (the bundle's outcome)
    start_nonfinite = 0
    for (j, t) in enumerate(tracks), (i, o) in enumerate(t)
        e, J = NLLSsolver.computeresjac(3, observation(nothing, j, i, o), vars[fid[first(o)]]::_FrameChart, vars[length(frames) + j]::VgtkCore.Eliminable{E})
        (all(isfinite, e) && all(isfinite, J)) || (start_nonfinite += 1)
    end
    start_rotations = [SMatrix{3, 3, Float64, 9}((vars[i]::_FrameChart).R) for i in eachindex(frames)]
    result = nothing
    for _ in 1:iterations
        # the loss width: a multiple of the Rayleigh scale of the residual norms the loss acts on (pixels, or whitened: the
        # one global scale that absorbs the noise level and the model error the unit-noise covariances leave out)
        rn = reduce(vcat, residual_norms(NLLSsolver.computeresidual))
        σ = VgtkCore.noise_sigma(filter(isfinite, rn), 2)
        width = scale_multiple * σ
        kernel = VgtkCore.loss_robustifier(loss, width)
        problem = NLLSsolver.NLLSProblem(V, _Observation{E, C, P, typeof(kernel), Wt})
        for v in vars
            NLLSsolver.addvariable!(problem, v)
        end
        for (j, t) in enumerate(tracks), (i, o) in enumerate(t)
            NLLSsolver.addcost!(problem, observation(kernel, j, i, o))
        end
        result = NLLSsolver.optimize!(problem, VgtkCore.nlls_options(; max_iter, ftol, xtol), unfixed)
        vars = problem.variables
    end
    code = result === nothing ? :max_iterations : VgtkCore.lm_retcode(VgtkCore.nlls_stop_code(result.termination))   # NLLSsolver's stop, decoded (no stall test here)
    out = Dict(k => RotMatrix{3, Float64, 9}(SMatrix{3, 3, Float64, 9}((vars[fid[k]]::_FrameChart).R)) for k in frames)
    dirs = [SVector{3, Float64}(VgtkCore.model((vars[length(frames) + j]::VgtkCore.Eliminable{E}).value)) for j in eachindex(tracks)]
    rn = residual_norms(_pixel_residual); flat = filter(isfinite, reduce(vcat, rn; init = Float64[]))
    unmoved = all(i -> SMatrix{3, 3, Float64, 9}((vars[i]::_FrameChart).R) == start_rotations[i], eachindex(frames))
    outcome = iterations == 0 ? :not_run : code === :converged ? :converged :
              code === :solve_failed && unmoved && start_nonfinite > 0 ? :nonfinite_start : code
    return VgtkCore.FitResult(out, code, iterations, sqrt(sum(abs2, flat) / max(length(flat), 1)), nobs,
                              RotationBundleMetadata(dirs, rn, width, nobs, start_nonfinite, outcome))
end
