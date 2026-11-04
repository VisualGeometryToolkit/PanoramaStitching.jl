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

# A track's direction, a 2-dof sphere chart; `E` says whether NLLSsolver may eliminate it.
struct _TrackDirection{E, C}
    ch::C
end
_TrackDirection{E}(ch::C) where {E, C} = _TrackDirection{E, C}(ch)
NLLSsolver.nvars(::_TrackDirection) = static(2)
NLLSsolver.update(d::_TrackDirection{E}, δ, start = 1) where {E} =
    _TrackDirection{E}(VgtkCore.recenter(d.ch, SVector(δ[start], δ[start + 1])))
const _SCHUR = VgtkCore.nlls_features().schur
if _SCHUR
    NLLSsolver.iseliminable(::Type{<:_TrackDirection{true}}) = static(true)
end

# One observation: √w (π(R_kᵀ d_j) − x_jk), pixels, through the fixed camera: `projection` its forward map π (`_forward_map`),
# `camera` its back-projection (the Jacobian's).
struct _Observation{E, C, P, L} <: NLLSsolver.AbstractResidual
    camera::C
    projection::P
    x::SVector{2, Float64}
    varind::SVector{2, Int}
    kernel::L
end
NLLSsolver.ndeps(::_Observation) = static(2)
NLLSsolver.nres(::_Observation) = static(2)
NLLSsolver.varindices(r::_Observation) = r.varind
NLLSsolver.getvars(r::_Observation{E}, vars::Vector) where {E} = (vars[r.varind[1]]::_FrameChart, vars[r.varind[2]]::_TrackDirection{E, _SphereChart})
NLLSsolver.robustkernel(r::_Observation) = r.kernel
Base.eltype(::_Observation) = Float64

@inline _camera_direction(ch::VgtkCore.RotationChart, d::_TrackDirection) = SMatrix{3, 3}(ch.R)' * VgtkCore.model(d.ch)

# π(f): the forward map in closed form, NaN where it does not image AND where the camera does not image the point — off a grid
# model's imaged region (its calibrated area ∩ domain), where its back-projection Jacobian is NaN. A grid's SplineProjection
# extrapolates past its region (a finite pixel), so without the second test the residual would be finite where its Jacobian
# is not: NLLSsolver would accept such an iterate, linearize there and loop on its NaN system for ever (measured 2026-11-04,
# the synthetic Tamron sequence from its true rotations). With it the residual and its Jacobian are defined on the same set,
# an iterate past it is a NaN cost (a rejected step) and a start past it is a non-finite start (`:nonfinite_start`).
@inline function _project(r::_Observation, f)
    u = r.projection(f); p = SVector(u[1], u[2])
    return _images(r.camera, p) ? p : SVector(NaN, NaN)
end
# Whether the camera images the sensor point `u` (its back-projection and Jacobian defined there): a grid model's imaged region;
# a parametric camera's back-projection is defined on the whole plane.
@inline _images(camera::VgtkCore.CentralGenericModel, u) = VgtkCore.Point2(u[1], u[2]) ∈ VgtkCore.imaged_region(camera)
@inline _images(camera::VgtkCore.BoundedCamera, u) = _images(VgtkCore.model(camera), u)
@inline _images(camera, u) = true

NLLSsolver.computeresidual(r::_Observation, ch::VgtkCore.RotationChart, d::_TrackDirection) = _project(r, _camera_direction(ch, d)) - r.x

# The closed-form 2 × 5 Jacobian at the variables' anchors: ∂π/∂f on the tangent plane is the pseudo-inverse of the
# camera's back-projection Jacobian M at π(f) (`backproject_with_jacobian`), f = R_kᵀ·exp(−ω)·d moves by R_kᵀ[d]×·dω
# and by R_kᵀ·∂d/∂v along the direction's chart.
# `varflags` (NLLSsolver's): bit 1 the frame, bit 2 the direction — only the free variables' columns are returned (the
# gauge frame is fixed).
function NLLSsolver.computeresjac(varflags, r::_Observation, ch::_FrameChart, d::_TrackDirection)
    R = SMatrix{3, 3, Float64, 9}(ch.R); dir = VgtkCore.model(d.ch); f = R' * dir
    u = _project(r, f)
    _, M = VgtkCore.backproject_with_jacobian(r.camera, u)
    Mp = (M' * M) \ M'
    Jω = Mp * (R' * VgtkCore.skew(dir))
    Jv = Mp * (R' * VgtkCore.retract_jacobian(d.ch))
    e = u - r.x
    varflags == 3 && return e, hcat(Jω, Jv)
    return varflags == 2 ? (e, Jv) : (e, Jω)
end

"""
    RotationBundleMetadata

`FitResult` metadata of [`refine_rotations`](@ref): `directions` (one unit
direction per track, the common frame), `residuals` (per track, the pixel
reprojection error norm of each of its observations at the result),
`loss_width` (px, of the last NLLSsolver run), `n_observations`,
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
map π: the `camera` itself when it is a `CameraModel` (its
`imaging_transform`), or a `SplineProjection` — for a grid model, which has no
closed-form forward map, its `VisualGeometryToolkitEstimation.fit_camera(SplineProjection,
grid, region; tolerance)` (an `ArgumentError` names it when missing). The
rotations `R_k` (`d = R_k·f_k`) of every frame but the first of
the sorted keys of `rotations` (the gauge, held fixed) and one direction `d_j`
per track minimizing the robust sum of the pixel reprojection errors
`π(R_kᵀ d_j) − x_jk` of every observation. A refine: it starts from
`rotations` (a rotation averaging of the view graph,
`VisualGeometryToolkitEstimation.fit_rotation_averaging`) and each track's mean
bearing in the common frame.

- `tracks` — `AbstractVector` of tracks, each an `AbstractVector` of `frame =>
  pixel` observations (pixels in the camera's frame), two frames at least.
- `loss`, `scale_multiple` — the stack loss shape (NLLSsolver's robustifier,
  `loss_robustifier`) and its width as a multiple of the Rayleigh scale of the
  reprojection error norms at the start of each of the `iterations` NLLSsolver
  runs (`max_iter`, `ftol`, `xtol` each: `nlls_options`).
- `eliminate` — whether NLLSsolver may eliminate the directions (the Schur
  complement, where the loaded NLLSsolver has it); `false` solves every unknown
  together. Both reach the same optimum.

A refine ends in the local minimum its start leads to; the result's `retcode`
and its metadata's `outcome` ([`RotationBundleMetadata`](@ref)) say how the
bundle ended. A track observed in fewer than two frames, or naming a frame
without a start rotation, is an `ArgumentError`.
"""
function refine_rotations(camera, rotations::AbstractDict{K}, tracks::AbstractVector; projection = camera,
                          loss::VgtkCore.AbstractLossShape, scale_multiple::Real, iterations::Integer,
                          max_iter::Integer, ftol::Real, xtol::Real, eliminate::Bool = true) where {K}
    fwd = _forward_map(projection); C = typeof(camera); P = typeof(fwd)
    frames = sort!(collect(keys(rotations)))
    fid = Dict(k => v for (v, k) in enumerate(frames))
    E = eliminate && _SCHUR
    V = Union{_FrameChart, _TrackDirection{E, _SphereChart}}
    nobs = 0
    for t in tracks
        length(unique(first.(t))) >= 2 || throw(ArgumentError("refine_rotations: a track needs observations in two frames"))
        all(o -> haskey(fid, first(o)), t) || throw(ArgumentError("refine_rotations: a track names a frame with no start rotation"))
        nobs += length(t)
    end
    # the start: the frames' rotations, each track's mean bearing in the common frame
    vars = V[VgtkCore.SO3Manifold(SMatrix{3, 3, Float64, 9}(rotations[k])) for k in frames]
    for t in tracks
        m = sum(SMatrix{3, 3, Float64, 9}(rotations[first(o)]) * VgtkCore.backproject(camera, SVector{2, Float64}(last(o))) for o in t)
        push!(vars, _TrackDirection{E}(VgtkCore.SphereManifold(m)))
    end
    unfixed = trues(length(vars)); unfixed[1] = false                       # the gauge frame
    width = NaN
    residual_norms() = [[norm(NLLSsolver.computeresidual(_Observation{E, C, P, Nothing}(camera, fwd, SVector{2, Float64}(last(o)), SVector(fid[first(o)], length(frames) + j), nothing),
                                                         vars[fid[first(o)]]::_FrameChart, vars[length(frames) + j]::_TrackDirection{E})) for o in t]
                        for (j, t) in enumerate(tracks)]
    # the start's finiteness: the observations whose residual or Jacobian is not finite there (the bundle's outcome)
    start_nonfinite = 0
    for (j, t) in enumerate(tracks), o in t
        r = _Observation{E, C, P, Nothing}(camera, fwd, SVector{2, Float64}(last(o)), SVector(fid[first(o)], length(frames) + j), nothing)
        e, J = NLLSsolver.computeresjac(3, r, vars[fid[first(o)]]::_FrameChart, vars[length(frames) + j]::_TrackDirection{E})
        (all(isfinite, e) && all(isfinite, J)) || (start_nonfinite += 1)
    end
    start_rotations = [SMatrix{3, 3, Float64, 9}((vars[i]::_FrameChart).R) for i in eachindex(frames)]
    result = nothing
    for _ in 1:iterations
        rn = reduce(vcat, residual_norms())
        σ = Statistics.median(filter(isfinite, rn)) / sqrt(VgtkCore.chisq_median(2))
        width = scale_multiple * σ
        kernel = VgtkCore.loss_robustifier(loss, width)
        problem = NLLSsolver.NLLSProblem(V, _Observation{E, C, P, typeof(kernel)})
        for v in vars
            NLLSsolver.addvariable!(problem, v)
        end
        for (j, t) in enumerate(tracks), o in t
            NLLSsolver.addcost!(problem, _Observation{E, C, P, typeof(kernel)}(camera, fwd, SVector{2, Float64}(last(o)), SVector(fid[first(o)], length(frames) + j), kernel))
        end
        result = NLLSsolver.optimize!(problem, VgtkCore.nlls_options(; max_iter, ftol, xtol), unfixed)
        vars = problem.variables
    end
    code = result === nothing ? :max_iterations : VgtkCore.lm_retcode(VgtkCore.nlls_stop_code(result.termination))   # NLLSsolver's stop, decoded (no stall test here)
    out = Dict(k => RotMatrix{3, Float64, 9}(SMatrix{3, 3, Float64, 9}((vars[fid[k]]::_FrameChart).R)) for k in frames)
    dirs = [SVector{3, Float64}(VgtkCore.model((vars[length(frames) + j]::_TrackDirection{E}).ch)) for j in eachindex(tracks)]
    rn = residual_norms(); flat = filter(isfinite, reduce(vcat, rn; init = Float64[]))
    unmoved = all(i -> SMatrix{3, 3, Float64, 9}((vars[i]::_FrameChart).R) == start_rotations[i], eachindex(frames))
    outcome = iterations == 0 ? :not_run : code === :converged ? :converged :
              code === :solve_failed && unmoved && start_nonfinite > 0 ? :nonfinite_start : code
    return VgtkCore.FitResult(out, code, iterations, sqrt(sum(abs2, flat) / max(length(flat), 1)), nobs,
                              RotationBundleMetadata(dirs, rn, width, nobs, start_nonfinite, outcome))
end
