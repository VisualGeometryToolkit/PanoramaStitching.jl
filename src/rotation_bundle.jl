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

# One observation: √w (π(R_kᵀ d_j) − x_jk), pixels, through the fixed camera.
struct _Observation{E, C, L} <: NLLSsolver.AbstractResidual
    camera::C
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

# π(f) through any camera: a generic model's projection is a pixel-space LM, started at the guess `x` (the observed pixel in the
# bundle, the cell's predicted pixel in `sample_cells`); a parametric model projects in closed form and takes no guess.
@inline _project_with(camera::VgtkCore.CentralGenericModel, f, x) = VgtkCore.project(camera, f; init = x)
@inline _project_with(camera, f, x) = VgtkCore.project(camera, f)
@inline function _project(r::_Observation, f)
    u = _project_with(r.camera, f, r.x)
    return u === nothing ? SVector(NaN, NaN) : SVector(u[1], u[2])
end

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
`loss_width` (px, of the last NLLSsolver run), `n_observations`.
"""
struct RotationBundleMetadata
    directions::Vector{SVector{3, Float64}}
    residuals::Vector{Vector{Float64}}
    loss_width::Float64
    n_observations::Int
end

"""
    refine_rotations(camera, rotations, tracks; loss, scale_multiple, iterations, max_iter, ftol, xtol, eliminate = true)
        -> FitResult{Dict{K, RotMatrix{3, Float64}}, RotationBundleMetadata}

The rotation bundle of a pure-rotation sequence seen by the calibrated `camera`
(any camera with `project(camera, f; init)` and `backproject_with_jacobian`,
fixed): the rotations `R_k` (`d = R_k·f_k`) of every frame but the first of
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
"""
function refine_rotations(camera, rotations::AbstractDict{K}, tracks::AbstractVector;
                          loss::VgtkCore.AbstractLossShape, scale_multiple::Real, iterations::Integer,
                          max_iter::Integer, ftol::Real, xtol::Real, eliminate::Bool = true) where {K}
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
    residual_norms() = [[norm(NLLSsolver.computeresidual(_Observation{E, typeof(camera), Nothing}(camera, SVector{2, Float64}(last(o)), SVector(fid[first(o)], length(frames) + j), nothing),
                                                         vars[fid[first(o)]]::_FrameChart, vars[length(frames) + j]::_TrackDirection{E})) for o in t]
                        for (j, t) in enumerate(tracks)]
    result = nothing
    for _ in 1:iterations
        rn = reduce(vcat, residual_norms())
        σ = Statistics.median(filter(isfinite, rn)) / sqrt(VgtkCore.chisq_median(2))
        width = scale_multiple * σ
        kernel = VgtkCore.loss_robustifier(loss, width)
        problem = NLLSsolver.NLLSProblem(V, _Observation{E, typeof(camera), typeof(kernel)})
        for v in vars
            NLLSsolver.addvariable!(problem, v)
        end
        for (j, t) in enumerate(tracks), o in t
            NLLSsolver.addcost!(problem, _Observation{E, typeof(camera), typeof(kernel)}(camera, SVector{2, Float64}(last(o)), SVector(fid[first(o)], length(frames) + j), kernel))
        end
        result = NLLSsolver.optimize!(problem, VgtkCore.nlls_options(; max_iter, ftol, xtol), unfixed)
        vars = problem.variables
    end
    code = result === nothing ? :max_iterations : VgtkCore.lm_retcode(VgtkCore.nlls_stop_code(result.termination))   # NLLSsolver's stop, decoded (no stall test here)
    out = Dict(k => RotMatrix{3, Float64, 9}(SMatrix{3, 3, Float64, 9}((vars[fid[k]]::_FrameChart).R)) for k in frames)
    dirs = [SVector{3, Float64}(VgtkCore.model((vars[length(frames) + j]::_TrackDirection{E}).ch)) for j in eachindex(tracks)]
    rn = residual_norms(); flat = filter(isfinite, reduce(vcat, rn; init = Float64[]))
    return VgtkCore.FitResult(out, code, iterations, sqrt(sum(abs2, flat) / max(length(flat), 1)), nobs,
                              RotationBundleMetadata(dirs, rn, width, nobs))
end
