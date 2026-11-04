# =============================================================================
# fit_vignetting — the vignetting of a rotating camera from its own frames, interpolation free
# =============================================================================
#
# The model of every pixel p of every frame k inside the fit domain:
#
#     I_k(p) = g_k · L(R_k·f(p)) · V(u(p))        log I_k(p) = log g_k + log L(d) + log V(u)
#
# `f(p)` the pixel's ray (the camera's back-projection of its sensor point u(p), computed once:
# the pixel geometry is the same in every frame), `R_k` the frame's rotation (`d = R_k·f`), `L` the
# scene's radiance on the sphere, `V` the vignetting on the sensor and `g_k` the frame's exposure gain
# (`gain_correction`; g₁ = 1). V is a VisualGeometryToolkitCore `LogVignetting` (V = exp(s), `s` a
# `RadialVignetting` or `TensorVignetting` spline of log V, linear in its control points); log L a
# uniform cubic B-spline over an azimuthal-equidistant chart of the covered sphere (`AzimuthalChart`).
# The photo is read at the pixel centres only — no interpolation of the frames at all: the scene
# carries the interpolation, as a spline whose values are fitted, not sampled.
#
# The model is linear in (log L, log V's control points, log g): the estimate is ONE weighted linear
# least squares, with a difference penalty on L's controls along both chart axes, a weak
# second-difference penalty on V's control points, and, when V is fitted, the gauge mean log V = 0
# over the domain's pixels (the overall scale L and V share). V's control points move along the
# caller's free directions Z (`vec(P) = vec(P₀) + Z y`: a radial profile's pole-regular subspace, all
# of a tensor surface's, none for a V held fixed — then any vignetting enters as the fixed numbers
# log V(u): gain compensation under a given V). The normal equations are accumulated on the frames'
# backend by atomic scatter-add (KernelAbstractions; GPU arrays or the host, one code path) and solved
# by Jacobi-preconditioned conjugate gradients on that backend (`ConjugateGradient`) or by a sparse
# Cholesky factorization on the host (`SparseCholesky`).
#
# The weights (`IntensityWeights`, `SampsonWeights`) are fixed per solve. `IntensityWeights` is I²,
# the inverse variance of log I for read-noise-limited data: one solve. `SampsonWeights` treats a row
# as the implicit constraint g(θ; x) = log I − log L(R_k·f(u)) − log V(u) − γ_k on the measurement
# x = (u, log I) — the pixel's sensor point is uncertain too (the registration's residual scale), so a
# row where the model's log I varies fast across the sensor (an edge of the scene) is worth less — and
# weights it by Core's frozen whitening of that constraint, 1/(G_x·Σ_x·G_xᵀ) (`whitening_factor`),
# its measurement gradient G_x = [−∇_u(log L∘R_k∘f + log V), 1] the MODEL's, evaluated at the
# previous solve (the first solve is `IntensityWeights`'). Each solve is the exact linear one.
#
# Why `fit_vignetting` (naming.md): correspondences in, no guess, no hypotheses → not refine or
# find; reweighted and repeated (Sampson's solves), the CG's tolerance and budget → rule 4.
# =============================================================================

"""
    ConjugateGradient(; tolerance, max_iter)

[`fit_vignetting`](@ref)'s solve by Jacobi-preconditioned conjugate gradients
on the frames' backend, to the relative residual `tolerance` within `max_iter`
iterations (`retcode = :max_iterations` when the budget ends first).
"""
struct ConjugateGradient
    tolerance::Float64
    max_iter::Int
end
ConjugateGradient(; tolerance::Real, max_iter::Integer) = ConjugateGradient(Float64(tolerance), Int(max_iter))

"""
    SparseCholesky()

[`fit_vignetting`](@ref)'s solve by a sparse Cholesky factorization (CHOLMOD)
of the same system, assembled on the host (`retcode = :solved`).
"""
struct SparseCholesky end

"""
    IntensityWeights()

[`fit_vignetting`](@ref)'s row weights `I²`: the inverse variance of log I to
first order when the frames' noise does not depend on the value (read noise),
normalized by their mean over the rows. One linear solve. The weights of every
fit before PanoramaStitching 0.3.
"""
struct IntensityWeights end

"""
    SampsonWeights(; point_covariance, log_intensity_scale = 0.01, solves = 1)
    SampsonWeights(; point_scale, log_intensity_scale = 0.01, solves = 1)

[`fit_vignetting`](@ref)'s Sampson row weights: a row is the implicit constraint

    g(θ; x) = log I − log L(R_k·f(u)) − log V(u) − γ_k = 0

on the measurement `x = (u, log I)`, the pixel's sensor point `u` (camera
pixels) and its log value, of covariance `Σ_x = blockdiag(Σ_u, σ²)`, and its
weight is the frozen whitening of that constraint,
`1/(G_x·Σ_x·G_xᵀ)` (VisualGeometryToolkitCore's `whitening_factor` of the
constraint's `constraint_with_jacobians`), with the measurement gradient
`G_x = [−∇_u(log L∘R_k∘f + log V)ᵀ, 1]` of the MODEL (its scene spline and V)
at the previous solve; the first solve is [`IntensityWeights`](@ref)'.

- `point_covariance`: `Σ_u`, the 2 × 2 covariance of a pixel's sensor point
  in camera pixels² — the registration's: a rotation bundle's robust core
  scale is its `loss_width / scale_multiple` ([`RotationBundleMetadata`](@ref)),
  `point_scale` that scale itself (`Σ_u = point_scale²·I`).
- `log_intensity_scale`: `σ`, the standard deviation of log I (a constant
  floor: 0.01 is 1 % of the value).
- `solves`: the Sampson solves after the `IntensityWeights` one, 1 or 2 (the
  second re-evaluates G_x at the first's solution).
"""
struct SampsonWeights
    point_covariance::SMatrix{2, 2, Float64, 4}
    log_intensity_scale::Float64
    solves::Int
end
function SampsonWeights(; point_covariance = nothing, point_scale = nothing, log_intensity_scale::Real = 0.01, solves::Integer = 1)
    (point_covariance === nothing) == (point_scale === nothing) &&
        throw(ArgumentError("SampsonWeights: give the point's noise as exactly one of point_covariance and point_scale"))
    Σ = point_covariance === nothing ? SMatrix{2, 2, Float64, 4}(Float64(point_scale)^2, 0.0, 0.0, Float64(point_scale)^2) :
        SMatrix{2, 2, Float64, 4}(point_covariance)
    (Σ[1, 2] == Σ[2, 1] && LinearAlgebra.isposdef(Matrix(Σ))) ||
        throw(ArgumentError("SampsonWeights: the point covariance must be symmetric positive definite, got $Σ"))
    (log_intensity_scale > 0 && isfinite(log_intensity_scale)) ||
        throw(ArgumentError("SampsonWeights: log_intensity_scale must be positive, got $log_intensity_scale"))
    solves in (1, 2) || throw(ArgumentError("SampsonWeights: solves is 1 or 2, got $solves"))
    return SampsonWeights(Σ, Float64(log_intensity_scale), Int(solves))
end
# The measurement covariance Σ_x over (u₁, u₂, log I).
function _measurement_covariance(w::SampsonWeights)
    Σ = w.point_covariance; s2 = w.log_intensity_scale^2
    return SMatrix{3, 3, Float64, 9}(Σ[1, 1], Σ[2, 1], 0.0, Σ[1, 2], Σ[2, 2], 0.0, 0.0, 0.0, s2)
end

# The first control (0-based) and the weights at x (in control spacings) of an n-control uniform cubic spline: x clamped to
# its knot range [0, n − 3), the weights Core's at the span's local coordinate t − i ∈ [0, 1) (in x's type: ForwardDiff's
# `floor` of a dual is its value's, so a dual x gives the span of its value and dual weights).
@inline function _spline_span(x, n)
    t = clamp(x, 0.0, (n - 3) * (1 - 1.0e-12))
    i = unsafe_trunc(Int, floor(t))
    return i, VgtkCore.cubic_bspline_weights(t - i)
end

# The second differences of a vignetting's control points along each axis of its lattice (Core's `difference_stencil`): DᵀD,
# the weak penalty that fixes the control points the domain does not reach.
function _vignetting_penalty(v::VgtkCore.AbstractVignetting)
    sz = size(VgtkCore.control_points(v)); lin = LinearIndices(sz); s = VgtkCore.difference_stencil(2)
    rows = Vector{Vector{Float64}}()
    for axis in 1:2, I in CartesianIndices(sz)
        I[axis] + 2 <= sz[axis] || continue
        r = zeros(prod(sz))
        for j in 0:2
            r[lin[I + CartesianIndex(ntuple(a -> a == axis ? j : 0, 2))]] = s[j + 1]
        end
        push!(rows, r)
    end
    isempty(rows) && return zeros(prod(sz), prod(sz))
    D = permutedims(reduce(hcat, rows))
    return D' * D
end

"""
    AzimuthalChart

The chart of the scene's log L: the azimuthal-equidistant map of the sphere
about `e3` (`(a, b) = θ·(cos φ, sin φ)` degrees, θ the angle from `e3`, φ the
azimuth from `e1` towards `e2`), singular only at `-e3`, and a uniform cubic
tensor B-spline of `n × n` controls `spacing_deg` apart over
`[−half_extent_deg, half_extent_deg]²`. [`fit_vignetting`](@ref) centres it so
that its singular point is as far from the frames' coverage as the sphere
allows (for a cap-like coverage, close to the mean optical axis; for a full
360° sweep, the pole the frames do not see).
"""
struct AzimuthalChart
    e1::SVector{3, Float64}
    e2::SVector{3, Float64}
    e3::SVector{3, Float64}
    half_extent_deg::Float64
    spacing_deg::Float64
    n::Int
end
function AzimuthalChart(center::StaticVector{3}, half_extent_deg::Real, spacing_deg::Real)
    e3 = normalize(SVector{3, Float64}(center))
    x = abs(e3[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
    e1 = normalize(x - dot(x, e3) * e3); e2 = cross(e3, e1)
    n = ceil(Int, 2 * half_extent_deg / spacing_deg) + 4
    return AzimuthalChart(e1, e2, e3, Float64(half_extent_deg), Float64(spacing_deg), n)
end

# The chart point of the direction d (degrees), and its spline span: the first controls (0-based) and the weights per axis.
@inline function _chart_point(ch::AzimuthalChart, d)
    x = dot(d, ch.e1); y = dot(d, ch.e2); z = dot(d, ch.e3)
    s = sqrt(x * x + y * y)
    θ = rad2deg(atan(s, z))
    return s > 0 ? SVector(θ * x / s, θ * y / s) : SVector(zero(θ), zero(θ))    # θ's type: a dual stays a dual
end
@inline function _chart_span(ch::AzimuthalChart, d)
    p = _chart_point(ch, d)
    ia, wa = _spline_span((p[1] + ch.half_extent_deg) / ch.spacing_deg, ch.n)
    ib, wb = _spline_span((p[2] + ch.half_extent_deg) / ch.spacing_deg, ch.n)
    return ia, ib, wa, wb
end
@inline _chart_index(n, a, b) = (b - 1) * n + a

"""
    VignettingMetadata

The metadata of [`fit_vignetting`](@ref)'s `FitResult`: `chart` (the scene's
[`AzimuthalChart`](@ref)), `scene` (log L's `n²` controls, host, the chart's
column-major order; `NaN` where no observation reached a control), `gains`
(the frames' exposure gains `g_k` of `I_k = g_k·L·V`, one per frame, `g₁ = 1`;
all 1 without `gain_correction`), the residual spread of log I over the rows
(`residual_rms` unweighted, `residual_mad` Core's `mad_scale`, the
1.4826·MAD, `residual_quantiles` the 1 %, 50 % and 99 % quantiles; the
result's `rms` is the one weighted by the last solve's weights),
`relative_residual` (the last solve's ‖r‖/‖b‖), `solver`, `weights` (the
[`IntensityWeights`](@ref) or [`SampsonWeights`](@ref) of the fit: the
Sampson point covariance and log scale it used), `outer_iterations` (the
linear solves: 1 for `IntensityWeights`, 1 + `solves` for `SampsonWeights`),
`log_change` (the largest change of the model's log I over the domain's rows
between the last two solves; 0 for one solve), `n_pixels` (the domain's
pixels), `n_active` (scene controls with data) and `seconds` (the stages' wall
times, summed over the solves).
"""
struct VignettingMetadata{T <: NamedTuple, W <: Union{IntensityWeights, SampsonWeights}}
    chart::AzimuthalChart
    scene::Vector{Float64}
    gains::Vector{Float64}
    residual_rms::Float64
    residual_mad::Float64
    residual_quantiles::NTuple{3, Float64}
    relative_residual::Float64
    solver::Symbol
    weights::W
    outer_iterations::Int
    log_change::Float64
    n_pixels::Int
    n_active::Int
    seconds::T
end

# ----------------------------------------------------------------------------- device helpers

_zeros(be, ::Type{T}, dims...) where {T} = fill!(KA.allocate(be, T, dims...), zero(T))
_upload(be, x::AbstractArray) = copyto!(KA.allocate(be, eltype(x), size(x)), Array{eltype(x)}(x))   # a BitArray as Bools
_elapsed(be, t0) = (KA.synchronize(be); (time_ns() - t0) / 1.0e9)

# The sensor point of a frame pixel (row, col) for a plane whose sample sits at `offset` (frame pixels): the frame's COLMAP point
# (col − ½, row − ½) + offset, scaled by `s` camera pixels per frame pixel.
struct _PlanePoint{T}
    s::T
    offset::SVector{2, T}
end
@inline (p::_PlanePoint{T})(I) where {T} = SVector{2, T}(p.s * (I[2] - T(0.5) + p.offset[1]), p.s * (I[1] - T(0.5) + p.offset[2]))

# A LogVignetting's rows at a domain pixel (per-point functors of Core's `pointwise!`): the active control points of its spline
# of log V (Core's `active_coefficients` of `log_spline`), their rows ∂log V/∂P (the spline's basis), and log V.
struct _ActiveIndices{V}
    v::V
end
@inline (a::_ActiveIndices)(u) = SVector(VgtkCore.active_coefficients(VgtkCore.log_spline(a.v), u)[1])
struct _LogRows{V}
    v::V
end
@inline (a::_LogRows)(u) = SVector(VgtkCore.active_coefficients(VgtkCore.log_spline(a.v), u)[2])
struct _LogValue{V}
    v::V
end
@inline (a::_LogValue)(u) = VgtkCore.log_spline(a.v)(u)
# log V of a V held fixed: any vignetting the render takes, `v(u, f)` (a Core model reads u only)
struct _LogAt{V}
    v::V
end
@inline (a::_LogAt)(u, f) = log(Float64(a.v(u, f)))
# log V at a sensor point u and its ray f, in u's own number type (a ForwardDiff dual for the Sampson gradient): a LogVignetting's
# spline of log V, else the log of any vignetting held fixed.
@inline _log_vignetting(v::VgtkCore.LogVignetting, u, f) = VgtkCore.log_spline(v)(u)
@inline _log_vignetting(v, u, f) = log(v(u, f))

# log L at the world direction d: the chart spline's value at d's chart point (any number type: a dual differentiates it).
@inline function _log_radiance(ch::AzimuthalChart, cL, d)
    ia, ib, wa, wb = _chart_span(ch, d)
    lL = zero(eltype(wa))
    for jb in 0:3, ja in 0:3
        lL += wa[ja + 1] * wb[jb + 1] * cL[_chart_index(ch.n, ia + ja + 1, ib + jb + 1)]
    end
    return lL
end

# ----------------------------------------------------------------------------- the observation as an implicit constraint

# One frame's model of log I as the implicit constraint of a pixel's measurement x = (u, log I) (Core's
# `constraint_with_jacobians`, the scalar-constraint protocol of the epipolar and bearing models: `(g, G_x, G_θ)`):
#
#     g(θ; x) = log I − log L(R·f(u)) − log V(u) − γ,     G_x = [−∇_u(log L∘R∘f + log V)ᵀ, 1]
#
# `scene` log L's chart controls, `vignetting` V (a LogVignetting or one held fixed), `rotation` the frame's R and `log_gain` its
# γ. The pixel's ray `ray = f(u)` and its back-projection Jacobian `ray_jacobian = ∂f/∂u` (Core's `backproject_with_jacobian`,
# tangent to the sphere) are its keywords, as the bearing models' `M₁`, `M₂`: the camera is evaluated once per pixel, not per
# frame. ∇_u is exact (ForwardDiff through the two splines along the tangent directions f + M·δ: the chart point is scale
# invariant, so the first order is the ray's). A gradient that is not finite (the radial profile's centre, the chart's) is
# zero, the value at the point (both splines are smooth there). G_θ is over the row's active coefficients: log L's 16 chart
# controls (the span's column-major order), then V's active control points (none for a V held fixed), then γ.
struct _LogIntensityModel{V, A}
    vignetting::V
    chart::AzimuthalChart
    scene::A
    rotation::SMatrix{3, 3, Float64, 9}
    log_gain::Float64
end
Adapt.@adapt_structure _LogIntensityModel

@inline function VgtkCore.constraint_with_jacobians(m::_LogIntensityModel, x::StaticVector{3}; ray, ray_jacobian)
    u = SVector{2, Float64}(x[1], x[2]); f = SVector{3, Float64}(ray); M = SMatrix{3, 2, Float64, 6}(ray_jacobian)
    model(δ) = _log_radiance(m.chart, m.scene, m.rotation * (f + M * δ)) + _log_vignetting(m.vignetting, u + δ, f + M * δ)
    δ0 = zero(SVector{2, Float64})
    ∇ = ForwardDiff.gradient(model, δ0)
    ∇ = ifelse(isfinite(∇[1]) & isfinite(∇[2]), ∇, δ0)
    g = x[3] - model(δ0) - m.log_gain
    _, _, wa, wb = _chart_span(m.chart, m.rotation * f)
    G_L = SVector{16, Float64}(ntuple(k -> -wa[(k - 1) % 4 + 1] * wb[(k - 1) ÷ 4 + 1], Val(16)))
    G_θ = vcat(G_L, -_active_log_rows(m.vignetting, u), SVector(-1.0))
    return g, SVector(-∇[1], -∇[2], 1.0), G_θ
end
@inline _active_log_rows(v::VgtkCore.LogVignetting, u) = SVector(VgtkCore.active_coefficients(VgtkCore.log_spline(v), u)[2])
@inline _active_log_rows(v, u) = SVector{0, Float64}()

# The row weights the kernels read, `w(I, p)` at the observed value I of the domain pixel p. `_IntensityWeight`: I².
# `_SampsonWeight`: Core's frozen whitening of the observation's constraint, 1/(G_x·Σ_x·G_xᵀ) (`whitening_factor` squared; zero
# where the propagated variance is degenerate), at this frame's model.
struct _IntensityWeight end
@inline (::_IntensityWeight)(I, p) = I * I
struct _SampsonWeight{M, U, F, J}
    model::M
    upix::U
    rays::F
    ray_jacobians::J
    Σ::SMatrix{3, 3, Float64, 9}
end
Adapt.@adapt_structure _SampsonWeight
@inline function (w::_SampsonWeight)(I, p)
    u = w.upix[p]
    _, G_x, _ = VgtkCore.constraint_with_jacobians(w.model, SVector(u[1], u[2], log(I)); ray = w.rays[p], ray_jacobian = w.ray_jacobians[p])
    return VgtkCore.whitening_factor(G_x, w.Σ)^2
end

# A pixel's back-projection Jacobian ∂f/∂u (Core's `backproject_with_jacobian` of the camera's model), per point.
struct _RayJacobian{C}
    camera::C
end
@inline (r::_RayJacobian)(u) = SMatrix{3, 2, Float64, 6}(VgtkCore.backproject_with_jacobian(r.camera, u)[2])

# The minimum over a batch of rays of the cosine to `c` after the rotation R: the chart's extent.
struct _CosTo
    c::SVector{3, Float64}
    R::SMatrix{3, 3, Float64, 9}
end
@inline (k::_CosTo)(f) = dot(k.c, k.R * f)

# The chart's centre: the antipode of the direction farthest from the coverage (sampled by `rays` rotated by every frame).
function _chart_center(rays_host, rotations)
    covered = [SMatrix{3, 3, Float64, 9}(R) * f for R in rotations for f in rays_host]
    best = (-Inf, SVector(0.0, 0.0, 1.0))
    for a in fibonacci_directions(4000)
        m = minimum(d -> 1 - dot(a, d), covered)              # 1 − cos of the nearest covered direction
        m > best[1] && (best = (m, a))
    end
    return -best[2]
end

# ----------------------------------------------------------------------------- the kernels

# One frame's row weights `weight(I, p)` (`_IntensityWeight`, `_SampsonWeight`) at its rows (0 elsewhere): the weights' mean.
@kernel function _weight_kernel!(wout, @Const(frame), @Const(sat), @Const(lin), plane_offset::Int, weight)
    p = @index(Global)
    @inbounds begin
        j = lin[p]
        I = Float64(frame[j + plane_offset])
        wout[p] = (!sat[j] && I > 0) ? weight(I, p) : 0.0
    end
end

# One frame's rows: per domain pixel p (frame value plane[j] at the linear index j = lin[p] + offset), when unsaturated and
# positive, the weight w = weight(I, p)/scale and y = log I − log V₀(u_p); scatter the L–L stencil (7 × 7 per control, both
# triangles), the L–V block (V's rows `VW[p]` at its control points `VI[p]`) and b_L; the per-pixel sums W_p, Y_p and this
# frame's own Wk_p, Yk_p (one thread per pixel per launch: no race). With the frames' gains (`gain_col` > 0, this frame's gain
# column) also the L–gain column BG[:, gain_col].
@kernel function _accumulate_kernel!(S, BV, bL, Wp, Yp, Wk, Yk, BG, @Const(frame), @Const(sat), @Const(lin), @Const(rays),
                                     @Const(VI), @Const(VW), @Const(LV), R, ch::AzimuthalChart, inv_scale::Float64,
                                     plane_offset::Int, gain_col::Int, weight)
    p = @index(Global)
    @inbounds begin
        j = lin[p]
        I = Float64(frame[j + plane_offset])
        Wk[p] = 0.0; Yk[p] = 0.0
        if !sat[j] && I > 0
            ia, ib, wa, wb = _chart_span(ch, R * rays[p])
            w = weight(I, p) * inv_scale
            y = log(I) - LV[p]
            vidx = VI[p]; vw = VW[p]
            Wp[p] += w; Yp[p] += w * y
            Wk[p] = w; Yk[p] = w * y
            n = ch.n
            for jb in 0:3, ja in 0:3
                Bq = wa[ja + 1] * wb[jb + 1]
                q = _chart_index(n, ia + ja + 1, ib + jb + 1)
                Atomix.@atomic bL[q] += w * Bq * y
                gain_col > 0 && (Atomix.@atomic BG[q, gain_col] += w * Bq)
                for m in 1:length(vidx)
                    Atomix.@atomic BV[q, vidx[m]] += w * Bq * vw[m]
                end
                for lb in 0:3, la in 0:3
                    Atomix.@atomic S[q, (lb - jb + 3) * 7 + (la - ja + 3) + 1] += w * Bq * wa[la + 1] * wb[lb + 1]
                end
            end
        end
    end
end

# One frame's gain column of the V block: GV[:, col] = Σ_p Wk_p · b(u_p), Wk this frame's weights.
@kernel function _gain_vblock_kernel!(GV, @Const(Wk), @Const(VI), @Const(VW), col::Int)
    p = @index(Global)
    @inbounds begin
        W = Wk[p]
        if W > 0
            idx = VI[p]; w = VW[p]
            for m in 1:length(idx)
                Atomix.@atomic GV[idx[m], col] += W * w[m]
            end
        end
    end
end

# The V–V block, b_V and the gauge row (the mean row over the domain's pixels) from the per-pixel sums.
@kernel function _vblock_kernel!(AVV, bV, g, @Const(Wp), @Const(Yp), @Const(VI), @Const(VW))
    p = @index(Global)
    @inbounds begin
        idx = VI[p]; w = VW[p]
        for m in 1:length(idx)
            Atomix.@atomic g[idx[m]] += w[m]
        end
        W = Wp[p]
        if W > 0
            Y = Yp[p]
            for m in 1:length(idx)
                Atomix.@atomic bV[idx[m]] += Y * w[m]
                for l in 1:length(idx)
                    Atomix.@atomic AVV[idx[m], idx[l]] += W * w[m] * w[l]
                end
            end
        end
    end
end

# The difference penalty of order 2 or 3 along each chart axis, on runs of order + 1 active controls; an inactive control gets
# a unit diagonal (its coefficient solves to 0 and is reported NaN). `dd` the difference's coefficients, Core's
# `difference_stencil(order)` (padded with a zero to four).
@kernel function _penalty_kernel!(S, @Const(active), λ::Float64, n::Int, order::Int, dd::NTuple{4, Float64})
    q = @index(Global)
    @inbounds begin
        a = (q - 1) % n + 1; b = (q - 1) ÷ n + 1
        for axis in 1:2
            ok = true
            for j in 0:order
                aj = axis == 1 ? a + j : a; bj = axis == 1 ? b : b + j
                (aj > n || bj > n || !active[_chart_index(n, aj, bj)]) && (ok = false)
            end
            if ok
                for j in 0:order, l in 0:order
                    aj = axis == 1 ? a + j : a; bj = axis == 1 ? b : b + j
                    off = axis == 1 ? 3 * 7 + (l - j + 3) + 1 : (l - j + 3) * 7 + 3 + 1
                    Atomix.@atomic S[_chart_index(n, aj, bj), off] += λ * dd[j + 1] * dd[l + 1]
                end
            end
        end
        active[q] || (S[q, 3 * 7 + 3 + 1] = 1.0)
    end
end

# y = S x for the 7 × 7 stencil matrix S.
@kernel function _stencil_matvec_kernel!(y, @Const(S), @Const(x), n::Int)
    q = @index(Global)
    @inbounds begin
        a = (q - 1) % n + 1; b = (q - 1) ÷ n + 1
        acc = 0.0
        for ob in -3:3, oa in -3:3
            aa = a + oa; bb = b + ob
            if 1 <= aa <= n && 1 <= bb <= n
                acc += S[q, (ob + 3) * 7 + (oa + 3) + 1] * x[_chart_index(n, aa, bb)]
            end
        end
        y[q] = acc
    end
end

# y += Bᵀx for a tall nL × m matrix B: each thread reduces a chunk of rows, one atomic per column (a transposed BLAS product
# on this shape was measured 60× slower).
@kernel function _tall_tmatvec_kernel!(y, @Const(B), @Const(x), nL::Int, m::Int, chunk::Int)
    t = @index(Global)
    @inbounds begin
        lo = (t - 1) * chunk + 1; hi = min(t * chunk, nL)
        for j in 1:m
            acc = 0.0
            for q in lo:hi
                acc += B[q, j] * x[q]
            end
            Atomix.@atomic y[j] += acc
        end
    end
end

# One frame's model of log I at every domain pixel, log L + log V + γ_k: its change between two solves.
@kernel function _model_kernel!(LM, @Const(rays), @Const(LV), @Const(cL), R, ch::AzimuthalChart, log_gain::Float64)
    p = @index(Global)
    @inbounds begin
        ia, ib, wa, wb = _chart_span(ch, R * rays[p])
        lL = 0.0
        for jb in 0:3, ja in 0:3
            lL += wa[ja + 1] * wb[jb + 1] * cL[_chart_index(ch.n, ia + ja + 1, ib + jb + 1)]
        end
        LM[p] = lL + LV[p] + log_gain
    end
end

# One frame's residuals of log I (NaN where the row is not an observation; `LV` log V at the domain pixels, `log_gain` the
# frame's γ_k) and their weights `weight(I, p)`.
@kernel function _residual_kernel!(res, wout, @Const(frame), @Const(sat), @Const(lin), @Const(rays), @Const(LV), @Const(cL),
                                   R, ch::AzimuthalChart, plane_offset::Int, log_gain::Float64, weight)
    p = @index(Global)
    @inbounds begin
        j = lin[p]
        I = Float64(frame[j + plane_offset])
        if !sat[j] && I > 0
            ia, ib, wa, wb = _chart_span(ch, R * rays[p])
            lL = 0.0
            for jb in 0:3, ja in 0:3
                lL += wa[ja + 1] * wb[jb + 1] * cL[_chart_index(ch.n, ia + ja + 1, ib + jb + 1)]
            end
            res[p] = Float32(log(I) - lL - LV[p] - log_gain); wout[p] = Float32(weight(I, p))
        else
            res[p] = NaN32; wout[p] = 0.0f0
        end
    end
end

# Jacobi-preconditioned conjugate gradients on the arrays' backend, from `x0` (zero by default).
function _pcg(apply!, b, dinv; x0 = nothing, tolerance, max_iter)
    Ap = similar(b)
    if x0 === nothing
        x = zero(b); r = copy(b)
    else
        x = copy(x0); apply!(Ap, x); r = b .- Ap
    end
    z = dinv .* r; p = copy(z)
    rz = dot(r, z); nb = norm(b); it = 0; rel = 1.0
    while it < max_iter
        it += 1
        apply!(Ap, p)
        α = rz / dot(p, Ap)
        x .+= α .* p; r .-= α .* Ap
        rel = norm(r) / nb
        rel < tolerance && break
        z .= dinv .* r
        rz2 = dot(r, z); p .= z .+ (rz2 / rz) .* p; rz = rz2
    end
    return x, it, rel
end

# The same system assembled sparse on the host and factored (CHOLMOD).
function _cholesky_solve(S, BVz, AVVz, rhs, n)
    nL = size(S, 1); nV = size(BVz, 2)
    Is = Int[]; Js = Int[]; Vs = Float64[]
    sizehint!(Is, 30nL); sizehint!(Js, 30nL); sizehint!(Vs, 30nL)
    for q in 1:nL
        a = (q - 1) % n + 1; b = (q - 1) ÷ n + 1
        for ob in -3:3, oa in -3:3
            v = S[q, (ob + 3) * 7 + (oa + 3) + 1]
            v == 0 && continue
            aa = a + oa; bb = b + ob
            (1 <= aa <= n && 1 <= bb <= n) || continue
            push!(Is, q); push!(Js, _chart_index(n, aa, bb)); push!(Vs, v)
        end
    end
    for j in 1:nV, q in 1:nL
        v = BVz[q, j]; v == 0 && continue
        push!(Is, q); push!(Js, nL + j); push!(Vs, v); push!(Is, nL + j); push!(Js, q); push!(Vs, v)
    end
    for i in 1:nV, j in 1:nV
        push!(Is, nL + i); push!(Js, nL + j); push!(Vs, AVVz[i, j])
    end
    M = SparseArrays.sparse(Is, Js, Vs, nL + nV, nL + nV)
    return LinearAlgebra.cholesky(LinearAlgebra.Symmetric(M)) \ rhs
end

# ----------------------------------------------------------------------------- the fit

"""
    fit_vignetting(vignetting, camera, frames, rotations; free_directions, gain_correction, saturated, circle, raw_scale,
                   rim_margin_px, fov_deg, plane, plane_offsets, chart_spacing_deg, penalty_order, penalty_weight,
                   vignetting_penalty_weight, solver, weights = IntensityWeights())
        -> FitResult{<:VisualGeometryToolkitCore.AbstractVignetting, VignettingMetadata}

The vignetting V and the exposure gains of a camera that only rotated, from
its frames alone (no flat field): every pixel of every frame is the scene's
radiance in its direction times V at its sensor point times the frame's gain,
`I_k(p) = g_k·L(R_k·f(p))·V(u(p))`, and V is the factor the frames share
where their views of the scene overlap (self-calibration). No interpolation
of the frames: each value is read at its own pixel centre, its ray `f(p)` the
`camera`'s back-projection (Core's batched `backproject` of a `CameraModel`,
a `CentralGenericModel` or a `BoundedCamera` of one, computed once for the
pixel grid the frames share), and the scene L is a B-spline over an
[`AzimuthalChart`](@ref) of the covered sphere whose values are fitted with
V and the gains.

- `vignetting`: V's model and start, a VisualGeometryToolkitCore
  `LogVignetting` (V = exp(s), `s` a `RadialVignetting` or `TensorVignetting`
  B-spline of log V on the sensor, camera pixels; its control points the
  start of log V, e.g. zeros); held fixed, any vignetting the render takes (a
  Core model, or a callable `vignetting(u, f)`: a calibration's map, a flat
  field; differentiable in `u` for [`SampsonWeights`](@ref)). A
  sweep about one axis cannot observe a free 2-D V's tilt
  across that axis (a field `exp(α·f_y)` is unchanged by a pure yaw, so L
  absorbs it): use a tensor surface for frames with pitch and roll diversity.
- `free_directions`: the matrix `Z` (`n_control_points × m`) V's control
  points move along, `vec(P) = vec(P₀) + Z y`: Core's
  `pole_regular_basis(v)` for a radial profile (smooth at its centre), the
  identity for a tensor surface, `zeros(n, 0)` to hold the given V fixed —
  then the fit is gain compensation under that V (a calibration's), the same
  system without V's block.
- `gain_correction`: with `true`, every frame has its own exposure gain,
  `log I_k = log g_k + log L(d) + log V(u)`, `K − 1` unknowns `log g_k` with
  the gauge `g₁ = 1` (L carries the first frame's exposure). Use it when the
  exposure or the light can change between frames: a fixed shutter under a
  drifting sky changes every frame's level, and the fit then puts the
  difference into V (on the rooftop sweep the gains reached +18 % and V at the
  domain's edge came out 7 % too dark without them). With `false` every frame
  is taken at one exposure (`g_k = 1`). Required, no default.
- The fit domain: every pixel of the `H × W × 3` `frames` (`raw_scale` camera
  pixels per frame pixel) whose sensor point lies inside `circle` (a Core
  `Circle` or `Ellipse` in camera pixels, `WholePlane()` for a lens without
  one) less `rim_margin_px` and 2 px inside the sensor, with its ray within
  `fov_deg` of the optical axis. A row (a pixel of a frame) counts when its
  `saturated` mask is false and its value positive. `plane` is the plane fitted
  (2: green), read at its own sample position: `plane_offsets` (frame pixels,
  `nothing` for the pixel centre; VisualGeometryToolkitFeatures'
  `load_raw(…).plane_offsets`).
- The model: `log I = log g + log L(d) + log V(u)`, linear in log L's
  controls, log V's control points and the log gains: one weighted linear
  least squares (the weights normalized by their mean over the rows), with
  the difference penalty of order `penalty_order` (2 or 3) and weight
  `penalty_weight` (in units of one row of mean weight) on L's controls along
  both chart axes, `chart_spacing_deg` apart, a weak second-difference penalty
  on V's control points of weight `vignetting_penalty_weight` relative to the
  V block's mean diagonal (it fixes the control points the domain does not
  reach), and, with free directions, the gauge mean log V = 0 over the
  domain's pixels. A V held fixed enters as the numbers log V(u).
- The weights: [`IntensityWeights`](@ref) (the default; `I²`, one solve) or
  [`SampsonWeights`](@ref) (the observation's Sampson whitening, with the
  sensor point's noise: after the `IntensityWeights` solve, `solves` more,
  each with its weights from the previous solve's model).
- The solve: `solver`, a [`ConjugateGradient`](@ref) (Jacobi-preconditioned, on
  the frames' backend) or a [`SparseCholesky`](@ref) (the same system assembled
  sparse and factored on the host).

Every stage runs on the backend that owns `frames` (GPU arrays: KernelAbstractions
kernels and atomic scatter-add; host arrays, the same kernels on the CPU). The
`saturated` masks live on the same backend.

Returns a `FitResult` whose estimate is the fitted Core vignetting model
(isbits: the `vignetting` keyword of [`render_panorama`](@ref),
[`sample_frames`](@ref) and [`score_seams`](@ref), on the host or a device;
the given model when V is held fixed), `retcode` `:converged` or
`:max_iterations` (the last CG) or `:solved` (Cholesky),
`iterations` the CG's summed over the solves (0 for Cholesky),
`rms` the rms of the log residual over `n` rows weighted by the last solve's
weights, and
[`VignettingMetadata`](@ref) (its `gains`, the frames' `g_k`: the `gains`
keyword of [`score_seams`](@ref) and [`render_panorama`](@ref)).
"""
function fit_vignetting(vignetting, camera, frames::AbstractVector, rotations::AbstractVector;
                        free_directions::AbstractMatrix, gain_correction::Bool, saturated::AbstractVector, circle, raw_scale::Integer,
                        rim_margin_px::Real, fov_deg::Real, plane::Integer, plane_offsets, chart_spacing_deg::Real,
                        penalty_order::Integer, penalty_weight::Real, vignetting_penalty_weight::Real,
                        solver::Union{ConjugateGradient, SparseCholesky}, weights::Union{IntensityWeights, SampsonWeights} = IntensityWeights())
    penalty_order in (2, 3) || throw(ArgumentError("fit_vignetting: penalty_order must be 2 or 3, got $penalty_order"))
    K = length(frames)
    length(rotations) == K && length(saturated) == K || throw(DimensionMismatch("fit_vignetting: frames, rotations and saturated disagree"))
    K >= 2 || throw(ArgumentError("fit_vignetting: at least two frames, got $K"))
    Zp = Matrix{Float64}(free_directions); fits_v = size(Zp, 2) > 0
    fits_v && !(vignetting isa VgtkCore.LogVignetting) &&
        throw(ArgumentError("fit_vignetting: free directions move a VisualGeometryToolkitCore LogVignetting's control points (the " *
                            "fit is linear in log V); a $(typeof(vignetting).name.name) can only be held fixed (free_directions with no column)"))
    # a V held fixed is read as log V at the domain's pixels only: its block of the system is one zero column
    nV = fits_v ? VgtkCore.n_control_points(vignetting) : 1
    fits_v && size(Zp, 1) != nV &&
        throw(DimensionMismatch("fit_vignetting: free_directions has $(size(Zp, 1)) rows for $nV control points"))
    Hf, Wf, C = size(frames[1])
    1 <= plane <= C || throw(ArgumentError("fit_vignetting: plane $plane of a $C-plane frame"))
    be = KA.get_backend(frames[1])
    t_all = time_ns(); seconds = Dict{Symbol, Float64}(s => 0.0 for s in _STAGES)   # the stages' wall times
    offset = _plane_offsets(plane_offsets)[plane]
    Rs = [SMatrix{3, 3, Float64, 9}(R) for R in rotations]

    # the domain: the pixels' sensor points and rays, once
    t0 = time_ns()
    U2 = similar(frames[1], SVector{2, Float64}, Hf, Wf)
    VgtkCore.pointwise!(_PlanePoint{Float64}(Float64(raw_scale), SVector{2, Float64}(offset)), U2, CartesianIndices((Hf, Wf)))
    Uall = vec(U2)
    Fall = VgtkCore.backproject(VgtkCore.model(camera), Uall)
    inside = similar(frames[1], Bool, Hf * Wf)
    VgtkCore.pointwise!(_Sees(_frame_region(circle, Wf * raw_scale, Hf * raw_scale, rim_margin_px), cosd(Float64(fov_deg))), inside, Uall, Fall)
    lin = findall(inside)                                                 # the domain: linear indices into one H × W plane
    P = length(lin)
    P > 0 || throw(ArgumentError("fit_vignetting: the fit domain is empty"))
    upix = Uall[lin]; rays = Fall[lin]
    Uall = Fall = U2 = inside = nothing
    sampson = weights isa SampsonWeights
    rayJ = nothing                                                        # the rays' back-projection Jacobians (Sampson's G_x)
    if sampson
        rayJ = similar(upix, SMatrix{3, 2, Float64, 6})
        VgtkCore.pointwise!(_RayJacobian(VgtkCore.model(camera)), rayJ, upix)
    end
    seconds[:backproject] = _elapsed(be, t0)

    # the chart: centred away from the coverage, its extent the coverage's
    t0 = time_ns()
    stride = max(1, P ÷ 2000)
    center = _chart_center(Array(rays[1:stride:end]), Rs)
    mincos = minimum(minimum(_CosTo(center, R), rays) for R in Rs)
    ch = AzimuthalChart(center, acosd(clamp(mincos, -1.0, 1.0)) + chart_spacing_deg, chart_spacing_deg)
    nL = ch.n^2
    seconds[:chart] = _elapsed(be, t0)

    poff = (plane - 1) * Hf * Wf
    nrows = sum(k -> count(.!(saturated[k][lin]) .& (frames[k][lin .+ poff] .> 0)), 1:K)
    nrows > 0 || throw(ArgumentError("fit_vignetting: no unsaturated positive value in the domain"))
    wbuf = _zeros(be, Float64, P)
    # the weight scale of one solve's weights (`weight_of(k)` frame k's): their mean over the rows is 1
    function scale_of(weight_of)
        sumw = 0.0; wk! = _weight_kernel!(be, 256)
        for k in 1:K
            wk!(wbuf, frames[k], saturated[k], lin, poff, weight_of(k); ndrange = P)
            KA.synchronize(be); sumw += sum(wbuf)
        end
        return nrows / sumw
    end

    # the frames' gains: frame k ≥ 2 has the gain column k − 1 (g₁ = 1, the gauge)
    nG = gain_correction ? K - 1 : 0
    D2 = fits_v ? _vignetting_penalty(vignetting) : zeros(1, 1)
    v = vignetting
    MA = fits_v ? length(VgtkCore.active_coefficients(VgtkCore.log_spline(v), Array(upix[1:1])[1])[1]) : 1
    VI = similar(frames[1], SVector{MA, Int}, P); VW = similar(frames[1], SVector{MA, Float64}, P); LV = similar(frames[1], Float64, P)
    if !fits_v                                                            # V fixed: log V once, no V rows
        fill!(VI, SVector{1, Int}(1)); fill!(VW, SVector{1, Float64}(0.0))
        VgtkCore.pointwise!(_LogAt(v), LV, upix, rays)
        all(isfinite, Array(LV)) || throw(ArgumentError("fit_vignetting: the given V is not positive at every domain pixel"))
    end
    xh = Float64[]; cL = Float64[]; active = falses(nL); rel = NaN; retcode = :solved
    iters = 0; change = 0.0
    nsolves = sampson ? 1 + weights.solves : 1
    Σx = sampson ? _measurement_covariance(weights) : nothing
    LMs = sampson ? [_zeros(be, Float64, P) for _ in 1:K] : nothing        # the model of log I per frame: its change between solves
    weight_of = k -> _IntensityWeight()                                   # the first solve's weights: I²
    for it in 1:nsolves
        # log V's rows at the domain pixels (linear in its control points: exact)
        if fits_v
            VgtkCore.pointwise!(_ActiveIndices(v), VI, upix); VgtkCore.pointwise!(_LogRows(v), VW, upix)
            VgtkCore.pointwise!(_LogValue(v), LV, upix)
        end
        inv_scale = scale_of(weight_of)

        # the normal equations
        t0 = time_ns()
        S = _zeros(be, Float64, nL, 49); BG = _zeros(be, Float64, nL, max(nG, 1))
        BV = _zeros(be, Float64, nL, nV); bL = _zeros(be, Float64, nL)
        Wp = _zeros(be, Float64, P); Yp = _zeros(be, Float64, P); Wk = _zeros(be, Float64, P); Yk = _zeros(be, Float64, P)
        GVd = _zeros(be, Float64, nV, max(nG, 1)); Gdiag = zeros(nG); bG = zeros(nG)
        acc! = _accumulate_kernel!(be, 256); gvb! = _gain_vblock_kernel!(be, 256)
        for k in 1:K
            col = gain_correction && k >= 2 ? k - 1 : 0
            acc!(S, BV, bL, Wp, Yp, Wk, Yk, BG, frames[k], saturated[k], lin, rays, VI, VW, LV, Rs[k], ch, inv_scale, poff, col,
                 weight_of(k); ndrange = P)
            if col > 0
                gvb!(GVd, Wk, VI, VW, col; ndrange = P)
                KA.synchronize(be)
                Gdiag[col] = sum(Wk); bG[col] = sum(Yk)
            end
        end
        seconds[:accumulate] += _elapsed(be, t0)
        t0 = time_ns()
        AVVd = _zeros(be, Float64, nV, nV); bVd = _zeros(be, Float64, nV); gd = _zeros(be, Float64, nV)
        _vblock_kernel!(be, 256)(AVVd, bVd, gd, Wp, Yp, VI, VW; ndrange = P)
        AVV = Array(AVVd); bV = Array(bVd); g = Array(gd) ./ P
        # the weak penalty on the control points P₀ + Z y: λ DᵀD on the increment, −λ DᵀD P₀ on the right-hand side
        P0 = fits_v ? vec(Matrix{Float64}(VgtkCore.control_points(v))) : zeros(1)
        λV = vignetting_penalty_weight * Statistics.mean(LinearAlgebra.diag(AVV))
        fits_v && (AVV .+= λV .* D2; bV .-= λV .* (D2 * P0))
        seconds[:vignetting_block] += _elapsed(be, t0)
        t0 = time_ns()
        active = Array(S[:, 3 * 7 + 3 + 1]) .> 0
        dd = penalty_order == 3 ? VgtkCore.difference_stencil(3) : (VgtkCore.difference_stencil(2)..., 0.0)
        _penalty_kernel!(be, 256)(S, _upload(be, active), Float64(penalty_weight), ch.n, Int(penalty_order), dd; ndrange = nL)
        seconds[:penalty] += _elapsed(be, t0)

        # V's increment over the free directions under the gauge: Z = Zp·N, N the null space of the gauge row gᵀZp (the overall
        # scale L and V share). The small block of the system is V's mV coordinates then the frames' nG log gains (none without
        # gain_correction): BVz its L columns, AVVz its own block, bVz its right-hand side.
        Z = fits_v ? Zp * LinearAlgebra.nullspace(reshape(Zp' * g, 1, :)) : zeros(nV, 0)
        mV = size(Z, 2)
        BVz = BV * _upload(be, Z)
        AVVz = Z' * AVV * Z; bVz = Z' * bV
        if nG > 0
            GVz = Z' * Array(GVd)
            BVz = hcat(BVz, BG)
            AVVz = [AVVz GVz; GVz' LinearAlgebra.diagm(Gdiag)]
            bVz = vcat(bVz, bG)
        end
        m = mV + nG
        rhs = vcat(bL, _upload(be, bVz))

        t0 = time_ns()
        if solver isa ConjugateGradient
            AVVzd = _upload(be, AVVz)
            tmp = _zeros(be, Float64, nL); yV = _zeros(be, Float64, m)
            smv! = _stencil_matvec_kernel!(be, 256); ttm! = _tall_tmatvec_kernel!(be, 64)
            function apply!(y, x)
                xL = view(x, 1:nL); xV = view(x, (nL + 1):(nL + m))
                smv!(tmp, S, xL, ch.n; ndrange = nL)
                view(y, 1:nL) .= tmp .+ BVz * xV
                yV .= AVVzd * xV
                m > 0 && ttm!(yV, BVz, xL, nL, m, 16; ndrange = cld(nL, 16))
                view(y, (nL + 1):(nL + m)) .= yV
                return y
            end
            dinv = vcat(1 ./ S[:, 3 * 7 + 3 + 1], _upload(be, 1 ./ LinearAlgebra.diag(AVVz)))
            # warm start after the first solve: the scene and the gains as solved (absolute unknowns), V's increment 0
            x0 = it == 1 ? nothing : _upload(be, vcat(cL, zeros(mV), xh[(end - nG + 1):end]))
            x, it_cg, rel = _pcg(apply!, rhs, dinv; x0, solver.tolerance, solver.max_iter)
            iters += it_cg
            retcode = rel < solver.tolerance ? :converged : :max_iterations
        else
            xh = _cholesky_solve(Array(S), Array(BVz), AVVz, Array(rhs), ch.n)
            x = xh
            rel = norm(_apply_host(Array(S), Array(BVz), AVVz, ch.n, xh) .- Array(rhs)) / norm(Array(rhs))
            retcode = :solved
        end
        seconds[:solve] += _elapsed(be, t0)
        xh = Array(x)
        cL = xh[1:nL]
        if fits_v
            # log V is linear in its control points: the solve's increment is exact, P₀ + δ; then its gauge exact (mean log V = 0
            # over the domain: the scale moves into L)
            vn = VgtkCore.with_control_points(v, P0 .+ Z * xh[(nL + 1):(nL + mV)])
            VgtkCore.pointwise!(_LogValue(vn), LV, upix)
            c = Statistics.mean(Array(LV))
            v = VgtkCore.with_control_points(vn, vec(Matrix{Float64}(VgtkCore.control_points(vn))) .- c)
            cL .+= c; LV .-= c
        end
        sampson || break
        # the next solve's weights: Sampson's at this solve's model of every frame (and its log I, for the change between solves)
        lg = vcat(0.0, xh[(end - nG + 1):end], zeros(K - 1 - nG)); cLm = _upload(be, cL); mk! = _model_kernel!(be, 256)
        change = 0.0
        for k in 1:K
            old = Array(LMs[k])
            mk!(LMs[k], rays, LV, cLm, Rs[k], ch, lg[k]; ndrange = P)
            KA.synchronize(be)
            it > 1 && (change = max(change, maximum(abs, Array(LMs[k]) .- old)))
        end
        @debug "fit_vignetting: solve $it, $retcode; the model's log I moved by $change at most"
        it == nsolves && break
        models = [_LogIntensityModel(v, ch, cLm, Rs[k], lg[k]) for k in 1:K]
        weight_of = k -> _SampsonWeight(models[k], upix, rays, rayJ, Σx)
    end
    log_gains = vcat(0.0, xh[(end - nG + 1):end], zeros(K - 1 - nG))
    cLd = _upload(be, cL)

    # the residuals
    t0 = time_ns()
    res = _zeros(be, Float32, P); ww = _zeros(be, Float32, P); R_all = Float32[]; swr = 0.0; sw = 0.0
    rk! = _residual_kernel!(be, 256)
    for k in 1:K
        rk!(res, ww, frames[k], saturated[k], lin, rays, LV, cLd, Rs[k], ch, poff, log_gains[k], weight_of(k); ndrange = P)
        KA.synchronize(be)
        r = Array(res); w = Array(ww); ok = .!isnan.(r)
        append!(R_all, r[ok]); swr += sum(Float64.(w[ok]) .* Float64.(r[ok]) .^ 2); sw += sum(Float64, w[ok])
    end
    q = Statistics.quantile(R_all, (0.01, 0.5, 0.99))
    seconds[:residuals] = _elapsed(be, t0)
    seconds[:total] = (time_ns() - t_all) / 1.0e9
    scene = copy(cL); scene[.!active] .= NaN
    meta = VignettingMetadata(ch, scene, exp.(log_gains), sqrt(Statistics.mean(abs2, Float64.(R_all))), VgtkCore.mad_scale(R_all),
                              (Float64(q[1]), Float64(q[2]), Float64(q[3])), rel, solver isa ConjugateGradient ? :cg : :cholesky,
                              weights, nsolves, change, P, count(active), NamedTuple{_STAGES}(map(s -> seconds[s], _STAGES)))
    return VgtkCore.FitResult(v, retcode, iters, sqrt(swr / sw), length(R_all), meta)
end

const _STAGES = (:backproject, :chart, :accumulate, :vignetting_block, :penalty, :solve, :residuals, :total)

# The full system's product on the host (the Cholesky solve's relative residual).
function _apply_host(S, BVz, AVVz, n, x)
    nL = size(S, 1); m = size(BVz, 2)
    xL = x[1:nL]; xV = x[(nL + 1):end]
    y = similar(x)
    _stencil_matvec_kernel!(KA.CPU(), 256)(view(y, 1:nL), S, xL, n; ndrange = nL)
    y[1:nL] .+= BVz * xV
    y[(nL + 1):end] .= BVz' * xL .+ AVVz * xV
    return y
end
