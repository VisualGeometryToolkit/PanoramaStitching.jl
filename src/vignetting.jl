# =============================================================================
# fit_vignetting — the vignetting of a rotating camera from its own frames, interpolation free
# =============================================================================
#
# The model of every pixel p of every frame k inside the fit domain:
#
#     I_k(p) = L(R_k·f(p)) · V(u(p))        log I_k(p) = log L(d) + log V(u)
#
# `f(p)` the pixel's ray (the camera's back-projection of its sensor point u(p), computed once:
# the pixel geometry is the same in every frame), `R_k` the frame's rotation (`d = R_k·f`), `L` the
# scene's radiance on the sphere and `V` the vignetting on the sensor. Both logs are uniform cubic
# B-splines: log V on the sensor (a `RadialVignetting` in the distance from a centre, or a
# `TensorVignetting` over a box), log L over an azimuthal-equidistant chart of the covered sphere
# (`AzimuthalChart`). The photo is read at the pixel centres only — no interpolation of the frames
# at all: the scene carries the interpolation, as a spline whose values are fitted, not sampled.
#
# The estimate is a linear least squares in (log L, log V), weighted by I² (the log's first-order
# noise model for a photon-limited value), with a difference penalty on L's controls along both
# chart axes, a weak second-difference penalty on V's controls (the controls whose support misses
# the domain), and one gauge: mean log V = 0 over the domain's pixels (the overall scale L and V
# share; no special point). The normal equations are accumulated on the frames' backend by atomic
# scatter-add (KernelAbstractions; GPU arrays or the host, one code path), the V–V block too, and
# solved by Jacobi-preconditioned conjugate gradients on that backend (`ConjugateGradient`) or by a
# sparse Cholesky factorization on the host (`SparseCholesky`).
#
# Why `fit_vignetting` (naming.md): correspondences in, no guess, no hypotheses → not refine or
# find. The weights are fixed (rule 3's one-step case would allow an algorithm name), but the
# estimator is a penalized fit whose solve is iterative — the CG has a tolerance and a budget and
# can end without converging (`retcode = :max_iterations`), which is a fit's failure mode, not an
# algorithm's (an algorithm can only be degenerate). Rule 4: `fit_*`.
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
    RadialVignetting(center, radius, n_controls)

The radial vignetting space: log V a uniform cubic B-spline with `n_controls`
controls in the distance `ρ = ‖u − center‖` from `center` (camera pixels) over
`[0, radius]` (camera pixels; a point beyond `radius` takes the value at it).
"""
struct RadialVignetting
    center::SVector{2, Float64}
    radius::Float64
    n::Int
    spacing::Float64
end
function RadialVignetting(center, radius::Real, n_controls::Integer)
    n_controls >= 4 || throw(ArgumentError("RadialVignetting: a cubic spline needs at least 4 controls, got $n_controls"))
    radius > 0 || throw(ArgumentError("RadialVignetting: the radius must be positive, got $radius"))
    return RadialVignetting(SVector{2, Float64}(center[1], center[2]), Float64(radius), Int(n_controls), Float64(radius) / (n_controls - 3))
end

"""
    TensorVignetting(origin, extent, n_controls)

The 2-D vignetting space: log V a uniform cubic tensor B-spline over the box
`origin .+ [0, extent]` of the sensor (camera pixels), `n_controls = (nx, ny)`
controls along x and y. No special point: the space represents an off-centre
or asymmetric V as well as a radial one.
"""
struct TensorVignetting
    origin::SVector{2, Float64}
    extent::SVector{2, Float64}
    n::Tuple{Int, Int}
    spacing::SVector{2, Float64}
end
function TensorVignetting(origin, extent, n_controls)
    nx, ny = Int(n_controls[1]), Int(n_controls[2])
    min(nx, ny) >= 4 || throw(ArgumentError("TensorVignetting: a cubic spline needs at least 4 controls per axis, got $n_controls"))
    e = SVector{2, Float64}(extent[1], extent[2])
    all(>(0), e) || throw(ArgumentError("TensorVignetting: the extent must be positive, got $extent"))
    return TensorVignetting(SVector{2, Float64}(origin[1], origin[2]), e, (nx, ny), SVector(e[1] / (nx - 3), e[2] / (ny - 3)))
end

_n_coefficients(s::RadialVignetting) = s.n
_n_coefficients(s::TensorVignetting) = s.n[1] * s.n[2]

# The first control (0-based) and the weights at x (in control spacings) of an n-control uniform cubic spline: x clamped to
# its knot range [0, n − 3), the weights Core's at the span's local coordinate t − i ∈ [0, 1).
@inline function _spline_span(x, n)
    t = clamp(x, 0.0, (n - 3) * (1 - 1.0e-12))
    i = unsafe_trunc(Int, floor(t))
    return i, VgtkCore.cubic_bspline_weights(t - i)
end

# The controls (1-based) and weights of log V at the sensor point u: 4 for the radial space, 16 for the tensor space.
@inline function _vbasis(s::RadialVignetting, u)
    ρ = sqrt((u[1] - s.center[1])^2 + (u[2] - s.center[2])^2)
    i, w = _spline_span(ρ / s.spacing, s.n)
    return (i + 1, i + 2, i + 3, i + 4), w
end
@inline function _vbasis(s::TensorVignetting, u)
    ix, wx = _spline_span((u[1] - s.origin[1]) / s.spacing[1], s.n[1])
    iy, wy = _spline_span((u[2] - s.origin[2]) / s.spacing[2], s.n[2])
    nx = s.n[1]
    idx = ntuple(k -> (iy + (k - 1) ÷ 4) * nx + ix + (k - 1) % 4 + 1, Val(16))
    w = ntuple(k -> wx[(k - 1) % 4 + 1] * wy[(k - 1) ÷ 4 + 1], Val(16))
    return idx, w
end

# The second differences of V's controls (along both axes of the tensor space): DᵀD, the weak penalty that fixes the controls
# the domain does not reach.
function _vignetting_penalty(s::RadialVignetting)
    D = zeros(s.n - 2, s.n)
    for j in 1:(s.n - 2)
        D[j, j] = 1; D[j, j + 1] = -2; D[j, j + 2] = 1
    end
    return D' * D
end
function _vignetting_penalty(s::TensorVignetting)
    nx, ny = s.n
    lin(a, b) = (b - 1) * nx + a
    rows = Vector{Vector{Float64}}()
    for b in 1:ny, a in 1:(nx - 2)
        r = zeros(nx * ny); r[lin(a, b)] = 1; r[lin(a + 1, b)] = -2; r[lin(a + 2, b)] = 1; push!(rows, r)
    end
    for b in 1:(ny - 2), a in 1:nx
        r = zeros(nx * ny); r[lin(a, b)] = 1; r[lin(a, b + 1)] = -2; r[lin(a, b + 2)] = 1; push!(rows, r)
    end
    D = permutedims(reduce(hcat, rows))
    return D' * D
end

"""
    SplineVignetting{S, N}

A fitted vignetting: `space` (a [`RadialVignetting`](@ref) or
[`TensorVignetting`](@ref)) and the `N` `coefficients` of log V in it, in the
gauge mean log V = 0 over the fit's domain. Callable as the render's and the
sampling's vignetting, `v(u, f) = V(u)` (the sensor point `u` in camera
pixels; the ray `f` is not read) — isbits, so it runs in a device kernel.
[`log_vignetting`](@ref)`(v, u)` is log V.
"""
struct SplineVignetting{S, N}
    space::S
    coefficients::SVector{N, Float64}
end

"""
    log_vignetting(v::SplineVignetting, u) -> Float64

log V at the sensor point `u` (camera pixels).
"""
@inline function log_vignetting(v::SplineVignetting, u)
    idx, w = _vbasis(v.space, u)
    s = 0.0
    for m in eachindex(idx)
        s += w[m] * v.coefficients[idx[m]]
    end
    return s
end
@inline (v::SplineVignetting)(u, f) = exp(log_vignetting(v, u))

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
    return s > 0 ? SVector(θ * x / s, θ * y / s) : SVector(0.0, 0.0)
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
column-major order; `NaN` where no observation reached a control), the residual
spread of log I over the rows (`residual_rms` unweighted, `residual_mad`
Core's `mad_scale`, the 1.4826·MAD, `residual_quantiles` the 1 %, 50 % and 99 % quantiles; the result's
`rms` is the I²-weighted one), `relative_residual` (the solve's ‖r‖/‖b‖),
`solver`, `n_pixels` (the domain's pixels), `n_active` (scene controls with
data), `gains` (the frames' exposure gains `g_k` of `I_k = g_k·L·V`, one per
frame, `g₁ = 1`; all 1 when the fit was asked for none, `frame_gains = false`)
and `seconds` (the stages' wall times).
"""
struct VignettingMetadata{T <: NamedTuple}
    chart::AzimuthalChart
    scene::Vector{Float64}
    gains::Vector{Float64}
    residual_rms::Float64
    residual_mad::Float64
    residual_quantiles::NTuple{3, Float64}
    relative_residual::Float64
    solver::Symbol
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

# A batch of sensor points back-projected to unit rays (NaN where the camera has none): a grid model's batched kernel, a
# parametric camera's `Backprojection` through `pointwise!`.
_rays(camera::VgtkCore.CentralGenericModel, U) = VgtkCore.backproject(camera, U)
_rays(camera::VgtkCore.BoundedCamera, U) = _rays(VgtkCore.model(camera), U)
function _rays(camera::VgtkCore.CameraModel, U)
    F = similar(U, SVector{3, Float64})
    VgtkCore.pointwise!(_RayOrNaN(VgtkCore.Backprojection(camera)), F, U)
    return F
end
_rays(camera, U) = throw(ArgumentError("fit_vignetting: the camera must back-project (a CameraModel or a CentralGenericModel), got $(typeof(camera).name.name)"))
struct _RayOrNaN{B}
    b::B
end
@inline function (r::_RayOrNaN)(u)
    f = r.b(u)
    return f === nothing ? SVector(NaN, NaN, NaN) : SVector{3, Float64}(f)
end

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

# One frame's rows: per domain pixel p (frame value plane[j] at the linear index j = lin[p] + offset), when unsaturated and
# positive, the weight w = I²/scale and y = log I; scatter the L–L stencil (7 × 7 per control, both triangles), the L–V block
# and b_L; the per-pixel sums W_p, Y_p and this frame's own Wk_p, Yk_p (one thread per pixel per launch: no race). With the
# frames' gains (`gain_col` > 0, this frame's gain column) also the L–gain column BG[:, gain_col].
@kernel function _accumulate_kernel!(S, BV, bL, Wp, Yp, Wk, Yk, BG, @Const(frame), @Const(sat), @Const(lin), @Const(rays),
                                     @Const(upix), space, R, ch::AzimuthalChart, inv_scale::Float64, plane_offset::Int, gain_col::Int)
    p = @index(Global)
    @inbounds begin
        j = lin[p]
        I = Float64(frame[j + plane_offset])
        Wk[p] = 0.0; Yk[p] = 0.0
        if !sat[j] && I > 0
            ia, ib, wa, wb = _chart_span(ch, R * rays[p])
            w = I * I * inv_scale
            y = log(I)
            vidx, vw = _vbasis(space, upix[p])
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
@kernel function _gain_vblock_kernel!(GV, @Const(Wk), @Const(upix), space, col::Int)
    p = @index(Global)
    @inbounds begin
        W = Wk[p]
        if W > 0
            idx, w = _vbasis(space, upix[p])
            for m in 1:length(idx)
                Atomix.@atomic GV[idx[m], col] += W * w[m]
            end
        end
    end
end

# The V–V block, b_V and the gauge row (the mean basis over the domain's pixels) from the per-pixel sums.
@kernel function _vblock_kernel!(AVV, bV, g, @Const(Wp), @Const(Yp), @Const(upix), space)
    p = @index(Global)
    @inbounds begin
        idx, w = _vbasis(space, upix[p])
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
# a unit diagonal (its coefficient solves to 0 and is reported NaN).
@kernel function _penalty_kernel!(S, @Const(active), λ::Float64, n::Int, order::Int)
    q = @index(Global)
    @inbounds begin
        a = (q - 1) % n + 1; b = (q - 1) ÷ n + 1
        dd = order == 3 ? (-1.0, 3.0, -3.0, 1.0) : (1.0, -2.0, 1.0, 0.0)
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

# One frame's residuals of log I (NaN where the row is not an observation; `log_gain` the frame's γ_k) and their weights.
@kernel function _residual_kernel!(res, wout, @Const(frame), @Const(sat), @Const(lin), @Const(rays), @Const(upix), @Const(cL),
                                   vignetting, R, ch::AzimuthalChart, plane_offset::Int, log_gain::Float64)
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
            res[p] = Float32(log(I) - lL - log_vignetting(vignetting, upix[p]) - log_gain); wout[p] = Float32(I * I)
        else
            res[p] = NaN32; wout[p] = 0.0f0
        end
    end
end

# Jacobi-preconditioned conjugate gradients on the arrays' backend.
function _pcg(apply!, b, dinv; tolerance, max_iter)
    x = zero(b); r = copy(b); z = dinv .* r; p = copy(z); Ap = similar(b)
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
    fit_vignetting(space, camera, frames, rotations; saturated, circle, raw_scale, rim_margin_px, fov_deg, plane,
                   plane_offsets, chart_spacing_deg, penalty_order, penalty_weight, vignetting_penalty_weight, frame_gains, solver)
        -> FitResult{SplineVignetting, VignettingMetadata}

The vignetting V of a camera that only rotated, from its frames alone (no flat
field): every pixel of every frame is the scene's radiance in its direction
times V at its sensor point, `I_k(p) = L(R_k·f(p))·V(u(p))`, and V is the
factor the frames share where their views of the scene overlap. No
interpolation of the frames: each value is read at its own pixel centre, its
ray `f(p)` the `camera`'s back-projection (a `CameraModel` or a
`CentralGenericModel`, computed once for the pixel grid the frames share), and
the scene L is a B-spline over an [`AzimuthalChart`](@ref) of the covered
sphere whose values are fitted with V.

- `space`: log V's spline, a [`RadialVignetting`](@ref) (a cubic B-spline in the
  distance from a centre) or a [`TensorVignetting`](@ref) (a cubic tensor
  B-spline over the sensor, no special point). A sweep about one axis cannot
  observe a free 2-D V's tilt across that axis (a field `exp(α·f_y)` is
  unchanged by a pure yaw, so L absorbs it): use the tensor space for frames
  with pitch and roll diversity.
- The fit domain: every pixel of the `H × W × 3` `frames` (`raw_scale` camera
  pixels per frame pixel) whose sensor point lies inside `circle` (a Core
  `Circle` or `Ellipse` in camera pixels, `WholePlane()` for a lens without
  one) less `rim_margin_px` and 2 px inside the sensor, with its ray within
  `fov_deg` of the optical axis. A row (a pixel of a frame) counts when its
  `saturated` mask is false and its value positive. `plane` is the plane fitted
  (2: green), read at its own sample position: `plane_offsets` (frame pixels,
  `nothing` for the pixel centre; VisualGeometryToolkitFeatures'
  `load_raw(…).plane_offsets`).
- The model: `log I = log L(d) + log V(u)`, least squares weighted by I²
  (normalized by its mean over the rows), the difference penalty of order
  `penalty_order` (2 or 3) and weight `penalty_weight` (in units of one row of
  mean weight) on L's controls along both chart axes, `chart_spacing_deg` apart,
  a weak second-difference penalty on V's controls of weight
  `vignetting_penalty_weight` relative to the V block's mean diagonal (it fixes
  the controls the domain does not reach), and the gauge mean log V = 0 over
  the domain's pixels.
- The frames' exposure: with `frame_gains = true`, every frame has its own gain,
  `log I_k = log g_k + log L(d) + log V(u)`, with K − 1 unknowns `log g_k` and
  the gauge `g₁ = 1` (L carries the first frame's exposure). Use it when the
  exposure or the light can change between frames: a fixed shutter under a
  drifting sky changes every frame's level, and the fit then puts the
  difference into V — on the rooftop sweep the gains reached +18 % and V at the
  domain's edge came out 7 % too dark without them. With `false` every frame
  is taken at one exposure (`g_k = 1`). Required, no default.
- The solve: `solver`, a [`ConjugateGradient`](@ref) (Jacobi-preconditioned, on
  the frames' backend) or a [`SparseCholesky`](@ref) (the same system assembled
  sparse and factored on the host).

Every stage runs on the backend that owns `frames` (GPU arrays: KernelAbstractions
kernels and atomic scatter-add; host arrays, the same kernels on the CPU). The
`saturated` masks live on the same backend.

Returns a `FitResult` whose estimate is the [`SplineVignetting`](@ref) (callable
`v(u, f)`, the vignetting keyword of [`render_panorama`](@ref) and
[`sample_frames`](@ref)), `retcode` `:converged` or `:max_iterations` (CG) or
`:solved` (Cholesky), `iterations` the CG's (0 for Cholesky), `rms` the
I²-weighted rms of the log residual over `n` rows, and [`VignettingMetadata`](@ref)
(its `gains`, the frames' `g_k`: the `gains` keyword of [`score_seams`](@ref) and
[`render_panorama`](@ref)).
"""
function fit_vignetting(space::Union{RadialVignetting, TensorVignetting}, camera, frames::AbstractVector, rotations::AbstractVector;
                        saturated::AbstractVector, circle, raw_scale::Integer, rim_margin_px::Real, fov_deg::Real, plane::Integer,
                        plane_offsets, chart_spacing_deg::Real, penalty_order::Integer, penalty_weight::Real,
                        vignetting_penalty_weight::Real, frame_gains::Bool, solver::Union{ConjugateGradient, SparseCholesky})
    penalty_order in (2, 3) || throw(ArgumentError("fit_vignetting: penalty_order must be 2 or 3, got $penalty_order"))
    K = length(frames)
    length(rotations) == K && length(saturated) == K || throw(DimensionMismatch("fit_vignetting: frames, rotations and saturated disagree"))
    K >= 2 || throw(ArgumentError("fit_vignetting: at least two frames, got $K"))
    Hf, Wf, C = size(frames[1])
    1 <= plane <= C || throw(ArgumentError("fit_vignetting: plane $plane of a $C-plane frame"))
    be = KA.get_backend(frames[1])
    t_all = time_ns(); seconds = Dict{Symbol, Float64}()     # the stages' wall times, read into a NamedTuple at the end
    offset = _plane_offsets(plane_offsets)[plane]
    Rs = [SMatrix{3, 3, Float64, 9}(R) for R in rotations]

    # the domain: the pixels' sensor points and rays, once
    t0 = time_ns()
    U2 = similar(frames[1], SVector{2, Float64}, Hf, Wf)
    VgtkCore.pointwise!(_PlanePoint{Float64}(Float64(raw_scale), SVector{2, Float64}(offset)), U2, CartesianIndices((Hf, Wf)))
    Uall = vec(U2)
    Fall = _rays(camera, Uall)
    inside = similar(frames[1], Bool, Hf * Wf)
    VgtkCore.pointwise!(_Sees(_frame_region(circle, Wf * raw_scale, Hf * raw_scale, rim_margin_px), cosd(Float64(fov_deg))), inside, Uall, Fall)
    lin = findall(inside)                                                 # the domain: linear indices into one H × W plane
    P = length(lin)
    P > 0 || throw(ArgumentError("fit_vignetting: the fit domain is empty"))
    upix = Uall[lin]; rays = Fall[lin]
    Uall = Fall = U2 = inside = nothing
    seconds[:backproject] = _elapsed(be, t0)

    # the chart: centred away from the coverage, its extent the coverage's
    t0 = time_ns()
    stride = max(1, P ÷ 2000)
    center = _chart_center(Array(rays[1:stride:end]), Rs)
    mincos = minimum(minimum(_CosTo(center, R), rays) for R in Rs)
    ch = AzimuthalChart(center, acosd(clamp(mincos, -1.0, 1.0)) + chart_spacing_deg, chart_spacing_deg)
    nL = ch.n^2
    seconds[:chart] = _elapsed(be, t0)

    # the weight scale: the mean I² over the rows
    poff = (plane - 1) * Hf * Wf
    sumw = 0.0; nrows = 0
    for k in 1:K
        I = Float64.(frames[k][lin .+ poff]); ok = .!(saturated[k][lin]) .& (I .> 0)
        sumw += sum(ifelse.(ok, I .^ 2, 0.0)); nrows += count(ok)
    end
    nrows > 0 || throw(ArgumentError("fit_vignetting: no unsaturated positive value in the domain"))
    inv_scale = nrows / sumw

    # the normal equations
    t0 = time_ns()
    nV = _n_coefficients(space)
    S = _zeros(be, Float64, nL, 49); BV = _zeros(be, Float64, nL, nV); bL = _zeros(be, Float64, nL)
    Wp = _zeros(be, Float64, P); Yp = _zeros(be, Float64, P); Wk = _zeros(be, Float64, P); Yk = _zeros(be, Float64, P)
    # the frames' gains: frame k ≥ 2 has the gain column k − 1 (g₁ = 1, the gauge); its L, V and diagonal blocks and b
    nG = frame_gains ? K - 1 : 0
    BG = _zeros(be, Float64, nL, max(nG, 1)); GVd = _zeros(be, Float64, nV, max(nG, 1)); Gdiag = zeros(nG); bG = zeros(nG)
    acc! = _accumulate_kernel!(be, 256); gvb! = _gain_vblock_kernel!(be, 256)
    for k in 1:K
        col = frame_gains && k >= 2 ? k - 1 : 0
        acc!(S, BV, bL, Wp, Yp, Wk, Yk, BG, frames[k], saturated[k], lin, rays, upix, space, Rs[k], ch, inv_scale, poff, col; ndrange = P)
        if col > 0
            gvb!(GVd, Wk, upix, space, col; ndrange = P)
            KA.synchronize(be)
            Gdiag[col] = sum(Wk); bG[col] = sum(Yk)
        end
    end
    seconds[:accumulate] = _elapsed(be, t0)
    t0 = time_ns()
    AVVd = _zeros(be, Float64, nV, nV); bVd = _zeros(be, Float64, nV); gd = _zeros(be, Float64, nV)
    _vblock_kernel!(be, 256)(AVVd, bVd, gd, Wp, Yp, upix, space; ndrange = P)
    AVV = Array(AVVd); bV = Array(bVd); g = Array(gd) ./ P
    AVV .+= vignetting_penalty_weight * Statistics.mean(LinearAlgebra.diag(AVV)) .* _vignetting_penalty(space)
    seconds[:vignetting_block] = _elapsed(be, t0)
    t0 = time_ns()
    active = Array(S[:, 3 * 7 + 3 + 1]) .> 0
    _penalty_kernel!(be, 256)(S, _upload(be, active), Float64(penalty_weight), ch.n, Int(penalty_order); ndrange = nL)
    seconds[:penalty] = _elapsed(be, t0)

    # the gauge mean log V = 0: V = Z·z, Z the null space of gᵀ. The small block of the system is V's m coefficients then the
    # frames' nG log gains (none without frame_gains): BVz its L columns, AVVz its own block, bVz its right-hand side.
    Z = LinearAlgebra.nullspace(reshape(g, 1, :)); mV = size(Z, 2)
    Zd = _upload(be, Z)
    BVz = BV * Zd
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
            ttm!(yV, BVz, xL, nL, m, 16; ndrange = cld(nL, 16))
            view(y, (nL + 1):(nL + m)) .= yV
            return y
        end
        dinv = vcat(1 ./ S[:, 3 * 7 + 3 + 1], _upload(be, 1 ./ LinearAlgebra.diag(AVVz)))
        x, iters, rel = _pcg(apply!, rhs, dinv; solver.tolerance, solver.max_iter)
        retcode = rel < solver.tolerance ? :converged : :max_iterations
    else
        xh = _cholesky_solve(Array(S), Array(BVz), AVVz, Array(rhs), ch.n)
        x = _upload(be, xh); iters = 0
        rel = norm(_apply_host(Array(S), Array(BVz), AVVz, ch.n, xh) .- Array(rhs)) / norm(Array(rhs))
        retcode = :solved
    end
    seconds[:solve] = _elapsed(be, t0)
    xh = Array(x)
    cL = x[1:nL]
    v = SplineVignetting(space, SVector{nV, Float64}(Z * xh[(nL + 1):(nL + mV)]))
    log_gains = vcat(0.0, frame_gains ? xh[(nL + mV + 1):end] : zeros(K - 1))

    # the residuals
    t0 = time_ns()
    res = _zeros(be, Float32, P); ww = _zeros(be, Float32, P); R_all = Float32[]; swr = 0.0; sw = 0.0
    rk! = _residual_kernel!(be, 256)
    for k in 1:K
        rk!(res, ww, frames[k], saturated[k], lin, rays, upix, cL, v, Rs[k], ch, poff, log_gains[k]; ndrange = P)
        KA.synchronize(be)
        r = Array(res); w = Array(ww); ok = .!isnan.(r)
        append!(R_all, r[ok]); swr += sum(Float64.(w[ok]) .* Float64.(r[ok]) .^ 2); sw += sum(Float64, w[ok])
    end
    q = Statistics.quantile(R_all, (0.01, 0.5, 0.99))
    seconds[:residuals] = _elapsed(be, t0)
    seconds[:total] = (time_ns() - t_all) / 1.0e9
    scene = xh[1:nL]; scene[.!active] .= NaN
    meta = VignettingMetadata(ch, scene, exp.(log_gains), sqrt(Statistics.mean(abs2, Float64.(R_all))), VgtkCore.mad_scale(R_all),
                              (Float64(q[1]), Float64(q[2]), Float64(q[3])), rel, solver isa ConjugateGradient ? :cg : :cholesky,
                              P, count(active),
                              NamedTuple{_STAGES}(map(s -> seconds[s], _STAGES)))
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
