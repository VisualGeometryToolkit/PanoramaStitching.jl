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
# `RadialVignetting` or `TensorProductVignetting` spline of log V, linear in its control points); log L a
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
# A row is the implicit constraint g(θ; x) = log I − log L(R_k·f(u)) − log V(u) − γ_k on the
# measurement x = (u, log I), and its weight is always Core's frozen whitening of that constraint by
# the measurement noise, 1/(G_x·Σ_x·G_xᵀ) (`whitening_factor`), fixed per solve. The option
# `measurement_noise` chooses the noise model (measurement-noise.md: the noise lives on the
# measurements). `IntensityNoise`: noise in the value only, read noise of constant σ, so the
# whitening of log I is I² (up to the constant): one solve. `IntensityAndPositionNoise`: the
# pixel's sensor point is uncertain too (the registration's residual scale), so a row where the
# model's log I varies fast across the sensor (an edge of the scene) is worth less; its measurement
# gradient G_x = [−∇_u(log L∘R_k∘f + log V), 1] is the MODEL's, evaluated at the previous solve (the
# first solve is `IntensityNoise`'s). Each solve is the exact linear one.
#
# Why `fit_vignetting` (naming.md): correspondences in, no guess, no hypotheses → not refine or
# find; reweighted and repeated (the position noise's solves), the CG's tolerance and budget → rule 4.
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
    IntensityNoise()

[`fit_vignetting`](@ref)'s `measurement_noise`: noise in the pixel value only,
read noise of constant σ (the frames' noise does not depend on the value). The
whitening of a row's log I by it is the weight `I²` (to first order, up to the
constant, normalized by the weights' mean over the rows). One linear solve. The
weights of every fit before PanoramaStitching 0.3.
"""
struct IntensityNoise end

"""
    IntensityAndPositionNoise(; point_covariance, log_intensity_scale = 0.01, iterations = 1)

[`fit_vignetting`](@ref)'s `measurement_noise`: noise in the pixel value and
in the pixel's sensor position. A row is the implicit constraint

    g(θ; x) = log I − log L(R_k·f(u)) − log V(u) − γ_k = 0

on the measurement `x = (u, log I)`, the pixel's sensor point `u` (camera
pixels) and its log value, of covariance `Σ_x = blockdiag(Σ_u, σ²)`, and its
weight is the frozen whitening of that constraint by this noise,
`1/(G_x·Σ_x·G_xᵀ)` (VisualGeometryToolkitCore's `whitening_factor` of the
constraint's `constraint_with_jacobians`; in σ units), with the measurement
gradient `G_x = [−∇_u(log L∘R_k∘f + log V)ᵀ, 1]` of the MODEL (its scene
spline and V) at the previous solve; the first solve is
[`IntensityNoise`](@ref)'s.

- `point_covariance`: `Σ_u`, VisualGeometryToolkitCore's `CovMat{2}` of a
  pixel's sensor point in camera pixels² — the registration's: a rotation
  bundle's robust core scale `σ_u` is its `loss_width / scale_multiple`
  ([`RotationBundleMetadata`](@ref)), and `CovMat{2}(σ = σ_u)` its isotropic
  covariance `σ_u²·I`. A full or diagonal `CovMat` must be symmetric positive
  definite.
- `log_intensity_scale`: `σ`, the standard deviation of log I (a constant
  floor: 0.01 is 1 % of the value).
- `iterations`: the reweight-and-solve cycles after the `IntensityNoise`
  solve, 1 or 2 (the second re-evaluates G_x at the first's solution).
"""
struct IntensityAndPositionNoise{C <: VgtkCore.CovMat{2}}
    point_covariance::C
    log_intensity_scale::Float64
    iterations::Int
end
function IntensityAndPositionNoise(; point_covariance::VgtkCore.CovMat{2}, log_intensity_scale::Real = 0.01, iterations::Integer = 1)
    Σ = SMatrix{2, 2, Float64, 4}(point_covariance)
    (Σ[1, 2] == Σ[2, 1] && LinearAlgebra.isposdef(Matrix(Σ))) ||
        throw(ArgumentError("IntensityAndPositionNoise: the point covariance must be symmetric positive definite, got $Σ"))
    (log_intensity_scale > 0 && isfinite(log_intensity_scale)) ||
        throw(ArgumentError("IntensityAndPositionNoise: log_intensity_scale must be positive, got $log_intensity_scale"))
    iterations in (1, 2) || throw(ArgumentError("IntensityAndPositionNoise: iterations is 1 or 2, got $iterations"))
    return IntensityAndPositionNoise(point_covariance, Float64(log_intensity_scale), Int(iterations))
end
"""
    IntensityGains()

[`fit_vignetting`](@ref)'s `gain_correction`: one exposure gain per frame
shared by every colour plane (a gain on the intensity),
`log I_kc = log g_k + log L_c(d) + log V(u)`, `K − 1` unknowns with the gauge
`g₁ = 1`, fitted on the fitted `plane` and taken for every colour plane.
"""
struct IntensityGains end

"""
    ChannelGains()

[`fit_vignetting`](@ref)'s `gain_correction`: one exposure gain per frame and
colour plane, `log I_kc = log g_kc + log L_c(d) + log V(u)`, the gauge
`g_1c = 1` per plane: a frame's exposure and its white balance or colour shift.
V is one, shared by the planes: fitted (or held) on `plane` with that plane's
gains, then each other plane's gains solved under that V (the same system with
V held fixed, its own scene L_c).
"""
struct ChannelGains end

"""
    GAIN_CORRECTIONS

The TOML tags of [`fit_vignetting`](@ref)'s `gain_correction` (a
VisualGeometryToolkitCore `TagRegistry`, family suffix `Gains`): `"none"`
(`nothing`: every frame at one exposure), `"intensity"` ([`IntensityGains`](@ref))
and `"channel"` ([`ChannelGains`](@ref)); the example's `vignetting.gain_correction`.
A tag of an earlier release is refused with an `ArgumentError` that names its
new tag (CHANGELOG, 0.4.0).
"""
const GAIN_CORRECTIONS = VgtkCore.TagRegistry(:gain_correction, "Gains", "none" => nothing, "intensity" => IntensityGains,
                                              "channel" => ChannelGains; retired = ("frame" => "intensity",),
                                              paths = [r"^vignetting\.gain_correction$"])

"""
    MEASUREMENT_NOISES

The TOML tags of [`fit_vignetting`](@ref)'s `measurement_noise` (a
VisualGeometryToolkitCore `TagRegistry`, family suffix `Noise`): `"intensity"`
([`IntensityNoise`](@ref)) and `"intensity_and_position"`
([`IntensityAndPositionNoise`](@ref), whose settings the config states beside
it); the example's `vignetting.measurement_noise`. A tag of an earlier release
is refused with an `ArgumentError` that names its new tag (CHANGELOG, 0.4.0).
"""
const MEASUREMENT_NOISES = VgtkCore.TagRegistry(:measurement_noise, "Noise", "intensity" => IntensityNoise,
                                                "intensity_and_position" => IntensityAndPositionNoise;
                                                retired = ("sampson" => "intensity_and_position",),
                                                paths = [r"^vignetting\.measurement_noise$"])

# The measurement covariance Σ_x over (u₁, u₂, log I).
function _measurement_covariance(w::IntensityAndPositionNoise)
    Σ = w.point_covariance; s2 = w.log_intensity_scale^2
    return SMatrix{3, 3, Float64, 9}(Σ[1, 1], Σ[2, 1], 0.0, Σ[1, 2], Σ[2, 2], 0.0, 0.0, 0.0, s2)
end

"""
    AzimuthalChart(axis, half_extent, spacing)

The chart of the scene's log L: VisualGeometryToolkitCore's azimuthal-equidistant
direction chart about `axis` (`chart`, Core's `equidistant_chart(axis)`: a
direction at angle θ from the axis lands at radius θ, radians, in the
direction of its azimuth; singular only at `-axis`), and the uniform cubic
tensor B-spline of `n × n` controls laid over the closed square `area` of it,
`[−half_extent, −half_extent + (n − 3)·spacing]²` with
`n = ⌈2·half_extent / spacing⌉ + 4`, so the controls are `spacing` (radians)
apart: Core's `spline_window` / `spline_value` lattice, the controls
`[row, column]` along the chart's (second, first) coordinate.
[`fit_vignetting`](@ref) centres it so that its singular point is as far from
the frames' coverage as the sphere allows (for a cap-like coverage, close to
the mean optical axis; for a full 360° sweep, the pole the frames do not see).
"""
struct AzimuthalChart{C}
    chart::C
    area::VgtkCore.Rect{2, Float64}
    n::Int
end
function AzimuthalChart(axis::StaticVector{3}, half_extent::Real, spacing::Real)
    n = ceil(Int, 2 * half_extent / spacing) + 4
    h = Float64(half_extent); w = (n - 3) * Float64(spacing)
    return AzimuthalChart(VgtkCore.equidistant_chart(axis), VgtkCore.Rect(-h, -h, w, w), n)
end

# The spline window of the direction d (Core's `spline_window` at its chart point): the controls `[iy − 3 + r, ix − 3 + c]` and
# the weights `wx[c]`, `wy[r]` (in d's number type: a dual direction gives dual weights).
@inline _chart_window(ch::AzimuthalChart, d) = VgtkCore.spline_window(ch.area, (ch.n, ch.n), ch.chart(d))
# The linear index of the control (row, col) in the scene's column-major n × n controls.
@inline _chart_index(n, row, col) = (col - 1) * n + row

"""
    VignettingMetadata

The metadata of [`fit_vignetting`](@ref)'s `FitResult`: `chart` (the scene's
[`AzimuthalChart`](@ref)), `scene` (log L's `n × n` controls, host, the
chart's layout, so that `VisualGeometryToolkitCore.spline_value(scene, chart.area,
chart.chart(d))` is log L at the direction `d`; `NaN` where no observation
reached a control), `gains`
(`K × 3`: the frames' exposure gains `g_kc` of `I_kc = g_kc·L_c·V`, one per frame
and colour plane, `g_1c = 1`; all 1 without gain correction, and one per frame
— every column the same — with [`IntensityGains`](@ref)), the residual spread of
log I of the fitted `plane` over the rows
(`residual_rms` unweighted, `residual_mad` Core's `mad_scale`, the
1.4826·MAD, `residual_quantiles` the 1 %, 50 % and 99 % quantiles; the
result's `rms` is the one weighted by the last solve's weights),
`relative_residual` (the last solve's ‖r‖/‖b‖), `solver`, `measurement_noise`
(the [`IntensityNoise`](@ref) or [`IntensityAndPositionNoise`](@ref) of the
fit: the point covariance, log scale and iterations it used: the fitted
plane's linear solves are 1, or 1 + `iterations` for
`IntensityAndPositionNoise`),
`log_change` (the largest change of the model's log I over the domain's rows
between the last two solves; 0 for one solve), `n_pixels` (the domain's
pixels), `n_active` (scene controls with data) and `seconds` (the stages' wall
times, summed over the solves).
"""
struct VignettingMetadata{T <: NamedTuple, W <: Union{IntensityNoise, IntensityAndPositionNoise}, C <: AzimuthalChart}
    chart::C
    scene::Matrix{Float64}
    gains::Matrix{Float64}
    residual_rms::Float64
    residual_mad::Float64
    residual_quantiles::NTuple{3, Float64}
    relative_residual::Float64
    solver::Symbol
    measurement_noise::W
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
# (Core's `pixel_centre`) + offset, scaled by `s` camera pixels per frame pixel.
struct _PlanePoint{T}
    s::T
    offset::SVector{2, T}
end
@inline (p::_PlanePoint{T})(I) where {T} = p.s * (VgtkCore.pixel_centre(T, I) + p.offset)

# log V at a sensor point u and its ray f, in u's own number type (a ForwardDiff dual for the measurement gradient): a LogVignetting's
# spline of log V, else the log of any vignetting held fixed (any vignetting the render takes, `v(u, f)`; a Core model reads u
# only). And a LogVignetting's row ∂log V/∂P at its active control points (the spline's basis, Core's `active_coefficients`);
# none for a V held fixed.
@inline _log_vignetting(v::VgtkCore.LogVignetting, u, f) = VgtkCore.log_spline(v)(u)
@inline function _log_vignetting(v, u, f)
    V = v(u, f)
    return log(ifelse(V > 0, V, zero(V)))        # V ≤ 0: −Inf, refused by name afterwards (not Base's DomainError on the host)
end
@inline _active_log_rows(v::VgtkCore.LogVignetting, u) = SVector(VgtkCore.active_coefficients(VgtkCore.log_spline(v), u)[2])
@inline _active_log_rows(v, u) = SVector{0, Float64}()

# The per-point functors of Core's `pointwise!` over the domain pixels (moved to the frames' backend with `OnBackend`, so a V
# holding arrays, a calibration's map, reaches a device): a LogVignetting's active control points and their rows, and log V.
struct _ActiveIndices{V}
    v::V
end
@inline (a::_ActiveIndices)(u) = SVector(VgtkCore.active_coefficients(VgtkCore.log_spline(a.v), u)[1])
struct _LogRows{V}
    v::V
end
@inline (a::_LogRows)(u) = _active_log_rows(a.v, u)
struct _LogV{V}
    v::V
end
@inline (a::_LogV)(u, f) = Float64(_log_vignetting(a.v, u, f))
Adapt.@adapt_structure _ActiveIndices
Adapt.@adapt_structure _LogRows
Adapt.@adapt_structure _LogV

# ----------------------------------------------------------------------------- the observation as an implicit constraint

# One frame's model of log I as the implicit constraint of a pixel's measurement x = (u, log I) (Core's
# `constraint_with_jacobians`, the scalar-constraint protocol of the epipolar and bearing models: `(g, G_x, G_θ)`):
#
#     g(θ; x) = log I − log L(R·f(u)) − log V(u) − γ,     G_x = [−∇_u(log L∘R∘f + log V)ᵀ, 1]
#
# `scene` log L's n × n chart controls (log L at d is Core's `spline_value` at d's chart point), `vignetting` V (a LogVignetting or one held fixed), `rotation` the frame's R and `log_gain` its
# γ. The pixel's ray `ray = f(u)` and its back-projection Jacobian `ray_jacobian = ∂f/∂u` (Core's `backproject_with_jacobian`,
# tangent to the sphere) are its keywords, as the bearing models' `M₁`, `M₂`: the camera is evaluated once per pixel, not per
# frame. ∇_u is exact (ForwardDiff through the two splines along the tangent directions f + M·δ: the chart point is scale
# invariant, so the first order is the ray's). A gradient that is not finite (the radial profile's centre, the chart's) is
# zero, the value at the point (both splines are smooth there). G_θ is over the row's active coefficients: log L's 16 chart
# controls (the window's column-major order, `[iy − 3 + r, ix − 3 + c]` with r fastest), then V's active control points (none
# for a V held fixed), then γ.
struct _LogIntensityModel{V, C, A}
    vignetting::V
    chart::AzimuthalChart{C}
    scene::A
    rotation::SMatrix{3, 3, Float64, 9}
    log_gain::Float64
end
Adapt.@adapt_structure _LogIntensityModel

@inline function VgtkCore.constraint_with_jacobians(m::_LogIntensityModel, x::StaticVector{3}; ray, ray_jacobian)
    u = SVector{2, Float64}(x[1], x[2]); f = SVector{3, Float64}(ray)
    lm, G_x = _model_with_measurement_gradient(m, u, f, SMatrix{3, 2, Float64, 6}(ray_jacobian))
    _, _, wx, wy = _chart_window(m.chart, m.rotation * f)
    G_L = SVector{16, Float64}(ntuple(k -> -wy[(k - 1) % 4 + 1] * wx[(k - 1) ÷ 4 + 1], Val(16)))
    G_θ = vcat(G_L, -_active_log_rows(m.vignetting, u), SVector(-1.0))
    return x[3] - lm - m.log_gain, G_x, G_θ
end
# The model's log L + log V at the pixel (u, its ray f, ∂f/∂u = M) and the constraint's measurement gradient
# G_x = [−∇_u(log L∘R∘f + log V), 1]: the weights' part of `constraint_with_jacobians` (the whitened weight needs no G_θ).
@inline function _model_with_measurement_gradient(m::_LogIntensityModel, u, f, M)
    model(δ) = VgtkCore.spline_value(m.scene, m.chart.area, m.chart.chart(m.rotation * (f + M * δ))) + _log_vignetting(m.vignetting, u + δ, f + M * δ)
    δ0 = zero(SVector{2, Float64})
    ∇ = ForwardDiff.gradient(model, δ0)
    ∇ = ifelse(isfinite(∇[1]) & isfinite(∇[2]), ∇, δ0)
    return model(δ0), SVector(-∇[1], -∇[2], 1.0)
end

# The row weights the kernels read, `w(I, p)` at the observed value I of the domain pixel p: the whitening of the row's
# constraint by the measurement noise. `_IntensityWeight` (`IntensityNoise`): I². `_IntensityAndPositionWeight`
# (`IntensityAndPositionNoise`): Core's frozen whitening of the observation's constraint, 1/(G_x·Σ_x·G_xᵀ) (`whitening_factor`
# squared; zero where the propagated variance is degenerate), at this frame's model.
struct _IntensityWeight end
@inline (::_IntensityWeight)(I, p) = I * I
struct _IntensityAndPositionWeight{M, U, F, J}
    model::M
    upix::U
    rays::F
    ray_jacobians::J
    Σ::SMatrix{3, 3, Float64, 9}
end
Adapt.@adapt_structure _IntensityAndPositionWeight
@inline function (w::_IntensityAndPositionWeight)(I, p)
    _, G_x = _model_with_measurement_gradient(w.model, w.upix[p], w.rays[p], w.ray_jacobians[p])   # G_x does not read I
    return VgtkCore.whitening_factor(G_x, w.Σ)^2
end

# A pixel's back-projection Jacobian ∂f/∂u (Core's `backproject_with_jacobian` of the camera's model), per point.
struct _RayJacobian{C}
    camera::C
end
Adapt.@adapt_structure _RayJacobian          # a grid model's control grid moved to the device (Core's adapt rule)
@inline (r::_RayJacobian)(u) = SMatrix{3, 2, Float64, 6}(VgtkCore.backproject_with_jacobian(r.camera, u)[2])

# The cosine of the rotated ray R·f to `c` (its minimum over the rays and frames is the chart's extent).
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

# One frame's row weights `weight(I, p)` (`_IntensityWeight`, `_IntensityAndPositionWeight`) at its rows (0 elsewhere): the weights' mean.
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
            ix, iy, wx, wy = _chart_window(ch, R * rays[p])
            w = weight(I, p) * inv_scale
            y = log(I) - LV[p]
            vidx = VI[p]; vw = VW[p]
            Wp[p] += w; Yp[p] += w * y
            Wk[p] = w; Yk[p] = w * y
            n = ch.n
            for c in 1:4, r in 1:4                                        # the window's controls [iy − 3 + r, ix − 3 + c]
                Bq = wy[r] * wx[c]
                q = _chart_index(n, iy - 3 + r, ix - 3 + c)
                Atomix.@atomic bL[q] += w * Bq * y
                gain_col > 0 && (Atomix.@atomic BG[q, gain_col] += w * Bq)
                for m in 1:length(vidx)
                    Atomix.@atomic BV[q, vidx[m]] += w * Bq * vw[m]
                end
                for lc in 1:4, lr in 1:4
                    Atomix.@atomic S[q, (lc - c + 3) * 7 + (lr - r + 3) + 1] += w * Bq * wy[lr] * wx[lc]
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

# One frame's model of log I at every domain pixel, log L + log V + γ_k (log L Core's `spline_value` of the n × n scene
# controls `cL` at the ray's chart point): its change between two solves.
@kernel function _model_kernel!(LM, @Const(rays), @Const(LV), @Const(cL), R, ch::AzimuthalChart, log_gain::Float64)
    p = @index(Global)
    @inbounds LM[p] = VgtkCore.spline_value(cL, ch.area, ch.chart(R * rays[p])) + LV[p] + log_gain
end

# One frame's residuals of log I (NaN where the row is not an observation; `LV` log V at the domain pixels, `cL` the n × n scene
# controls, `log_gain` the frame's γ_k) and their weights `weight(I, p)`.
@kernel function _residual_kernel!(res, wout, @Const(frame), @Const(sat), @Const(lin), @Const(rays), @Const(LV), @Const(cL),
                                   R, ch::AzimuthalChart, plane_offset::Int, log_gain::Float64, weight)
    p = @index(Global)
    @inbounds begin
        j = lin[p]
        I = Float64(frame[j + plane_offset])
        if !sat[j] && I > 0
            res[p] = Float32(log(I) - VgtkCore.spline_value(cL, ch.area, ch.chart(R * rays[p])) - LV[p] - log_gain); wout[p] = Float32(weight(I, p))
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
    fit_vignetting(vignetting, camera::BoundedCamera, frames, rotations; free_directions, gain_correction, saturated, raw_scale,
                   rim_margin_px, fov_deg, plane, plane_offsets, chart_spacing_deg, penalty_order, penalty_weight,
                   vignetting_penalty_weight, solver, measurement_noise = IntensityNoise())
        -> FitResult{<:VisualGeometryToolkitCore.AbstractVignetting, VignettingMetadata}

The vignetting V and the exposure gains of a camera that only rotated, from
its frames alone (no flat field): every pixel of every frame is the scene's
radiance in its direction times V at its sensor point times the frame's gain,
`I_k(p) = g_k·L(R_k·f(p))·V(u(p))`, and V is the factor the frames share
where their views of the scene overlap (self-calibration). No interpolation
of the frames: each value is read at its own pixel centre, its ray `f(p)` the
`camera`'s back-projection (Core's batched `backproject` of the model of the
VisualGeometryToolkitCore `BoundedCamera`, computed once for the
pixel grid the frames share), and the scene L is a B-spline over an
[`AzimuthalChart`](@ref) of the covered sphere whose values are fitted with
V and the gains.

- `vignetting`: V's model and start, a VisualGeometryToolkitCore
  `LogVignetting` (V = exp(s), `s` a `RadialVignetting` or
  `TensorProductVignetting` B-spline of log V on the sensor, camera pixels; its control points the
  start of log V, e.g. zeros); held fixed, any vignetting the render takes (a
  Core model, or a callable `vignetting(u, f)`: a calibration's map, a flat
  field; differentiable in `u` for [`IntensityAndPositionNoise`](@ref)). A
  sweep about one axis cannot observe a free 2-D V's tilt
  across that axis (a field `exp(α·f_y)` is unchanged by a pure yaw, so L
  absorbs it): use a tensor surface for frames with pitch and roll diversity.
- `free_directions`: the matrix `Z` (`n_control_points × m`) V's control
  points move along, `vec(P) = vec(P₀) + Z y`: Core's
  `pole_regular_basis(v)` for a radial profile (smooth at its centre), the
  identity for a tensor surface, `zeros(n, 0)` to hold the given V fixed —
  then the fit is gain compensation under that V (a calibration's), the same
  system without V's block.
- `gain_correction`: `nothing` (every frame at one exposure, `g = 1`),
  [`IntensityGains`](@ref)`()` (one gain per frame, `log I_k = log g_k + log L(d)
  + log V(u)`, `K − 1` unknowns with the gauge `g₁ = 1`: L carries the first
  frame's exposure) or [`ChannelGains`](@ref)`()` (one per frame and colour
  plane, the gauge `g_1c = 1`: each other plane's gains solved under the fitted
  V, which stays one for all planes). In both modes of V — fitted, or held
  fixed. Use gains when the exposure or the light can change between frames: a
  fixed shutter under a drifting sky changes every frame's level, and the fit
  then puts the difference into V (on the rooftop sweep the gains reached
  +18 % and V at the domain's edge came out 7 % too dark without them); per
  channel when the white balance or the light's colour can change too.
  Required, no default (TOML: [`GAIN_CORRECTIONS`](@ref)).
- The fit domain: every pixel of the `H × W × 3` `frames` (`raw_scale` camera
  pixels per frame pixel) whose sensor point lies inside the camera's
  `imaged_region` (its sensor ∩ its lens coverage ∩ its model's valid domain)
  less `rim_margin_px` at its lens coverage's edge, and 2 px inside the frame, with its ray within
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
- `measurement_noise`: the noise model of a row's measurement; the row weights
  are always the whitening of the log-I constraint by it (Core's
  `whitening_factor`). [`IntensityNoise`](@ref) (the default: noise in the
  value only, weight `I²`, one solve) or [`IntensityAndPositionNoise`](@ref)
  (noise in the value and the pixel's sensor position: after the
  `IntensityNoise` solve, `iterations` more, each with its weights from the
  previous solve's model). TOML: [`MEASUREMENT_NOISES`](@ref).
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
`:max_iterations` (a CG that ended at its budget: the last solve of any plane
solved, the other planes' of [`ChannelGains`](@ref) included) or `:solved`
(Cholesky), `iterations` the CG's summed over every solve of every plane (0 for
Cholesky),
`rms` the rms of the log residual over `n` rows weighted by the last solve's
weights, and
[`VignettingMetadata`](@ref) (its `gains`, the frames' `K × 3` `g_kc`: the
`gains` keyword of [`score_seams`](@ref) and [`render_panorama`](@ref); the
residuals and the solves are the fitted `plane`'s).
"""
function fit_vignetting(vignetting, camera::VgtkCore.BoundedCamera, frames::AbstractVector, rotations::AbstractVector;
                        gain_correction::Union{Nothing, IntensityGains, ChannelGains}, plane::Integer, kw...)
    r = _fit_plane(vignetting, camera, frames, rotations; gains = gain_correction !== nothing, plane, kw...)::VgtkCore.FitResult
    if gain_correction isa ChannelGains
        # each other plane's gains under the fitted V (held fixed: its log V(u) the fixed numbers), its own scene, into its
        # column of the K × C gains (the fit's own plane's in every column until then)
        held = merge(NamedTuple(kw), (; free_directions = zeros(0, 0)))
        retcode = VgtkCore.retcode(r); iters = r.iterations
        for c in axes(r.metadata.gains, 2)
            c == plane && continue
            g, rc, it = _fit_plane(VgtkCore.estimate(r), camera, frames, rotations; gains = true, plane = c, gains_only = true,
                                   held...)::Tuple{Vector{Float64}, Symbol, Int}
            r.metadata.gains[:, c] .= g; iters += it
            rc === :max_iterations && (retcode = rc)                      # an unconverged plane's gains are not hidden
        end
        return VgtkCore.FitResult(VgtkCore.estimate(r), retcode, iters, r.rms, r.n, r.metadata)
    end
    return r
end

# One plane's fit (`fit_vignetting`'s system), its gains in every column of the K × C metadata gains (`gains`: K − 1 free,
# g₁ = 1; else all 1). `gains_only`: the K gains, the last solve's retcode and the CG iterations alone, without the residuals'
# statistics (ChannelGains' other planes).
function _fit_plane(vignetting, camera, frames::AbstractVector, rotations::AbstractVector;
                    free_directions::AbstractMatrix, gains::Bool, saturated::AbstractVector, raw_scale::Integer,
                    rim_margin_px::Real, fov_deg::Real, plane::Integer, plane_offsets, chart_spacing_deg::Real,
                    penalty_order::Integer, penalty_weight::Real, vignetting_penalty_weight::Real,
                    solver::Union{ConjugateGradient, SparseCholesky},
                    measurement_noise::Union{IntensityNoise, IntensityAndPositionNoise} = IntensityNoise(),
                    gains_only::Bool = false)
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
    VgtkCore.pointwise!(_Sees(_frame_region(camera, Wf * raw_scale, Hf * raw_scale, rim_margin_px), cosd(Float64(fov_deg))), inside, Uall, Fall)
    lin = findall(inside)                                                 # the domain: linear indices into one H × W plane
    P = length(lin)
    P > 0 || throw(ArgumentError("fit_vignetting: the fit domain is empty"))
    upix = Uall[lin]; rays = Fall[lin]
    Uall = Fall = U2 = inside = nothing
    position_noise = measurement_noise isa IntensityAndPositionNoise
    rayJ = nothing                                                        # the rays' back-projection Jacobians (the noise's G_x)
    if position_noise
        rayJ = similar(upix, SMatrix{3, 2, Float64, 6})
        VgtkCore.pointwise!(_RayJacobian(VgtkCore.model(camera)), rayJ, upix; exec = VgtkCore.OnBackend(be))   # the camera moved too
    end
    seconds[:backproject] = _elapsed(be, t0)

    # the chart: centred away from the coverage, its extent the coverage's
    t0 = time_ns()
    stride = max(1, P ÷ 2000)
    center = _chart_center(Array(rays[1:stride:end]), Rs)
    mincos = minimum(minimum(_CosTo(center, R), rays) for R in Rs)
    spacing = deg2rad(Float64(chart_spacing_deg))
    ch = AzimuthalChart(center, acos(clamp(mincos, -1.0, 1.0)) + spacing, spacing)
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
    nG = gains ? K - 1 : 0
    # the second differences of V's control points along each axis of their lattice (Core's `difference_rows`): DᵀD, the weak
    # penalty that fixes the control points the domain does not reach
    D2 = fits_v ? (D = VgtkCore.difference_rows(size(VgtkCore.control_points(vignetting)), 2); Matrix(D' * D)) : zeros(1, 1)
    v = vignetting
    MA = fits_v ? length(VgtkCore.active_coefficients(VgtkCore.log_spline(v), Array(upix[1:1])[1])[1]) : 1
    VI = similar(frames[1], SVector{MA, Int}, P); VW = similar(frames[1], SVector{MA, Float64}, P); LV = similar(frames[1], Float64, P)
    exec = VgtkCore.OnBackend(be)
    if fits_v
        # log V's rows at the domain pixels: its basis at its active control points, which its control points do not move (log V
        # is linear in them), so once
        VgtkCore.pointwise!(_ActiveIndices(v), VI, upix; exec); VgtkCore.pointwise!(_LogRows(v), VW, upix; exec)
    else                                                                  # V fixed: no V rows
        fill!(VI, SVector{1, Int}(1)); fill!(VW, SVector{1, Float64}(0.0))
    end
    VgtkCore.pointwise!(_LogV(v), LV, upix, rays; exec)
    fits_v || all(isfinite, Array(LV)) || throw(ArgumentError("fit_vignetting: the given V is not positive at every domain pixel"))
    xh = Float64[]; cL = Float64[]; active = falses(nL); rel = NaN; retcode = :solved
    iters = 0; change = 0.0
    nsolves = position_noise ? 1 + measurement_noise.iterations : 1
    Σx = position_noise ? _measurement_covariance(measurement_noise) : nothing
    LMs = position_noise ? [_zeros(be, Float64, P) for _ in 1:K] : nothing        # the model of log I per frame: its change between solves
    weight_of = k -> _IntensityWeight()                                   # the first solve's weights: I²
    for it in 1:nsolves
        inv_scale = scale_of(weight_of)

        # the normal equations
        t0 = time_ns()
        S = _zeros(be, Float64, nL, 49); BG = _zeros(be, Float64, nL, max(nG, 1))
        BV = _zeros(be, Float64, nL, nV); bL = _zeros(be, Float64, nL)
        Wp = _zeros(be, Float64, P); Yp = _zeros(be, Float64, P); Wk = _zeros(be, Float64, P); Yk = _zeros(be, Float64, P)
        GVd = _zeros(be, Float64, nV, max(nG, 1)); Gdiag = zeros(nG); bG = zeros(nG)
        acc! = _accumulate_kernel!(be, 256); gvb! = _gain_vblock_kernel!(be, 256)
        for k in 1:K
            col = gains && k >= 2 ? k - 1 : 0
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
            VgtkCore.pointwise!(_LogV(vn), LV, upix, rays; exec)
            c = Statistics.mean(Array(LV))
            v = VgtkCore.with_control_points(vn, vec(Matrix{Float64}(VgtkCore.control_points(vn))) .- c)
            cL .+= c; LV .-= c
        end
        position_noise || break
        # the next solve's weights: the whitening at this solve's model of every frame (and its log I, for the change between solves)
        lg = vcat(0.0, xh[(end - nG + 1):end], zeros(K - 1 - nG)); cLm = _upload(be, reshape(cL, ch.n, ch.n)); mk! = _model_kernel!(be, 256)
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
        weight_of = k -> _IntensityAndPositionWeight(models[k], upix, rays, rayJ, Σx)
    end
    log_gains = vcat(0.0, xh[(end - nG + 1):end], zeros(K - 1 - nG))
    gains_only && return exp.(log_gains), retcode, iters
    cLd = _upload(be, reshape(cL, ch.n, ch.n))

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
    scene = reshape(copy(cL), ch.n, ch.n); scene[reshape(.!active, ch.n, ch.n)] .= NaN
    meta = VignettingMetadata(ch, scene, repeat(exp.(log_gains), 1, C), sqrt(Statistics.mean(abs2, Float64.(R_all))), VgtkCore.mad_scale(R_all),
                              (Float64(q[1]), Float64(q[2]), Float64(q[3])), rel, solver isa ConjugateGradient ? :cg : :cholesky,
                              measurement_noise, change, P, count(active), NamedTuple{_STAGES}(map(s -> seconds[s], _STAGES)))
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
