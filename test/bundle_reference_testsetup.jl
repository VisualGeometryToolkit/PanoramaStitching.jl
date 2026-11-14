# The rotation bundle's references, derived and independent of the solver: its residuals and their Jacobian by ForwardDiff
# through the camera's `project`, the Cauchy M-estimator's asymptotic covariance under Gaussian pixel noise (the bound on an
# estimate's error from the truth), and the stationarity radius of an estimate (the bound on its distance from the minimum).
@testsetup module BundleReference
using LinearAlgebra, StaticArrays
import ForwardDiff
import VisualGeometryToolkitCore as VgtkCore
export bundle_model, cauchy_efficiency, chisq_even_quantile, stationarity_radius, coordinates_between

# A tangent basis of the unit sphere at d.
function tangent_basis(d)
    a = abs(d[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
    t1 = normalize(cross(d, a))
    return t1, cross(d, t1)
end

"""
    bundle_model(camera, R, D, tracks, free) -> (; r, J, obs)

The bundle's residuals `π(R_kᵀ d_j) − x_jk` (2 per observation, `r`) at rotations `R` (frame => matrix, `d = R_k·f_k`) and
directions `D` (one unit vector per track), and their Jacobian `J` in the variables: a right rotation step `δ_k`
(`R_k·exp(δ_k)`) of each frame in `free` (sorted, 3 each), then a tangent step `η_j` of each direction
(`normalize(d_j + T_j η_j)`, 2 each). `obs` lists each residual pair's (track, frame).
"""
function bundle_model(camera, R, D, tracks, free)
    col = Dict(k => 3 * (i - 1) for (i, k) in enumerate(free))
    nR = 3 * length(free)
    m = sum(length, tracks)
    r = zeros(2m); J = zeros(2m, nR + 2 * length(D)); obs = Tuple{Int, Int}[]
    row = 0
    for (j, t) in enumerate(tracks)
        t1, t2 = tangent_basis(D[j])
        for (k, x) in t
            Rk = SMatrix{3, 3, Float64, 9}(R[k])
            f(v) = begin
                Rv = haskey(col, k) ? Rk * VgtkCore.exp_so3(SVector(v[1], v[2], v[3])) : Rk
                d = normalize(D[j] + v[4] * t1 + v[5] * t2)
                VgtkCore.project(camera, Rv' * d) - x
            end
            v0 = zeros(5)
            r[(row + 1):(row + 2)] = f(v0)
            Jl = ForwardDiff.jacobian(f, v0)
            haskey(col, k) && (J[(row + 1):(row + 2), (col[k] + 1):(col[k] + 3)] = Jl[:, 1:3])
            J[(row + 1):(row + 2), (nR + 2j - 1):(nR + 2j)] = Jl[:, 4:5]
            push!(obs, (j, k))
            row += 2
        end
    end
    return (; r, J, obs)
end

"""
    cauchy_efficiency(c, σ) -> (; κ, curvature)

The Cauchy M-estimator `ρ(s) = c² log(1 + s/c²)` (`s = ‖r‖²`, Core's `CauchyShape` at loss width `c`) on 2-D residuals
`r ~ N(0, σ²I)`: `κ`, its asymptotic variance over least squares' (`cov = κ σ² (JᵀJ)⁻¹`, the sandwich `B⁻¹ M B⁻¹` with
`B = E[∇²ρ]`, `M = E[∇ρ ∇ρᵀ]`, both isotropic), and `curvature`, `B`'s scalar (its Hessian per unit `JᵀJ`). The
expectations are over the radius's Rayleigh density, by the midpoint rule to 12σ.
"""
function cauchy_efficiency(c, σ; n = 200_000)
    h = 12σ / n
    Ew = Ews = Ew2s = 0.0
    for i in 1:n
        ρr = (i - 0.5) * h
        p = ρr / σ^2 * exp(-ρr^2 / (2σ^2)) * h
        s = ρr^2
        w = c^2 / (c^2 + s)
        Ew += p * w; Ews += p * (-w^2 / c^2) * s; Ew2s += p * w^2 * s
    end
    B = 2Ew + 2Ews               # E[∇²ρ] = (2E[w] + 2E[w′s]) I, w′ = dw/ds = −w²/c²
    M = 2Ew2s                    # E[∇ρ∇ρᵀ] = 4E[w² r rᵀ] = 2E[w² s] I
    return (; κ = M / B^2 / σ^2, curvature = B)
end

"""
    chisq_even_quantile(p, ν) -> x

The `p` quantile of χ² with an even `ν` degrees of freedom (`1 − e^{−x/2} Σ_{k<ν/2} (x/2)^k/k!`, by bisection).
"""
function chisq_even_quantile(p, ν)
    iseven(ν) || throw(ArgumentError("chisq_even_quantile: ν = $ν is odd"))
    cdf(x) = 1 - exp(-x / 2) * sum((x / 2)^k / factorial(k) for k in 0:(ν ÷ 2 - 1))
    lo, hi = 0.0, 10.0 * ν + 100
    for _ in 1:200
        mid = (lo + hi) / 2
        cdf(mid) < p ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"""
    stationarity_radius(model, c) -> (; radius, newton, H, cost)

How far an estimate can be from the bundle's minimum, from its own stationarity: the robust cost's gradient
`g = Σ 2wᵢ Jᵢᵀ rᵢ` and Hessian `H = Σ Jᵢᵀ (2wᵢ I + 4wᵢ′ rᵢrᵢᵀ)₊ Jᵢ` (each 2 × 2 block's negative curvature clipped, as
Triggs' correction does) at the estimate. The Newton step to the minimum of the local quadratic, in H's metric, is
`√(gᵀH⁻¹g)`; the rounding of `g`'s sum (each component within `m·eps·Σ|terms|`, m the observations) bounds the rest,
`√λmax(H⁻¹)·‖g_round‖`.
"""
function stationarity_radius(model, c)
    (; r, J) = model
    n = size(J, 2)
    g = zeros(n); gabs = zeros(n); H = zeros(n, n); cost = 0.0
    m = length(r) ÷ 2
    for i in 1:m
        rows = (2i - 1):(2i)
        ri = SVector(r[rows[1]], r[rows[2]])
        Ji = J[rows, :]
        s = dot(ri, ri)
        w = c^2 / (c^2 + s); w′ = -w^2 / c^2
        cost += c^2 * log1p(s / c^2)
        ψ = 2w * ri
        g .+= Ji' * ψ
        gabs .+= abs.(Ji') * abs.(ψ)
        A = Symmetric(Matrix(2w * I(2) + 4w′ * ri * ri'))
        e = eigen(A)
        A₊ = e.vectors * Diagonal(max.(e.values, 0.0)) * e.vectors'
        H .+= Ji' * A₊ * Ji
    end
    F = cholesky(Symmetric(H))
    newton = sqrt(max(dot(g, F \ g), 0.0))
    rounding = sqrt(1 / eigmin(Symmetric(H))) * norm(m * eps() .* gabs)
    return (; radius = newton + rounding, newton, H = Symmetric(H), cost)
end

"""
    coordinates_between(Ra, Da, Rb, Db, free) -> Δ

Estimate b in estimate a's coordinates (`bundle_model`'s variables at a): `log(R_aᵀ R_b)` per free frame, then each
direction's tangent coordinates `T_jᵀ d_b` at `d_a`.
"""
function coordinates_between(Ra, Da, Rb, Db, free)
    Δ = Float64[]
    for k in free
        append!(Δ, VgtkCore.log_so3(SMatrix{3, 3, Float64, 9}(Ra[k])' * SMatrix{3, 3, Float64, 9}(Rb[k])))
    end
    for (da, db) in zip(Da, Db)
        t1, t2 = tangent_basis(da)
        append!(Δ, (dot(t1, db), dot(t2, db)))
    end
    return Δ
end
end
