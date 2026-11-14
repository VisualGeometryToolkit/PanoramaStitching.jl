# The rotation bundle's references, derived and independent of the solver: its residuals and their Jacobian by ForwardDiff
# through the camera's `project`, the Cauchy M-estimator's asymptotic covariance under Gaussian pixel noise (the bound on an
# estimate's error from the truth), and the stationarity radius of an estimate (the bound on its distance from the minimum).
@testsetup module BundleReference
using LinearAlgebra, StaticArrays
import ForwardDiff
import VisualGeometryToolkitCore as VgtkCore
export bundle_model, cauchy_gradient, cauchy_efficiency, chisq_even_quantile, stationarity_radius, coordinates_between

"""
    bundle_model(camera, R, D, tracks, free) -> (; r, J, cols)

The bundle's residuals `π(R_kᵀ d_j) − x_jk` (2 per observation, `r`) at rotations `R` (frame => matrix, `d = R_k·f_k`) and
directions `D` (one unit vector per track), and their Jacobian `J` in the variables: a right rotation step `δ_k`
(`R_k·exp(δ_k)`) of each frame in `free` (sorted, 3 each), then each direction's step on Core's `SphereManifold` chart at
`d_j` (2 each). `cols[i]` are observation i's nonzero columns (its frame's, if free, and its track's).
"""
function bundle_model(camera, R, D, tracks, free)
    col = Dict(k => 3 * (i - 1) for (i, k) in enumerate(free))
    nR = 3 * length(free)
    m = sum(length, tracks)
    r = zeros(2m); J = zeros(2m, nR + 2 * length(D)); cols = Vector{Vector{Int}}()
    row = 0
    for (j, t) in enumerate(tracks)
        ch = VgtkCore.SphereManifold(SVector{3, Float64}(D[j]))
        for (k, x) in t
            Rk = SMatrix{3, 3, Float64, 9}(R[k])
            f(v) = (Rv = haskey(col, k) ? Rk * VgtkCore.exp_so3(v[SOneTo(3)]) : Rk;
                    VgtkCore.project(camera, Rv' * VgtkCore.retract(ch, SVector(v[4], v[5]))) - x)
            v0 = zero(SVector{5, Float64})
            r[(row + 1):(row + 2)] = f(v0)
            Jl = ForwardDiff.jacobian(f, v0)
            c = haskey(col, k) ? [col[k] .+ (1:3); (nR + 2j - 1):(nR + 2j)] : collect((nR + 2j - 1):(nR + 2j))
            J[(row + 1):(row + 2), c] = haskey(col, k) ? Jl : Jl[:, 4:5]
            push!(cols, c)
            row += 2
        end
    end
    return (; r, J, cols)
end

# Observation i's residual pair, and its Jacobian on its nonzero columns.
_obs(model, i) = (SVector(model.r[2i - 1], model.r[2i]), model.J[(2i - 1):(2i), model.cols[i]])

"""
    cauchy_gradient(model, c, obs) -> g

The gradient `Σ 2wᵢ Jᵢᵀ rᵢ` (`wᵢ = c²/(c² + ‖rᵢ‖²)`) of the Cauchy cost `Σ c² log(1 + ‖rᵢ‖²/c²)` over the observations `obs`.
"""
function cauchy_gradient(model, c, obs)
    g = zeros(size(model.J, 2))
    for i in obs
        ri, Ji = _obs(model, i)
        g[model.cols[i]] .+= Ji' * (2c^2 / (c^2 + dot(ri, ri)) .* ri)
    end
    return g
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
    stationarity_radius(model, c; H = nothing) -> (; radius, H)

How far an estimate can be from the bundle's minimum, from its own stationarity: the robust cost's gradient
`g = Σ 2wᵢ Jᵢᵀ rᵢ` and Hessian `H = Σ Jᵢᵀ (2wᵢ I + 4wᵢ′ rᵢrᵢᵀ)₊ Jᵢ` (each 2 × 2 block's negative curvature clipped, as
Triggs' correction does: its eigenvalues `2w + 4w′s` along rᵢ and `2w` across) at the estimate. The Newton step to the
minimum of the local quadratic, in H's metric, is `√(gᵀH⁻¹g)`; the rounding of `g`'s sum (each component within
`m·eps·Σ|terms|`, m the observations) bounds the rest, `√λmax(H⁻¹)·‖g_round‖`. With `H` given (another estimate's, at
the same minimum), the step is measured in that metric. It is the local quadratic model's distance, not a global bound.
"""
function stationarity_radius(model, c; H = nothing)
    Hgiven = H
    n = size(model.J, 2); m = length(model.cols)
    gabs = zeros(n); H = zeros(n, n)
    for i in 1:m
        ri, Ji = _obs(model, i)
        s = dot(ri, ri)
        w = c^2 / (c^2 + s)
        λ = max(2w - 4w^2 / c^2 * s, 0.0)                                # along rᵢ (w′ = −w²/c²); 2w across
        A₊ = 2w * I + (s > 0 ? (λ - 2w) / s : 0.0) * (ri * ri')
        ci = model.cols[i]
        H[ci, ci] .+= Ji' * A₊ * Ji
        gabs[ci] .+= abs.(Ji') * abs.(2w .* ri)
    end
    g = cauchy_gradient(model, c, 1:m)
    Hs = Hgiven === nothing ? Symmetric(H) : Hgiven
    newton = sqrt(max(dot(g, cholesky(Hs) \ g), 0.0))
    rounding = sqrt(1 / first(eigvals(Hs, 1:1))) * norm(m * eps() .* gabs)
    return (; radius = newton + rounding, H = Hs)
end

"""
    coordinates_between(Ra, Da, Rb, Db, free) -> Δ

Estimate b in estimate a's coordinates (`bundle_model`'s variables at a): `log(R_aᵀ R_b)` per free frame, then each
direction's chart coordinates at `d_a` (Core's `SphereManifold` `local_coordinates`, the inverse of its `retract`).
"""
function coordinates_between(Ra, Da, Rb, Db, free)
    Δ = Float64[]
    for k in free
        append!(Δ, VgtkCore.log_so3(SMatrix{3, 3, Float64, 9}(Ra[k])' * SMatrix{3, 3, Float64, 9}(Rb[k])))
    end
    for (da, db) in zip(Da, Db)
        append!(Δ, VgtkCore.local_coordinates(VgtkCore.SphereManifold(SVector{3, Float64}(da)), SVector{3, Float64}(db)))
    end
    return Δ
end
end
