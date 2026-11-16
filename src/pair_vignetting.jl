# =============================================================================
# The model-free vignetting of a sweep: the pair equation on rings of the field angle
# =============================================================================
#
# A camera that only rotates sees one direction d at a sensor point u_k in every frame k that sees it, I_k = L(d)·V(u_k) (one
# exposure: no gains). For two frames k, l the scene cancels, ln I_k − ln I_l = ln V(u_k) − ln V(u_l): the PAIR EQUATION. On rings
# of a field angle θ(u) (V taken constant on each), every correspondence of rings (a, b), a < b, measures ln V_a − ln V_b; the
# median over a ring pair's correspondences is its measurement (robust to the scene's edges under the registration error), and
# least squares over the ring pairs, weighted by their counts, gives ln V per ring with ln V₁ = 0 (the first ring the reference:
# the gauge eliminated, never pinned). No spline of V and no model of the scene: the supplement's model-free reference against
# which `fit_vignetting`'s V is read.
#
# An ADDITIVE level c in the frames (I = L·V + c: a veiling glare or a black-level error) breaks the cancellation — the pair's log
# ratio then depends on L — and flattens the solution at the rim, where V is small; `fit_additive_level` estimates c as the level
# at which the log ratio no longer depends on the scene's brightness (`additive_level_slope`) and the solution is read with it
# removed.
#
# The correspondences are the frames sampled at directions (`sample_frames`: one batch per frame on the frames' backend); a
# direction that any frame sees saturated is dropped from every frame (`symmetric`: the clipped frame and the frames that see the
# same direction unclipped alike — masking only the clipped frame leaves its partners, which biases the rim's pairs low). The
# per-frame-pair selection is one broadcast per pair on the backend; only the compacted correspondences come to the host, where
# the ring pairs' medians and the least squares (at most (R − 1)R/2 rows) are reductions.
# =============================================================================

"""
    PairCorrespondences

The correspondences of a sweep on rings of a field angle ([`ring_correspondences`](@ref)): per correspondence the frames
`frames[:, i] = (k, l)`, the rings `rings[:, i] = (a, b)` (a < b; frame `k` sees the direction in ring `a`), the values
`values[:, i] = (I_k, I_l)` of the plane read, the field angles `angles[:, i] = (θ_k, θ_l)` (degrees), and the ring edges
`edges` (degrees) with the field-angle cut `tmax`; `n_frames`
the frames of the sweep, `n_directions` the directions sampled and `n_dropped` those dropped as saturated in some frame.
Host arrays.
"""
struct PairCorrespondences
    edges::Vector{Float64}
    tmax::Float64
    frames::Matrix{Int32}
    rings::Matrix{Int32}
    values::Matrix{Float64}
    angles::Matrix{Float64}
    n_frames::Int
    n_directions::Int
    n_dropped::Int
end
Base.length(pc::PairCorrespondences) = size(pc.frames, 2)

# The ring of a field angle θ (degrees) among `edges` (an NTuple: kernel-lawful), 0 outside [edges[1], min(edges[end], tmax)).
@inline function _ring_of(θ, edges::NTuple{M, Float64}, tmax) where {M}
    (isfinite(θ) && θ >= edges[1] && θ < tmax && θ < edges[M]) || return Int32(0)
    q = Int32(1)
    for j in 2:(M - 1)
        θ >= edges[j] && (q = Int32(j))
    end
    return q
end
# the field angle of a seen sensor point, NaN where unseen
struct _AngleAt{F}
    field_angle::F
end
Adapt.adapt_structure(to, r::_AngleAt) = _AngleAt(Adapt.adapt(to, r.field_angle))
@inline (r::_AngleAt)(ok, u) = ok ? Float64(r.field_angle(u)) : NaN

"""
    ring_correspondences(camera::BoundedCamera, frames, rotations, directions; saturated, saturation_margin_px, frame_map, rim_margin_px, fov_deg,
                         plane, field_angle, edges, tmax = last(edges), symmetric = true,
                         plane_maps = ntuple(_ -> frame_map, 3)) -> PairCorrespondences

The correspondences of the pair equation: the frames sampled at `directions` ([`sample_frames`](@ref), its masks and keywords;
`plane` the colour plane read), each seen direction's sensor point put on its ring of `field_angle(u)` (degrees; a kernel-lawful
callable of the camera pixel, e.g. the equisolid angle about the vignetting's centre) among `edges`, cut at `tmax`, and every
pair of frames that see a direction in two different rings one correspondence. Each frame's `saturated` mask is dilated by
`saturation_margin_px` camera pixels (≈ the registration error; [`fit_vignetting`](@ref)'s one dilation). `symmetric = true`: a
direction any frame whose domain sees it has saturated there is dropped from every frame (`fit_vignetting`'s one symmetric
test); `false`: from that frame alone, as `sample_frames` masks. The sampling and the per-frame-pair selection run on the frames' backend.
"""
function ring_correspondences(camera::VgtkCore.BoundedCamera, frames::AbstractVector, rotations::AbstractVector,
                              directions::AbstractArray{<:StaticVector{3}}; saturated::AbstractVector, saturation_margin_px::Real,
                              frame_map, rim_margin_px::Real,
                              fov_deg::Real, plane::Integer, field_angle, edges::AbstractVector{<:Real}, tmax::Real = last(edges),
                              symmetric::Bool = true, plane_maps = ntuple(_ -> frame_map, 3))
    1 <= plane <= 3 || throw(ArgumentError("ring_correspondences: plane must be 1, 2 or 3, got $plane"))
    length(edges) >= 3 && issorted(edges) && allunique(edges) ||
        throw(ArgumentError("ring_correspondences: edges must be increasing with at least two rings, got $edges"))
    K = length(frames)
    kw = (; frame_map, rim_margin_px, fov_deg, vignetting = (u, f) -> 1.0, plane_maps)
    # the masks dilated by `saturation_margin_px` (fit_vignetting's one helper, `_dilated_saturation`)
    be = KA.get_backend(frames[1])
    margin = _checked_saturation_margin("ring_correspondences", saturation_margin_px)
    n_dropped = 0
    if symmetric                                                           # the directions some frame sees saturated: out of every frame
        clear = fill(fill!(similar(saturated[1]), false), K)               # one all-clear mask, read by every frame
        s = sample_frames(camera, frames, rotations, directions; saturated = clear, kw...)
        Hf, Wf = size(frames[1], 1), size(frames[1], 2)
        sees = _Sees(frame_region(camera, _covered_extent(frame_map, Wf, Hf)..., rim_margin_px), cosd(Float64(fov_deg)))
        clipped = _saturated_directions(saturated, VgtkCore.model(camera), sees, frame_map, rotations, directions,
                                        margin, be)                        # fit_vignetting's symmetric test
        n_dropped = count(clipped .& vec(any(s.seen; dims = 2)))
        seen = s.seen .& .!clipped
    else                                                                   # each frame's own dilated mask (sample_frames' rule)
        masks = _dilated_saturation(saturated, margin, frame_map, be)
        s = sample_frames(camera, frames, rotations, directions; saturated = [masks[:, :, k] for k in 1:K], kw...)
        seen = s.seen
    end
    ang = similar(seen, Float64)
    ang .= _AngleAt(field_angle).(seen, s.pixels)                          # the field angle once; NaN where unseen
    ring = similar(seen, Int32)
    ring .= _ring_of.(ang, Ref(Tuple(Float64.(edges))), Float64(tmax))     # 0 where unseen (a NaN angle)
    V = view(s.values, :, :, plane)
    A = Int32[]; B = Int32[]; Ia = Float64[]; Ib = Float64[]; Ta = Float64[]; Tb = Float64[]; Ka = Int32[]; Kb = Int32[]
    for k in 1:K, l in (k + 1):K                                           # one broadcast per frame pair on the backend
        rk = view(ring, :, k); rl = view(ring, :, l)
        keep = findall((rk .> 0) .& (rl .> 0) .& (rk .!= rl))
        isempty(keep) && continue
        a = Array(rk[keep]); b = Array(rl[keep]); va = Array(view(V, :, k)[keep]); vb = Array(view(V, :, l)[keep])
        lo = a .< b                                                        # orient every correspondence a < b
        append!(A, ifelse.(lo, a, b)); append!(B, ifelse.(lo, b, a))
        append!(Ia, ifelse.(lo, va, vb)); append!(Ib, ifelse.(lo, vb, va))
        ta = Array(view(ang, :, k)[keep]); tb = Array(view(ang, :, l)[keep])
        append!(Ta, ifelse.(lo, ta, tb)); append!(Tb, ifelse.(lo, tb, ta))
        append!(Ka, ifelse.(lo, Int32(k), Int32(l))); append!(Kb, ifelse.(lo, Int32(l), Int32(k)))
    end
    return PairCorrespondences(Float64.(edges), Float64(tmax), permutedims(hcat(Ka, Kb)), permutedims(hcat(A, B)),
                               permutedims(hcat(Ia, Ib)), permutedims(hcat(Ta, Tb)), K, length(directions), n_dropped)
end

"""
    PairVignettingMetadata

The diagnostics of [`ring_pair_vignetting`](@ref): the ring edges and cut, the additive level removed (`offset`), the ring
pairs' medians `pair_median[a, b]` (NaN where none) and summed weights `pair_weight[a, b]` (a < b), the robust residual scale
`residual_mad` (the weighted median of |ln I_k − ln I_l − (ln V_a − ln V_b)|), the correspondences used per ring
`ring_count`, and the rings the ring-pair graph does not connect to the first (`unidentified`: NaN in the estimate).
"""
struct PairVignettingMetadata
    edges::Vector{Float64}
    tmax::Float64
    offset::Float64
    pair_median::Matrix{Float64}
    pair_weight::Matrix{Float64}
    residual_mad::Float64
    ring_count::Vector{Float64}
    unidentified::Vector{Int}
end

# The weighted median of the sorted values `y` under the weights `w` (the lower median: the first value whose cumulative
# weight reaches half of the total).
function _sorted_weighted_median(y, w)
    t = sum(w); t > 0 || return NaN
    c = zero(t)
    for i in eachindex(y, w)
        c += w[i]; 2c >= t && return Float64(y[i])
    end
    return Float64(y[end])
end

# The correspondences grouped by ring pair (a, b): the permutation sorting them by ring pair and then by `y`, and each ring
# pair's range of it (`ring_pair_vignetting`'s medians, `additive_level_slope`'s within-pair slopes).
function _ring_pair_runs(pc::PairCorrespondences, y)
    R = length(pc.edges) - 1
    key = (Int.(view(pc.rings, 1, :)) .- 1) .* R .+ Int.(view(pc.rings, 2, :))
    p = sortperm(collect(zip(key, y)))
    runs = UnitRange{Int}[]
    lo = 1
    while lo <= length(p)
        hi = lo
        while hi < length(p) && key[p[hi + 1]] == key[p[lo]]
            hi += 1
        end
        push!(runs, lo:hi); lo = hi + 1
    end
    return p, runs
end

# The pair equation's left-hand sides at the level `c` and their weights (the product of the frames' weights; a value at or
# below c: no row)
function _pair_rows(pc::PairCorrespondences, c::Real, frame_weights)
    I1 = view(pc.values, 1, :) .- c; I2 = view(pc.values, 2, :) .- c
    ok = (I1 .> 0) .& (I2 .> 0)
    y = ifelse.(ok, log.(max.(I1, floatmin())) .- log.(max.(I2, floatmin())), 0.0)
    w = ifelse.(ok, frame_weights[view(pc.frames, 1, :)] .* frame_weights[view(pc.frames, 2, :)], 0.0)
    return y, w
end

"""
    ring_pair_vignetting(pc::PairCorrespondences; offset = 0.0, frame_weights = ones(pc.n_frames)) -> FitResult

The model-free vignetting of the correspondences `pc` (one deterministic computation: the algorithm, no verb): per ring pair
(a, b) the weighted median of `ln(I_k − c) − ln(I_l − c)` (`c = offset`, the additive level removed; each correspondence
weighted by its frames' `frame_weights`, a bootstrap's multiplicities), then the least squares of `ln V_a − ln V_b` = that
median over the ring pairs, each row weighted by its summed weight, with `ln V₁ = 0` (the first ring the gauge, eliminated).
The estimate is `ln V` per ring (`NaN` for a ring the ring pairs do not connect to the first); the termination record
`(Success, ClosedForm)` (`(InitialFailure, NoRule)` when no ring pair has a correspondence), `rms` the weighted rms of the ring-pair rows, `n` the correspondences
used, the metadata a [`PairVignettingMetadata`](@ref).
"""
function ring_pair_vignetting(pc::PairCorrespondences; offset::Real = 0.0, frame_weights::AbstractVector{<:Real} = ones(pc.n_frames))
    length(frame_weights) == pc.n_frames ||
        throw(DimensionMismatch("ring_pair_vignetting: $(length(frame_weights)) frame weights for $(pc.n_frames) frames"))
    R = length(pc.edges) - 1
    y, w = _pair_rows(pc, offset, Float64.(frame_weights))
    a = view(pc.rings, 1, :); b = view(pc.rings, 2, :)
    p, runs = _ring_pair_runs(pc, y)                                      # by ring pair, then by value: the medians' order
    med = fill(NaN, R, R); wt = zeros(R, R); cnt = zeros(R)
    for run in runs
        idx = view(p, run); i = Int(a[first(idx)]); j = Int(b[first(idx)])
        ws = sum(view(w, idx))
        if ws > 0
            med[i, j] = _sorted_weighted_median(view(y, idx), view(w, idx)); wt[i, j] = ws
            cnt[i] += ws; cnt[j] += ws
        end
    end
    rows = [(i, j) for i in 1:R for j in (i + 1):R if wt[i, j] > 0]
    x = fill(NaN, R)
    meta(r, u) = PairVignettingMetadata(pc.edges, pc.tmax, Float64(offset), med, wt, r, cnt, u)
    isempty(rows) && return VgtkCore.FitResult(x, VgtkCore.ReturnCode.InitialFailure, VgtkCore.NoRule, 0, NaN, 0, meta(NaN, collect(2:R)))
    conn = _connected_to_first(R, rows)
    free = [q for q in 2:R if conn[q]]; col = Dict(q => c for (c, q) in enumerate(free))
    rows = [(i, j) for (i, j) in rows if conn[i] && conn[j]]
    M = zeros(length(rows), length(free)); m = zeros(length(rows)); s = zeros(length(rows))
    for (r, (i, j)) in enumerate(rows)
        haskey(col, i) && (M[r, col[i]] = 1.0); haskey(col, j) && (M[r, col[j]] = -1.0)
        m[r] = med[i, j]; s[r] = sqrt(wt[i, j])
    end
    z = isempty(free) ? Float64[] : (s .* M) \ (s .* m)
    x[1] = 0.0; x[free] .= z
    res = (M * z .- m)
    rms = sqrt(sum(s .^ 2 .* res .^ 2) / sum(abs2, s))
    # the robust scale of the correspondences' residuals under the solution
    ok = (w .> 0) .& isfinite.(x[a]) .& isfinite.(x[b])
    e = abs.(y[ok] .- (x[a[ok]] .- x[b[ok]])); we = w[ok]
    q = sortperm(e)
    mad = isempty(e) ? NaN : _sorted_weighted_median(e[q], we[q])
    return VgtkCore.FitResult(x, VgtkCore.ReturnCode.Success, VgtkCore.ClosedForm, 0, rms, count(ok), meta(mad, [q for q in 2:R if !conn[q]]))
end

# The rings the ring-pair graph connects to the first (the gauge's ring): a breadth-first search over the rows.
function _connected_to_first(R::Int, rows)
    adj = [Int[] for _ in 1:R]
    for (i, j) in rows
        push!(adj[i], j); push!(adj[j], i)
    end
    seen = falses(R); seen[1] = true; queue = [1]
    while !isempty(queue)
        q = popfirst!(queue)
        for n in adj[q]
            seen[n] || (seen[n] = true; push!(queue, n))
        end
    end
    return seen
end

"""
    additive_level_slope(pc::PairCorrespondences, c; frame_weights = ones(pc.n_frames)) -> Float64

The dependence of the pair equation on the scene's brightness at the level `c`: the pooled within-ring-pair least-squares slope of
the residual `e = ln(I_k − c) − ln(I_l − c) − (v(θ_k) − v(θ_l))` on the brightness `ln(I_k − c) − v(θ_k) + ln(I_l − c) − v(θ_l)`
(`v` the [`ring_pair_vignetting`](@ref) solution at `c`, linear in the field angle between the ring centres and held beyond: the
vignetting's variation inside a ring taken out of both, so neither carries it), each ring pair centred on its own median of `e`
and mean of the brightness, the correspondences past 5 robust deviations of their pair's median left out. Zero at the true
level (the log ratio then is the vignetting's, whatever `L`); positive below it (a residual level lifts the dark directions'
ratio toward 0) and negative above it.
"""
function additive_level_slope(pc::PairCorrespondences, c::Real; frame_weights::AbstractVector{<:Real} = ones(pc.n_frames))
    fw = Float64.(frame_weights)
    x = VgtkCore.estimate(ring_pair_vignetting(pc; offset = c, frame_weights = fw))
    mids = (pc.edges[1:(end - 1)] .+ pc.edges[2:end]) ./ 2
    ok = isfinite.(x); xm = mids[ok]; xv = x[ok]
    length(xv) >= 2 || return NaN
    v(θ) = θ <= xm[1] ? xv[1] : θ >= xm[end] ? xv[end] :
           (j = searchsortedlast(xm, θ); xv[j] + (xv[j + 1] - xv[j]) * (θ - xm[j]) / (xm[j + 1] - xm[j]))
    y, w = _pair_rows(pc, c, fw)
    I1 = view(pc.values, 1, :) .- c; I2 = view(pc.values, 2, :) .- c
    va = v.(view(pc.angles, 1, :)); vb = v.(view(pc.angles, 2, :))
    e0 = y .- (va .- vb)
    b = ifelse.(w .> 0, log.(max.(I1, floatmin())) .- va .+ log.(max.(I2, floatmin())) .- vb, 0.0)
    p, runs = _ring_pair_runs(pc, e0)
    num = 0.0; den = 0.0
    for run in runs
        idx = view(p, run)
        ws = view(w, idx); sum(ws) > 0 || continue
        es = view(e0, idx); m = _sorted_weighted_median(es, ws)
        e = es .- m; ae = abs.(e); q = sortperm(ae); sc = _sorted_weighted_median(ae[q], ws[q])
        keep = (ws .> 0) .& (ae .<= 5 * 1.4826 * sc)
        wk = ws[keep]; sum(wk) > 0 || continue
        bk = view(b, idx)[keep]; bk = bk .- sum(wk .* bk) / sum(wk)
        num += sum(wk .* e[keep] .* bk); den += sum(wk .* bk .^ 2)
    end
    return den > 0 ? num / den : NaN
end

"""
    fit_additive_level(pc::PairCorrespondences; bounds, tolerance, frame_weights = ones(pc.n_frames), max_iter = 100) -> FitResult

The additive level `c` of the sweep's frames (`I = L·V + c`, the plane `pc` read): the root in `bounds = (lo, hi)` (the frames'
units; a correspondence with a value at or below a level is no row at it) of [`additive_level_slope`](@ref) — the level at which the pair equation's log ratio no longer depends on the scene's
brightness — by bisection to a bracket of `tolerance`. The estimate is `c`; the termination record `(Success, ToleranceMet)` when
the bracket closed within `max_iter` iterations (`(MaxIters, IterationCap)` otherwise; `(Failure, NoRule)` when the slope does not
change sign over `bounds`: no root there, the estimate the end of the smaller slope); `rms` the robust residual scale of
[`ring_pair_vignetting`](@ref) at `c`, `n` its correspondences, the metadata `(; vignetting, slope)`: that solution at `c` and the
slope there.
"""
function fit_additive_level(pc::PairCorrespondences; bounds::Tuple{Real, Real}, tolerance::Real,
                            frame_weights::AbstractVector{<:Real} = ones(pc.n_frames), max_iter::Integer = 100)
    lo, hi = Float64.(bounds)
    lo < hi && tolerance > 0 || throw(ArgumentError("fit_additive_level: bounds must be (lo, hi) with lo < hi and tolerance > 0, got $bounds, $tolerance"))
    length(pc) > 0 || throw(ArgumentError("fit_additive_level: no correspondences"))
    any((view(pc.values, 1, :) .> hi) .& (view(pc.values, 2, :) .> hi)) ||
        throw(ArgumentError("fit_additive_level: no correspondence has both values above the upper bound $hi"))
    f(c) = additive_level_slope(pc, c; frame_weights)
    a, b = lo, hi; fa, fb = f(a), f(b)
    it = 0
    if !(isfinite(fa) && isfinite(fb) && fa * fb <= 0)
        c = abs(fa) <= abs(fb) ? a : b
        code, rule = VgtkCore.ReturnCode.Failure, VgtkCore.NoRule
    else
        while b - a > tolerance && it < max_iter
            it += 1
            m = (a + b) / 2; fm = f(m)
            if fa * fm <= 0
                b, fb = m, fm
            else
                a, fa = m, fm
            end
        end
        c = (a + b) / 2
        code, rule = b - a > tolerance ? (VgtkCore.ReturnCode.MaxIters, VgtkCore.IterationCap) : (VgtkCore.ReturnCode.Success, VgtkCore.ToleranceMet)
    end
    r = ring_pair_vignetting(pc; offset = c, frame_weights)
    return VgtkCore.FitResult(c, code, rule, it, VgtkCore.metadata(r).residual_mad, r.n, (; vignetting = r, slope = f(c)))
end
