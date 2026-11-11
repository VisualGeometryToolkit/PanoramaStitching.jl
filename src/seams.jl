# =============================================================================
# score_seams — how well a vignetting makes the frames agree where they overlap
# =============================================================================

# The frames' gains of `score_seams` and `render_panorama`: `nothing`, or a K × C matrix (per frame and colour plane, as
# `fit_vignetting`'s `metadata.gains`).
function _check_gains(f, gains, K, C)
    gains === nothing && return nothing
    gains isa AbstractMatrix && size(gains) == (K, C) ||
        throw(DimensionMismatch("$f: gains are a $K × $C matrix (per frame and colour plane: fit_vignetting's metadata.gains), got $(summary(gains))"))
    return nothing
end

# Whether frame k holds a measurement of direction i: seen (the masks, saturation included) with a positive value and
# vignetting.
@inline _measured(vals, vig, seen, i, k) = seen[i, k] && vals[i, k] > 0 && vig[i, k] > 0

# Per direction i: the median over the other frames that measure it of log(I_k/V_k) − log(I_last/V_last), `last` the highest
# index that measures it (the frame the unblended render shows); NaN unless at least two frames measure it. The median by rank
# selection (no buffer: at most a few dozen frames). A per-point functor of Core's `pointwise!` over the direction indices,
# the samples its fields (moved to the backend by `OnBackend`).
struct _SeamRatio{A, W, S}
    vals::A
    vig::W
    seen::S
    K::Int
end
Adapt.@adapt_structure _SeamRatio
@inline function (s::_SeamRatio)(I)
    vals = s.vals; vig = s.vig; seen = s.seen; K = s.K; i = I[1]
    @inbounds begin
        last = 0
        for k in 1:K
            _measured(vals, vig, seen, i, k) && (last = k)
        end
        n = 0
        for k in 1:K
            (k != last && _measured(vals, vig, seen, i, k)) && (n += 1)
        end
        n == 0 && return NaN
        ref = log(Float64(vals[i, last]) / Float64(vig[i, last]))
        r1 = (n + 1) ÷ 2; r2 = n ÷ 2 + 1
        v1 = NaN; v2 = NaN
        for k in 1:K
            (k != last && _measured(vals, vig, seen, i, k)) || continue
            xk = log(Float64(vals[i, k]) / Float64(vig[i, k]))
            nless = 0; nleq = 0
            for l in 1:K
                (l != last && _measured(vals, vig, seen, i, l)) || continue
                xl = log(Float64(vals[i, l]) / Float64(vig[i, l]))
                nless += xl < xk; nleq += xl <= xk
            end
            (nless < r1 <= nleq) && (v1 = xk)
            (nless < r2 <= nleq) && (v2 = xk)
        end
        return (v1 + v2) / 2 - ref
    end
end

"""
    score_seams(camera::BoundedCamera, frames, rotations, view; vignetting, saturated, frame_map, rim_margin_px, fov_deg, plane,
                plane_maps = ntuple(_ -> frame_map, 3), chunk, gains = nothing) -> (; ratio, rms, mad, n)

How well `vignetting` makes the frames agree where they overlap, at every
pixel of the panorama raster `view` ([`panorama_view`](@ref)): the pixel's ray
is sampled in every frame ([`sample_frames`](@ref): the exact sub-pixel sensor
point, the frame's bilinear value of `plane`, the vignetting there, the masks —
saturated pixels EXCLUDED, a clipped value is no measurement) and, where at
least two frames measure it, `ratio` is the median over the other frames of
`log(I_k/V_k) − log(I_last/V_last)`, `last` the frame the unblended
[`render_panorama`](@ref) shows (the highest index). A correct V leaves only
noise, misregistration and parallax; a wrong one leaves a step at every frame
boundary. The score is the ratio's spread over the pixels measured twice or
more: `rms` and the robust `mad` (StatsBase's `mad`, normalized: the 1.4826·MAD; misalignment and moving content
inflate the rms), over `n` pixels. `gains` (optional, `K × 3`, per frame and
colour plane: [`fit_vignetting`](@ref)'s `metadata.gains`) divides each
frame's exposure gain of `plane` out as well, `log(I_k/(g_k,plane·V_k))`;
`nothing` takes the frames as one exposure.

The camera, the masks and the frames' and planes' maps (`frame_map`,
`plane_maps`) are [`sample_frames`](@ref)'s (`camera` a
VisualGeometryToolkitCore `BoundedCamera`: its forward map and its `imaged_region`). The raster is sampled in
chunks of `chunk` pixels (the samples are `chunk × K × 3` values on the frames'
backend: chunk to fit a shared GPU's memory). Returns `ratio` (rows × columns,
NaN where fewer than two frames measure the pixel) on the frames' backend.
"""
function score_seams(camera::VgtkCore.BoundedCamera, frames::AbstractVector, rotations::AbstractVector, view::PanoramaView;
                     vignetting, saturated::AbstractVector, frame_map, rim_margin_px::Real, fov_deg::Real, plane::Integer,
                     plane_maps = ntuple(_ -> frame_map, 3), chunk::Integer, gains = nothing)
    K = length(frames)
    1 <= plane <= 3 || throw(ArgumentError("score_seams: plane $plane of a 3-plane frame"))
    chunk > 0 || throw(ArgumentError("score_seams: chunk must be positive, got $chunk"))
    _check_gains("score_seams", gains, K, size(frames[1], 3))
    nv, nλ = view.size; N = nv * nλ
    proto = frames[1]; be = KA.get_backend(proto)
    D = similar(proto, SVector{3, Float64}, nv, nλ)
    VgtkCore.pointwise!(_AtPixelCentre{Float64}(view.rays), D, CartesianIndices((nv, nλ)))
    D = vec(D)
    ratio = fill!(similar(proto, Float64, N), NaN)
    G = gains === nothing ? nothing : reshape(Adapt.adapt(be, Float64.(collect(gains[:, plane]))), 1, K)
    exec = VgtkCore.OnBackend(be)
    for lo in 1:chunk:N
        hi = min(lo + chunk - 1, N)
        S = sample_frames(camera, frames, rotations, D[lo:hi]; saturated, frame_map, rim_margin_px, fov_deg, vignetting,
                          plane_maps)
        vals = Base.view(S.values, :, :, plane)
        G === nothing || (S.vignetting .*= G)                             # I_k / (g_k·V_k): the frame's exposure too
        VgtkCore.pointwise!(_SeamRatio(vals, S.vignetting, S.seen, K), Base.view(ratio, lo:hi), CartesianIndices((hi - lo + 1,)); exec)
        KA.synchronize(be)
    end
    r = filter(!isnan, Array(ratio))
    isempty(r) && throw(ArgumentError("score_seams: no pixel of the raster is measured by two frames"))
    return (; ratio = reshape(ratio, nv, nλ), rms = sqrt(Statistics.mean(abs2, r)), mad = StatsBase.mad(Float64.(r); normalize = true),
            n = length(r))
end
