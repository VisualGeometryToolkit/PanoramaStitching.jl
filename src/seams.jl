# =============================================================================
# score_seams — how well a vignetting makes the frames agree where they overlap
# =============================================================================

# Whether frame k holds a measurement of direction i: seen (the masks, saturation included) with a positive value and
# vignetting.
@inline _measured(vals, vig, seen, i, k) = seen[i, k] && vals[i, k] > 0 && vig[i, k] > 0

# Per direction: the median over the other frames that measure it of log(I_k/V_k) − log(I_last/V_last), `last` the highest
# index that measures it (the frame the unblended render shows); NaN unless at least two frames measure it. The median by rank
# selection (no buffer: at most a few dozen frames).
@kernel function _seam_ratio_kernel!(out, @Const(vals), @Const(vig), @Const(seen), K::Int, offset::Int)
    i = @index(Global)
    @inbounds begin
        last = 0
        for k in 1:K
            _measured(vals, vig, seen, i, k) && (last = k)
        end
        n = 0
        for k in 1:K
            (k != last && _measured(vals, vig, seen, i, k)) && (n += 1)
        end
        if n == 0
            out[offset + i] = NaN
        else
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
            out[offset + i] = (v1 + v2) / 2 - ref
        end
    end
end

"""
    score_seams(camera, frames, rotations, view; vignetting, saturated, circle, raw_scale, rim_margin_px, fov_deg, plane,
                plane_offsets, chunk, gains = nothing) -> (; ratio, rms, mad, n)

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
more: `rms` and the robust `mad` (1.4826·MAD; misalignment and moving content
inflate the rms), over `n` pixels. `gains` (optional, one per frame: the
exposure gains `g_k` of [`fit_vignetting`](@ref)'s `metadata.gains`) divides
each frame's exposure out as well, `log(I_k/(g_k·V_k))`; `nothing` takes the
frames at one exposure.

The camera and the masks are [`sample_frames`](@ref)'s (`camera` its forward
map: a `CameraModel` or a grid's `SplineProjection`). The raster is sampled in
chunks of `chunk` pixels (the samples are `chunk × K × 3` values on the frames'
backend: chunk to fit a shared GPU's memory). Returns `ratio` (rows × columns,
NaN where fewer than two frames measure the pixel) on the frames' backend.
"""
function score_seams(camera, frames::AbstractVector, rotations::AbstractVector, view::PanoramaView; vignetting,
                     saturated::AbstractVector, circle, raw_scale::Integer, rim_margin_px::Real, fov_deg::Real, plane::Integer,
                     plane_offsets, chunk::Integer, gains = nothing)
    K = length(frames)
    1 <= plane <= 3 || throw(ArgumentError("score_seams: plane $plane of a 3-plane frame"))
    chunk > 0 || throw(ArgumentError("score_seams: chunk must be positive, got $chunk"))
    gains === nothing || length(gains) == K || throw(DimensionMismatch("score_seams: $K frames and $(length(gains)) gains"))
    nv, nλ = view.size; N = nv * nλ
    proto = frames[1]; be = KA.get_backend(proto)
    D = similar(proto, SVector{3, Float64}, nv, nλ)
    VgtkCore.pointwise!(_RayAt{Float64}(view.rays), D, CartesianIndices((nv, nλ)))
    D = vec(D)
    ratio = fill!(similar(proto, Float64, N), NaN)
    rk! = _seam_ratio_kernel!(be, 256)
    G = gains === nothing ? nothing : reshape(_upload(be, Float64.(collect(gains))), 1, K)
    for lo in 1:chunk:N
        hi = min(lo + chunk - 1, N)
        S = sample_frames(camera, frames, rotations, D[lo:hi]; saturated, circle, raw_scale, rim_margin_px, fov_deg, vignetting,
                          plane_offsets)
        G === nothing || (S.vignetting .*= G)                              # I_k / (g_k·V_k): the frame's exposure divided out too
        rk!(ratio, Base.view(S.values, :, :, plane), S.vignetting, S.seen, K, lo - 1; ndrange = hi - lo + 1)
        KA.synchronize(be)
    end
    r = filter(!isnan, Array(ratio))
    isempty(r) && throw(ArgumentError("score_seams: no pixel of the raster is measured by two frames"))
    med = Statistics.median(r)
    return (; ratio = reshape(ratio, nv, nλ), rms = sqrt(Statistics.mean(abs2, r)), mad = 1.4826 * Statistics.median(abs.(r .- med)),
            n = length(r))
end
