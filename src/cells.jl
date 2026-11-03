# =============================================================================
# The world cells of a registered sequence and their samples in every frame
# =============================================================================
#
# The common frame is the first frame's camera frame (x right, y down, z forward); a world
# cell is a (latitude, longitude) box of it, up = −y and longitude 0 forward — Core's
# `Equirectangular` axes.
# =============================================================================

"""
    world_cells(cell_deg) -> Matrix{Float64} (N × 4)

The near-equal-area latitude/longitude grid of the sphere: latitude bands of
`cell_deg` degrees from −90°, each split into `max(1, round(360 cos(lat)/cell_deg))`
equal longitude cells from −180°. Row `i` is `(lat_lo, lat_hi, lon_lo, lon_hi)` in
degrees.
"""
function world_cells(cell::Real)
    rows = NTuple{4, Float64}[]
    for lo in range(-90.0, 90.0 - cell; step = cell)
        hi = lo + cell; latc = (lo + hi) / 2
        n = max(1, round(Int, 360 * cosd(latc) / cell)); wl = 360 / n
        for q in 0:(n - 1)
            push!(rows, (lo, hi, -180 + q * wl, -180 + (q + 1) * wl))
        end
    end
    return [r[j] for r in rows, j in 1:4]
end

"The unit direction of (latitude, longitude) in degrees: Core's `Equirectangular` back-projection (up = −y, longitude 0 = +z, +90° = +x)."
world_direction(lat, lon) = VgtkCore.backproject(VgtkCore.Equirectangular(), SVector(deg2rad(lon), deg2rad(lat)))

"The centres (latitude, longitude) of the cells' rows."
cell_centres(cells::AbstractMatrix) = (0.5 .* (cells[:, 1] .+ cells[:, 2]), 0.5 .* (cells[:, 3] .+ cells[:, 4]))

"""
    CellSamples

The samples of the world cells in every frame: `cells` (N × 4, [`world_cells`](@ref)),
`val` (K × N × 3: the mean value per channel), `theta` (K × N: the recorded field
angle of the cell, degrees), `theta_model` (K × N: the lens's field angle — the mean
angle of the cell's sample directions to the optical axis, degrees), `px` (K × N × 2:
the mean pixel, camera pixels), `ok` (K × N: the frame saw the whole cell).
"""
struct CellSamples
    cells::Matrix{Float64}
    val::Array{Float32, 3}
    theta::Matrix{Float32}
    theta_model::Matrix{Float32}
    px::Array{Float32, 3}
    ok::BitMatrix
end

"""
    sample_cells(frames, saturated, rotations, camera; raw_scale, circle, cell_deg, sub, blur_px, rim_margin_px, fov_deg,
                 field_angle, f_px = nothing, rng, say = _ -> nothing) -> CellSamples

The world cells of `cell_deg` sampled in every frame: in every cell `sub`² sample
directions (the cell's sub-grid, each jittered uniformly within its sub-cell from
`rng`, the same in every frame), each taken into frame `k` (`f = R_kᵀ·d`,
`rotations[k]` with `d = R_k·f`) and projected through the calibrated `camera`
(Core's `project`, a generic model's pixel solve started at its equidistant guess about
the circle), sampled bilinearly (Core's `bilinear_sample`) in the frame smoothed by
VisualGeometryToolkitFeatures' masked `gaussian_blur` (σ `blur_px` frame pixels, the
image circle the mask). A cell is valid in a frame when its centre is within
`fov_deg` of the optical axis (the direction of the circle centre) and every
sample direction projects inside the `circle = (cx, cy, r)` less `rim_margin_px`, at
least 2 px inside the image, onto no `saturated` pixel and with a positive green
value. `field_angle = :lens` records the lens's angle as `theta`; `:equisolid`
records `2 asin(ρ / 2 f_px)` of the mean pixel's radius ρ about the circle centre
(a lens-independent ring convention).

`frames` are `H × W × 3` linear images with `raw_scale` camera pixels per frame pixel,
`saturated` their `H × W` masks.
"""
function sample_cells(frames::AbstractVector, saturated::AbstractVector, rotations::AbstractVector, camera; raw_scale::Integer, circle,
                      cell_deg::Real, sub::Integer, blur_px::Real, rim_margin_px::Real, fov_deg::Real, field_angle::Symbol,
                      f_px::Union{Nothing, Real} = nothing, rng::AbstractRNG, say = _ -> nothing)
    field_angle in (:lens, :equisolid) || throw(ArgumentError("sample_cells: field_angle = :$field_angle is not :lens or :equisolid"))
    field_angle === :equisolid && f_px === nothing && throw(ArgumentError("sample_cells: the equisolid field angle needs f_px"))
    K = length(frames); cells = world_cells(cell_deg); N = size(cells, 1); S = sub
    cx, cy, rc = circle; s = raw_scale
    Hf, Wf = size(frames[1], 1), size(frames[1], 2); W = Wf * s; H = Hf * s
    axis = VgtkCore.backproject(camera, SVector(cx, cy))
    jit = rand(rng, N, S, S, 2) .- 0.5
    val = zeros(Float32, K, N, 3); th = fill(NaN32, K, N); thm = fill(NaN32, K, N); px = fill(NaN32, K, N, 2); ok = falses(K, N)
    mask = [_inside(i, j, s, circle) for i in 1:Hf, j in 1:Wf]
    centre = [world_direction((c[1] + c[2]) / 2, (c[3] + c[4]) / 2) for c in eachrow(cells)]
    θr = rad2deg(VgtkCore.angle_between(VgtkCore.backproject(camera, SVector(cx + rc, cy)), axis)) / rc   # degrees per pixel at the rim (the guess)
    for k in 1:K
        smooth = VgtkFeatures.gaussian_blur(frames[k], blur_px, mask)
        R = SMatrix{3, 3, Float64, 9}(rotations[k]); sat = saturated[k]
        infov = findall(i -> rad2deg(VgtkCore.angle_between(R' * centre[i], axis)) < fov_deg, 1:N)
        Threads.@threads for i in infov
            acc = zero(SVector{3, Float64}); pacc = zero(SVector{2, Float64}); tacc = 0.0; good = true
            for a in 1:S, b in 1:S
                la = cells[i, 1] + (cells[i, 2] - cells[i, 1]) * ((a - 0.5) / S + jit[i, a, b, 1] / S)
                lo = cells[i, 3] + (cells[i, 4] - cells[i, 3]) * ((b - 0.5) / S + jit[i, a, b, 2] / S)
                f = R' * world_direction(la, lo)
                θ = VgtkCore.angle_between(f, axis); φ = atan(f[2], f[1]); r0 = rad2deg(θ) / θr
                u = _project_with(camera, f, SVector(cx + r0 * cos(φ), cy + r0 * sin(φ)))
                if u === nothing || !all(isfinite, u) || hypot(u[1] - cx, u[2] - cy) >= rc - rim_margin_px ||
                   !(2 < u[1] < W - 2 && 2 < u[2] < H - 2) || sat[clamp(floor(Int, u[2] / s) + 1, 1, Hf), clamp(floor(Int, u[1] / s) + 1, 1, Wf)]
                    good = false; break
                end
                v = VgtkCore.bilinear_sample(smooth, Val(3), SVector(u[1] / W, u[2] / H))
                v[2] > 0 || (good = false; break)
                acc += v; pacc += SVector{2, Float64}(u[1], u[2]); tacc += θ
            end
            good || continue
            n = S * S
            val[k, i, :] = acc / n; px[k, i, :] = pacc / n; thm[k, i] = rad2deg(tacc / n)
            th[k, i] = field_angle === :lens ? thm[k, i] : rad2deg(2 * asin(clamp(hypot(pacc[1] / n - cx, pacc[2] / n - cy) / (2 * f_px), 0, 1)))
            ok[k, i] = true
        end
        say("  frame $(k - 1): $(length(infov)) cells in the field, $(count(view(ok, k, :))) valid")
    end
    return CellSamples(cells, val, th, thm, px, ok)
end
