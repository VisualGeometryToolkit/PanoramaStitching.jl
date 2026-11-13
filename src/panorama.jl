# =============================================================================
# The panorama: the reverse warp of the frames through a view's pixel → ray map (Core's image_warp)
# =============================================================================

"""
    true_up(pitch, roll) -> SVector{3}

The true up direction in the common (first-frame) frame: `Rz(roll)·(0, −cos p, −sin p)`,
`p` the first frame's pitch-down against the true horizon and `roll` its roll, in
degrees. A level first frame's up is its camera frame's −y:

```jldoctest
julia> using PanoramaStitching

julia> true_up(0, 0) == [0, -1, 0]
true
```
"""
function true_up(pitch::Real, roll::Real)
    p, r = deg2rad(pitch), deg2rad(roll)
    u0 = SVector(0.0, -cos(p), -sin(p))
    return SVector(u0[1] * cos(r) - u0[2] * sin(r), u0[1] * sin(r) + u0[2] * cos(r), u0[3])
end

"""
    up_frame(up::AbstractVector) -> SMatrix{3,3}
    up_frame(pitch, roll) -> SMatrix{3,3}

`R` with `d = R·d_true`: the true-up frame of the up direction `up` (in the
common frame; normalized here) — its −y `up`, its forward (+z) the common frame's
forward `(0, 0, 1)` projected onto the horizontal, its x the right-handed
completion — VisualGeometryToolkitCore's `lookat_pose` frame (the `:cv`
convention) looking along that forward. `up_frame(pitch, roll)` is
`up_frame(true_up(pitch, roll))` ([`true_up`](@ref), degrees); `up` may also be
a sweep's own, [`sweep_up`](@ref)`(rotations).up`. The `up` keyword of
[`panorama_view`](@ref).
"""
function up_frame(up::AbstractVector)
    u = normalize(SVector{3, Float64}(up))
    z = SVector(0.0, 0.0, 1.0)
    forward = z - dot(z, u) * u                              # the common forward on the horizontal (Core normalizes it)
    # Core's look-at frame: z the forward, its y along the vector given (the image's down: −up), x = y × z
    return SMatrix{3, 3, Float64, 9}(VgtkCore.rotation(VgtkCore.lookat_pose(zero(SVector{3, Float64}), forward, -u)))
end
up_frame(pitch::Real, roll::Real) = up_frame(true_up(pitch, roll))

"""
    sweep_up(rotations) -> (; up, eigenvalues)

The up direction of a roll-free sweep from its frames' rotations alone
(`d = R_k·f_k`, the common frame's directions): a roll-free frame's image x axis
(`R_k`'s first column) is horizontal, so up is orthogonal to every one — the
eigenvector of the least eigenvalue of `Σ_k x_k x_kᵀ` — signed against the
frames' summed image-down axes (`R_k`'s second columns). `eigenvalues` are the
matrix's, ascending: the least is ≈ 0 when the sweep is roll free (the
diagnostic of the estimate). It holds whatever the frames' pitch, unlike the
axis of the frames' rotations. Its frame is [`up_frame`](@ref)`(sweep_up(R).up)`.
"""
function sweep_up(rotations)
    A = zeros(3, 3)
    for R in rotations
        x = SMatrix{3, 3, Float64, 9}(R)[:, 1]; A .+= x * x'
    end
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(A)); a = SVector{3}(E.vectors[:, 1])
    down = sum(SMatrix{3, 3, Float64, 9}(R)[:, 2] for R in rotations)
    return (; up = dot(a, down) > 0 ? -a : a, eigenvalues = E.values)
end

"""
    PanoramaView

The raster a panorama is rendered into: `rays`, its pixel → ray map (a
`Transformation` from the raster's COLMAP pixel `(x, y)` to the direction in
the common frame), `size` (rows, columns) and `extent` (`(lon₀, lon₁, lat₀,
lat₁)` degrees of a cylindrical-family raster, `nothing` for a virtual
camera). Built by [`panorama_view`](@ref).
"""
struct PanoramaView{M, E}
    rays::M
    size::Tuple{Int, Int}
    extent::E
end

# The vertical coordinate of a latitude (degrees) under a panorama projection, and back: the latitude for `Equirectangular`,
# tan(latitude) for `Cylindrical`, sin for `CylindricalEqualArea`, asinh(tan) for `Mercator`.
_vertical(projection, lat) = VgtkCore.project(projection, world_direction(lat, 0.0))[2]
_latitude(projection, v) = rad2deg(VgtkCore.project(VgtkCore.Equirectangular(), VgtkCore.backproject(projection, SVector(0.0, v)))[2])

"""
    panorama_view(projection, extent, deg_per_pixel; up = I) -> PanoramaView
    panorama_view(camera, width, height; rotation = I) -> PanoramaView

The raster of a panorama and its pixel → ray map.

- A cylindrical-family `projection` (Core's `Equirectangular()`, `Cylindrical()`, `CylindricalEqualArea()`, `Mercator()`):
  the raster spans `extent = (lon₀, lon₁, lat₀, lat₁)` degrees, `deg_per_pixel` along the longitude (its columns, from
  lon₀) and the same step in the projection's vertical coordinate (its rows, from lat₁ at the top: radians of latitude for
  the equirectangular, of `tan φ`, `sin φ` or `asinh(tan φ)` for the others), in the frame `up` (`d = up·d_up`,
  [`up_frame`](@ref) for a true-up panorama). Its pixel → ray map is `LinearMap(up) ∘ inv(projection) ∘ raster`, and its
  `extent` the raster's own (the latitudes of its row edges).
- An azimuthal view, a virtual `camera` (a Core `CameraModel`): its `width × height` pixels, their rays
  `LinearMap(rotation) ∘ Backprojection(camera)` (`d = rotation·f`, `f` the virtual camera's ray). No new projection type:
  the lens chooses the view.

```jldoctest
julia> using PanoramaStitching, StaticArrays

julia> using VisualGeometryToolkitCore: CameraModel, Lens, Perspective, Stereographic, Cylindrical, px

julia> cyl = panorama_view(Cylindrical(), (-120.0, 120.0, -40.0, 40.0), 0.05);   # cylindrical

julia> cyl.size                                     # rows in steps of tan φ, columns of longitude
(1923, 4800)

julia> rect = panorama_view(CameraModel(1000.0px, SA[960.0px, 540.0px]; lens = Lens(Perspective())), 1920, 1080);

julia> rect.size                                    # rectilinear
(1080, 1920)

julia> down = SMatrix{3, 3}(1.0, 0, 0, 0, 0, -1, 0, 1, 0);   # its axis along +y (down)

julia> planet = panorama_view(CameraModel(400.0px, SA[1000.0px, 1000.0px]; lens = Lens(Stereographic())), 2000, 2000;
                              rotation = down);    # a 'little planet'

julia> planet.size
(2000, 2000)
```
"""
function panorama_view(projection::VgtkCore.AbstractPanoramaProjection, extent, deg_per_pixel::Real;
                       up::AbstractMatrix = SMatrix{3, 3}(1.0I))
    lon0, lon1, lat0, lat1 = Float64.(extent); δ = deg2rad(Float64(deg_per_pixel))
    v0, v1 = _vertical(projection, lat0), _vertical(projection, lat1)
    nλ = round(Int, deg2rad(lon1 - lon0) / δ); nv = round(Int, (v1 - v0) / δ)
    raster = AffineMap(SMatrix{2, 2, Float64, 4}(δ, 0.0, 0.0, -δ), SVector(deg2rad(lon0), v1))
    rays = LinearMap(SMatrix{3, 3, Float64, 9}(up)) ∘ inv(projection) ∘ raster
    return PanoramaView(rays, (nv, nλ), (lon0, lon0 + rad2deg(nλ * δ), _latitude(projection, v1 - nv * δ), lat1))
end
function panorama_view(camera::VgtkCore.CameraModel, width::Integer, height::Integer; rotation::AbstractMatrix = SMatrix{3, 3}(1.0I))
    rays = LinearMap(SMatrix{3, 3, Float64, 9}(rotation)) ∘ VgtkCore.Backprojection(camera)
    return PanoramaView(rays, (Int(height), Int(width)), nothing)
end

"""
    render_panorama(camera::BoundedCamera, frames, rotations, view; vignetting, frame_map, rim_margin_px, fov_deg,
                    plane_maps = ntuple(_ -> frame_map, 3), gains = nothing) -> (; image, owner, coverage, extent)

The panorama of `frames` in the raster `view` ([`panorama_view`](@ref)) by the REVERSE warp, Core's
`image_warp`/`image_warp!`: every pixel's ray `d` (the view's pixel → ray map) is rotated into each frame `k`
(`f = R_kᵀ·d`, `rotations[k]` with `d = R_k·f`), projected through the calibrated `camera` (a VisualGeometryToolkitCore
`BoundedCamera`, through its model's `imaging_transform`) to a sensor point between pixels, and sampled there bilinearly — one fused map
`camera ∘ R_kᵀ ∘ view.rays` per frame — then divided by the vignetting `vignetting(u, f)`, looked up at the sensor point `u`
(camera pixels) and the ray `f` (the frame's camera frame) — a model defined on the sensor reads `u`, one defined on the
ray's direction (a field angle) reads `f`; a Core `AbstractVignetting` reads `u` — and by the frame's exposure gain
`gains[k, c]` of each colour plane c when `gains` is given (`K × 3`, per frame and plane: [`fit_vignetting`](@ref)'s
`metadata.gains`; `nothing`, the frames taken as one exposure). No seam, no blend, and
each pixel keeps the value of the LAST frame that sees it, so a wrong vignetting shows as a step at every frame boundary.

A frame sees a pixel when its ray is within `fov_deg` of the camera frame's z axis, its sensor point lies inside the camera's
`imaged_region` (its sensor ∩ its lens coverage ∩ its model's valid domain) less `rim_margin_px` at its lens coverage's edge,
and at least 2 px inside the frame — the geometric masks of [`sample_frames`](@ref), evaluated as batches at the warp's own
sensor points. The render shows the frames as they are: a saturated pixel is rendered with its clipped value (a window that
clips in every frame comes out bright, not empty); saturation masks observations only ([`sample_frames`](@ref),
[`fit_vignetting`](@ref), [`score_seams`](@ref)). `frames` are `H × W × 3` linear images. `frame_map` is the frames' map
onto the camera's pixels (VisualGeometryToolkitCore's `raw_frame_map(L)`), and `plane_maps` each plane's
(`bayer_plane_map(L, c)`; by default every plane's is the frame's); plane c is read through
`inv(plane_maps[c]) ∘ camera ∘ R_kᵀ ∘ rays`, all three planes in one Core `image_warp!`. The masks and the vignetting are at
the pixel centre's sensor point.

Every step runs on the backend that owns `frames` (GPU arrays: one kernel per warp, mask and update; the same code on the
host); for device arrays `vignetting` must be kernel-lawful (it is evaluated only where a frame sees the pixel). The warp's
math — the sensor points, the rays, the bilinear weights — is in Float64 whatever the frames' element type (a Float32 sensor
coordinate of a 4000 px sensor is 2.4e-4 px coarse, enough to move a sampled value by 1e-4 relative); the image is in the
frames' element type.

Returns `image` (rows × columns × 3, NaN where no frame sees the pixel), `owner` (the frame whose value a pixel holds, 0 for
none), `coverage` (how many frames see each pixel), all on the frames' backend, and the view's `extent`.
"""
function render_panorama(camera::VgtkCore.BoundedCamera, frames::AbstractVector, rotations::AbstractVector, view::PanoramaView;
                         vignetting, frame_map, rim_margin_px::Real, fov_deg::Real, plane_maps = ntuple(_ -> frame_map, 3),
                         gains = nothing)
    _checked_plane_maps("render_panorama", plane_maps)
    K = length(frames)
    length(rotations) == K || throw(DimensionMismatch("render_panorama: $K frames and $(length(rotations)) rotations"))
    Hf, Wf, C = size(frames[1])
    C == 3 || throw(DimensionMismatch("render_panorama: frames are H × W × 3, got $(size(frames[1]))"))
    _check_gains("render_panorama", gains, K, C)
    T = float(eltype(frames[1]))
    fwd = _forward_map(camera, frames[1])
    sees = _Sees(frame_region(camera, _covered_extent(frame_map, Wf, Hf)..., rim_margin_px), cosd(Float64(fov_deg)))
    nv, nλ = view.size
    rect = VgtkCore.Rect(0, 0, nλ, nv)
    proto = frames[1]
    image = fill!(similar(proto, T, nv, nλ, 3), T(NaN)); owner = fill!(similar(proto, Int, nv, nλ), 0)
    coverage = fill!(similar(proto, Int, nv, nλ), 0)
    U = similar(proto, SVector{2, Float64}, nv, nλ); F = similar(proto, SVector{3, Float64}, nv, nλ)
    ok = similar(proto, Bool, nv, nλ); V = similar(proto, T, nv, nλ); w3 = similar(proto, T, nv, nλ, 3)
    idx = CartesianIndices((nv, nλ))
    for k in 1:K
        Rt = LinearMap(SMatrix{3, 3, Float64, 9}(rotations[k])')
        # frame k's chain: the output pixel → its ray f in the frame's camera frame → the sensor point (camera pixels)
        fk = Rt ∘ view.rays
        # (the chain held as Core's BoundedMap, whose Adapt rule moves each link to the device: a SplineLens's control grids)
        VgtkCore.pointwise!(_AtPixelCentre{Float64}(VgtkCore.BoundedMap(fwd ∘ fk, rect)), U, idx)
        VgtkCore.pointwise!(_AtPixelCentre{Float64}(fk), F, idx)
        VgtkCore.pointwise!(sees, ok, U, F)
        VgtkCore.pointwise!(_WhereSeen(vignetting, T), V, ok, U, F; exec = VgtkCore.OnBackend(KA.get_backend(U)))
        # each plane read through its own map: the output (row, col) → the plane's (row, col)
        warps = map(P -> VgtkCore.image_warp(Float64, inv(P) ∘ fwd ∘ fk, rect), plane_maps)
        VgtkCore.image_warp!(w3, warps, frames[k])
        for c in 1:3
            img = Base.view(image, :, :, c)
            g = gains === nothing ? one(T) : T(gains[k, c])                   # the frame's exposure in this plane, divided out with V
            img .= ifelse.(ok, Base.view(w3, :, :, c) ./ (g .* V), img)
        end
        owner .= ifelse.(ok, k, owner); coverage .+= ok
    end
    return (; image, owner, coverage, view.extent)
end

# A map (an output pixel's chain: its ray, its sensor point) at the output pixel's COLMAP centre (Core's `pixel_centre`): an
# isbits functor (a closure over the element type would not reach a device).
struct _AtPixelCentre{T, M}
    map::M
end
_AtPixelCentre{T}(map::M) where {T, M} = _AtPixelCentre{T, M}(map)
# The device form: the map adapted by its own rule (a Core BoundedMap's: each link of its chain, a SplineLens's control grids
# moved to the device).
Adapt.adapt_structure(to, p::_AtPixelCentre{T}) where {T} = _AtPixelCentre{T}(Adapt.adapt(to, p.map))
@inline (p::_AtPixelCentre{T})(I) where {T} = p.map(VgtkCore.pixel_centre(T, I))

"""
    CUBE_FACES

The six faces of a cube map, `front, right, back, left, up, down`, each the rotation `R` taking the face camera's ray `f` to
the common frame (`d = R·f`): its columns the face's image right, image down and optical axis (see
[`render_cube_map`](@ref)).
"""
const CUBE_FACES = (
    front = SMatrix{3, 3, Float64, 9}(1, 0, 0, 0, 1, 0, 0, 0, 1),        # axis +z, right +x, down +y
    right = SMatrix{3, 3, Float64, 9}(0, 0, -1, 0, 1, 0, 1, 0, 0),       # axis +x, right −z
    back = SMatrix{3, 3, Float64, 9}(-1, 0, 0, 0, 1, 0, 0, 0, -1),       # axis −z, right −x
    left = SMatrix{3, 3, Float64, 9}(0, 0, 1, 0, 1, 0, -1, 0, 0),        # axis −x, right +z
    up = SMatrix{3, 3, Float64, 9}(1, 0, 0, 0, 0, 1, 0, -1, 0),          # axis −y, right +x, image down towards +z (front)
    down = SMatrix{3, 3, Float64, 9}(1, 0, 0, 0, 0, -1, 0, 1, 0),        # axis +y, right +x, image down towards −z (back)
)

"""
    render_cube_map(camera, frames, rotations, width; up = I, kwargs...) -> (; faces, cross)

The cube map of the panorama: six `width × width` 90° rectilinear views (a `Lens(Perspective())` virtual camera of focal
length `width/2` px, principal point at the face's centre), each a [`render_panorama`](@ref) (`kwargs` are its keywords) of
the [`panorama_view`](@ref) aimed by `up·R_face`. The faces, in the common frame `up` (x right, y down, z forward):

| face | optical axis | image right | image down |
|---|---|---|---|
| `front` | +z | +x | +y |
| `right` | +x | −z | +y |
| `back` | −z | −x | +y |
| `left` | −x | +z | +y |
| `up` | −y | +x | +z |
| `down` | +y | +x | −z |

`faces` is a NamedTuple of the six renders in that order; `cross` is the horizontal cross, `3·width × 4·width × 3`
(NaN off the cross) — `up` above `front`, the row `left, front, right, back`, `down` below `front` — in which every
shared edge of two faces is adjacent: the side faces' tops meet `up`'s edges and their bottoms `down`'s.
"""
function render_cube_map(camera, frames::AbstractVector, rotations::AbstractVector, width::Integer;
                         up::AbstractMatrix = SMatrix{3, 3}(1.0I), kwargs...)
    c = width / 2
    face = VgtkCore.CameraModel(c * VgtkCore.px, SVector(c * VgtkCore.px, c * VgtkCore.px); lens = VgtkCore.Lens(VgtkCore.Perspective()))
    U = SMatrix{3, 3, Float64, 9}(up)
    faces = map(R -> render_panorama(camera, frames, rotations, panorama_view(face, width, width; rotation = U * R); kwargs...), CUBE_FACES)
    n = Int(width); T = eltype(faces.front.image)
    cross = fill!(similar(faces.front.image, T, 3n, 4n, 3), T(NaN))
    place(img, i, j) = (cross[((i - 1) * n + 1):(i * n), ((j - 1) * n + 1):(j * n), :] .= img; nothing)
    place(faces.up.image, 1, 2); place(faces.left.image, 2, 1); place(faces.front.image, 2, 2)
    place(faces.right.image, 2, 3); place(faces.back.image, 2, 4); place(faces.down.image, 3, 2)
    return (; faces, cross)
end

"""
    coverage_extent(camera, frames, rotations, up, frame_map; δ = 0.5) -> (lon₀, lon₁, lat₀, lat₁)

The longitude and latitude box (degrees) the `frames` cover in the view of the
up frame `up` (e.g. [`up_frame`](@ref)`(sweep_up(rotations).up)`): the seen
pixels of a coarse equirectangular [`render_panorama`](@ref) at `δ` degrees per
pixel over the whole sphere (latitudes within ±89°), no vignetting, the
frames to their sensor's edge within 89° of the optical axis; the box of its
seen pixels' edges. The `extent` of a [`panorama_view`](@ref) that frames the
sweep.
"""
function coverage_extent(camera, frames::AbstractVector, rotations::AbstractVector, up::AbstractMatrix, frame_map; δ::Real = 0.5)
    v = panorama_view(VgtkCore.Equirectangular(), (-180.0, 180.0, -89.0, 89.0), δ; up)
    r = render_panorama(camera, frames, rotations, v; vignetting = (u, f) -> 1.0, frame_map, rim_margin_px = 0.0, fov_deg = 89.0)
    seen = findall(>(0), Array(r.coverage))
    isempty(seen) && throw(ArgumentError("coverage_extent: no frame covers any pixel of the view"))
    rows = extrema(I -> I[1], seen); cols = extrema(I -> I[2], seen)
    return (-180 + δ * (cols[1] - 1), -180 + δ * cols[2], 89 - δ * rows[2], 89 - δ * (rows[1] - 1))
end
