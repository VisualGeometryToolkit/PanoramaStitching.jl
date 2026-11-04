# =============================================================================
# The panorama: the reverse warp of the frames through a view's pixel → ray map (Core's image_warp)
# =============================================================================

"""
    true_up(up_pitch, up_roll) -> SVector{3}

The true up direction in the common (first-frame) frame: `Rz(roll)·(0, −cos p, −sin p)`,
`p` the first frame's pitch-down against the true horizon and `roll` its roll, in
degrees.
"""
function true_up(pitch::Real, roll::Real)
    p, r = deg2rad(pitch), deg2rad(roll)
    u0 = SVector(0.0, -cos(p), -sin(p))
    return SVector(u0[1] * cos(r) - u0[2] * sin(r), u0[1] * sin(r) + u0[2] * cos(r), u0[3])
end

"""
    up_frame(up_pitch, up_roll) -> SMatrix{3,3}

`R` with `d = R·d_true`: the true-up frame (its −y the [`true_up`](@ref)
direction, its forward the common frame's forward projected onto the horizontal).
"""
function up_frame(pitch::Real, roll::Real)
    u = normalize(true_up(pitch, roll)); z = SVector(0.0, 0.0, 1.0)
    zt = normalize(z - dot(z, u) * u); yt = -u
    return hcat(cross(yt, zt), yt, zt)
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

```julia
using VisualGeometryToolkitCore: CameraModel, Lens, Perspective, Equidistant, Stereographic, Cylindrical, px
cyl = panorama_view(Cylindrical(), (-120.0, 120.0, -40.0, 40.0), 0.05)                       # cylindrical
rect = panorama_view(CameraModel(1000.0px, SA[960.0px, 540.0px]; lens = Lens(Perspective())), 1920, 1080)   # rectilinear
down = SMatrix{3, 3}(1.0, 0, 0, 0, 0, -1, 0, 1, 0)                                           # looking along +y (down)
planet = panorama_view(CameraModel(400.0px, SA[1000.0px, 1000.0px]; lens = Lens(Stereographic())), 2000, 2000; rotation = down)   # 'little planet'
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
    render_panorama(camera, frames, rotations, view; vignetting, circle, raw_scale, rim_margin_px, fov_deg,
                    plane_offsets = nothing, gains = nothing) -> (; image, owner, coverage, extent)

The panorama of `frames` in the raster `view` ([`panorama_view`](@ref)) by the REVERSE warp, Core's
`image_warp`/`image_warp!`: every pixel's ray `d` (the view's pixel → ray map) is rotated into each frame `k`
(`f = R_kᵀ·d`, `rotations[k]` with `d = R_k·f`), projected through the calibrated `camera` to a sensor point between pixels
(a `CameraModel` through its `imaging_transform`, a `SplineProjection` — a grid model's forward map,
`fit_camera(SplineProjection, grid, region; tolerance)`), and sampled there bilinearly — one fused map
`camera ∘ R_kᵀ ∘ view.rays` per frame — then divided by the vignetting `vignetting(u, f)`, looked up at the sensor point `u`
(camera pixels) and the ray `f` (the frame's camera frame) — a model defined on the sensor reads `u`, one defined on the
ray's direction (a field angle) reads `f` — and by the frame's exposure gain `gains[k]` when `gains` is given (one per
frame: [`fit_vignetting`](@ref)'s `metadata.gains`; `nothing`, the frames taken as one exposure). No seam, no blend, and
each pixel keeps the value of the LAST frame that sees it, so a wrong vignetting shows as a step at every frame boundary.

A frame sees a pixel when its ray is within `fov_deg` of the camera frame's z axis, its sensor point lies inside `circle`
(a Core `Circle` or `Ellipse` in camera pixels, `WholePlane()` when the lens has no image circle) less `rim_margin_px` and
at least 2 px inside the sensor — the geometric masks of [`sample_frames`](@ref), evaluated as batches at the warp's own
sensor points. The render shows the frames as they are: a saturated pixel is rendered with its clipped value (a window that
clips in every frame comes out bright, not empty); saturation masks measurements only ([`sample_frames`](@ref),
[`fit_vignetting`](@ref), [`score_seams`](@ref)). `frames` are `H × W × 3` linear images with `raw_scale` camera pixels
per frame pixel; `plane_offsets` are the three planes' sample
positions relative to the pixel centre in frame pixels (`nothing`: all at the centre) — a 2 × 2 Bayer binning's R and B sit
at `(±¼, ±¼)` (VisualGeometryToolkitFeatures' `load_raw(…; mode = :bin2).plane_offsets`), and each plane is read at the
sensor point less its offset. The masks and the vignetting are at the pixel centre's point.

Every step runs on the backend that owns `frames` (GPU arrays: one kernel per warp, mask and update; the same code on the
host); for device arrays `vignetting` must be kernel-lawful (it is evaluated only where a frame sees the pixel). The warp's
math — the sensor points, the rays, the bilinear weights — is in Float64 whatever the frames' element type (a Float32 sensor
coordinate of a 4000 px sensor is 2.4e-4 px coarse, enough to move a sampled value by 1e-4 relative); the image is in the
frames' element type.

Returns `image` (rows × columns × 3, NaN where no frame sees the pixel), `owner` (the frame whose value a pixel holds, 0 for
none), `coverage` (how many frames see each pixel), all on the frames' backend, and the view's `extent`.
"""
function render_panorama(camera, frames::AbstractVector, rotations::AbstractVector, view::PanoramaView; vignetting, circle,
                         raw_scale::Integer, rim_margin_px::Real, fov_deg::Real, plane_offsets = nothing, gains = nothing)
    offsets = _plane_offsets(plane_offsets)
    K = length(frames)
    length(rotations) == K || throw(DimensionMismatch("render_panorama: $K frames and $(length(rotations)) rotations"))
    gains === nothing || length(gains) == K || throw(DimensionMismatch("render_panorama: $K frames and $(length(gains)) gains"))
    Hf, Wf, C = size(frames[1])
    C == 3 || throw(DimensionMismatch("render_panorama: frames are H × W × 3, got $(size(frames[1]))"))
    T = float(eltype(frames[1])); s = raw_scale
    fwd = _forward_map(camera, frames[1])
    sees = _Sees(_frame_region(circle, Wf * s, Hf * s, rim_margin_px), cosd(Float64(fov_deg)))
    nv, nλ = view.size
    rect = VgtkCore.Rect(0, 0, nλ, nv)
    proto = frames[1]
    image = fill!(similar(proto, T, nv, nλ, 3), T(NaN)); owner = fill!(similar(proto, Int, nv, nλ), 0)
    coverage = fill!(similar(proto, Int, nv, nλ), 0)
    U = similar(proto, SVector{2, Float64}, nv, nλ); F = similar(proto, SVector{3, Float64}, nv, nλ)
    ok = similar(proto, Bool, nv, nλ); V = similar(proto, T, nv, nλ); w = similar(proto, T, nv, nλ)
    idx = CartesianIndices((nv, nλ))
    scale = LinearMap(SMatrix{2, 2, Float64, 4}(1 / s, 0.0, 0.0, 1 / s))        # camera pixels → the binned frame's
    for k in 1:K
        Rt = LinearMap(SMatrix{3, 3, Float64, 9}(rotations[k])')
        # frame k's warp: the output (row, col) → the frame's (row, col); the same chain without the camera, the rays f
        fk = Rt ∘ view.rays
        bm = VgtkCore.image_warp(Float64, scale ∘ fwd ∘ fk, rect)
        VgtkCore.pointwise!(_SensorPoint{Float64}(bm, s), U, idx)
        VgtkCore.pointwise!(_RayAt{Float64}(fk), F, idx)
        VgtkCore.pointwise!(sees, ok, U, F)
        VgtkCore.pointwise!(_Vignetting(vignetting, T), V, ok, U, F)
        gains === nothing || (V .*= T(gains[k]))                           # the frame's exposure divided out with V
        for c in 1:3
            # a plane whose samples sit at o_c (frame pixels) is read at the frame point less o_c
            bmc = iszero(offsets[c]) ? bm : VgtkCore.image_warp(Float64, Translation(-offsets[c]) ∘ scale ∘ fwd ∘ fk, rect)
            VgtkCore.image_warp!(w, bmc, Base.view(frames[k], :, :, c))
            img = Base.view(image, :, :, c)
            img .= ifelse.(ok, w ./ V, img)
        end
        owner .= ifelse.(ok, k, owner); coverage .+= ok
    end
    return (; image, owner, coverage, view.extent)
end

# The warp's sensor point of an output pixel (row, col), in camera pixels: the warp lands on the binned frame's (row, col), whose
# COLMAP point scaled by `s` is the camera's. And the ray map of the output pixel's COLMAP centre. Isbits functors (a closure
# over the element type would not reach a device).
struct _SensorPoint{T, B}
    bm::B
    s::T
end
_SensorPoint{T}(bm::B, s) where {T, B} = _SensorPoint{T, B}(bm, T(s))
# The device form: the warp adapted by Core's rule (a SplineProjection's control grid moved to the device), the scale kept.
Adapt.adapt_structure(to, p::_SensorPoint{T}) where {T} = _SensorPoint{T}(Adapt.adapt(to, p.bm), p.s)
@inline (p::_SensorPoint{T})(I) where {T} =
    (q = p.bm(SVector{2, T}(I[1], I[2])); SVector{2, T}(p.s * (q[2] - T(0.5)), p.s * (q[1] - T(0.5))))
struct _RayAt{T, M}
    rays::M
end
_RayAt{T}(rays::M) where {T, M} = _RayAt{T, M}(rays)
@inline (r::_RayAt{T})(I) where {T} = SVector{3, T}(r.rays(SVector(I[2] - 0.5, I[1] - 0.5)))

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
