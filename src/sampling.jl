# =============================================================================
# The frames sampled at directions — the observations of the vignetting validation and fit
# =============================================================================
#
# The common frame is the first frame's camera frame (x right, y down, z forward); a direction
# `d` of it is seen by frame k as the ray `f = R_kᵀ·d` (`rotations[k]` with `d = R_k·f`), imaged
# at the sensor point `u = π(f)` of the calibrated camera's forward map (its model's `imaging_transform`), where the camera
# images it (`imaged_region`). Every per-direction step is a batch
# on the backend that owns the arrays (Core's `pointwise!` and `image_sample!`, KernelAbstractions
# kernels on a GPU, the same code on the host): the rays, the sensor points, the masks, the
# frame's value and the vignetting at the point.
# =============================================================================

"""
    world_direction(lat, lon) -> Vec3{Float64}

The unit direction of (latitude, longitude) in degrees: Core's `Equirectangular` back-projection (up = −y, longitude 0 = +z,
+90° = +x).
"""
world_direction(lat, lon) = VgtkCore.backproject(VgtkCore.Equirectangular(), SVector(deg2rad(lon), deg2rad(lat)))

# The camera's forward map ray → camera pixel as a Transformation (a kernel-lawful value on a device): its model's
# `imaging_transform` (a SplineLens camera's through its forward spline, VisualGeometryToolkitEstimation's
# `fit_camera(SplineLens, camera, imaged_region(camera); tolerance)`).
_forward_map(m::VgtkCore.CameraModel) = VgtkCore.imaging_transform(m)
_forward_map(b::VgtkCore.BoundedCamera) = _forward_map(VgtkCore.model(b))
# The forward map on the frames' backend: the camera's model moved to the device first (Core's adapt_structure: a SplineLens's
# control grids; a parametric model unchanged), then its map.
_forward_map(camera, proto) = _forward_map(Adapt.adapt(KA.get_backend(proto), VgtkCore.model(camera)))

# What the camera images less a margin `m` (camera pixels) at the edge of its lens coverage — the image circle's soft rim —
# and nowhere else: Core's `imaged_region` of the camera with its coverage eroded by `m`. The sensor and the model's valid
# domain are not eroded, so a camera whose coverage is its whole sensor (none stored: Core's default, the sensor's own `Rect`)
# is unchanged. A margin that erodes the coverage away leaves nothing imaged (an empty `Rect`), not an error.
function _imaged_less_rim(camera::VgtkCore.BoundedCamera, m)
    c = VgtkCore.lens_coverage(camera, m)                              # Core's: the coverage less m at the lens's edge
    return c === nothing ? VgtkCore.Rect2{Float64}() : VgtkCore.imaged_region(VgtkCore.with_lens_coverage(camera, c))
end

# A region of the camera's pixels in a frame's pixels, `frame_map` the frame's map onto the camera's pixels (a Core
# `SimilarityMap` without rotation: VisualGeometryToolkitCore's `raw_frame_map`), by its parts: an `Ellipse` by Core's
# similarity image under the map's inverse, a `Rect` by its corners (axis-aligned under a scale and a translation).
_in_frame(e::VgtkCore.Ellipse, frame_map) = inv(frame_map)(e)
function _in_frame(r::VgtkCore.Rect, frame_map)
    lo = inv(frame_map)(SVector{2, Float64}(r.origin[1], r.origin[2]))
    hi = inv(frame_map)(SVector{2, Float64}(r.origin[1] + r.widths[1], r.origin[2] + r.widths[2]))
    return VgtkCore.Rect(lo[1], lo[2], hi[1] - lo[1], hi[2] - lo[2])
end
_in_frame(r::VgtkCore.RegionIntersection, frame_map) = _in_frame(r.a, frame_map) ∩ _in_frame(r.b, frame_map)

# The camera's pixels a `width × height` frame covers (`frame_map` without a translation, as a Bayer cell grid's): its far
# corner's image.
_covered_extent(frame_map, width, height) = Tuple(frame_map(SVector{2, Float64}(width, height)))

# The planes' maps checked: one per plane of a 3-plane frame.
function _checked_plane_maps(what, plane_maps)
    plane_maps isa Tuple && length(plane_maps) == 3 ||
        throw(ArgumentError("$what: plane_maps holds one map per plane (3), got $(plane_maps isa Tuple ? length(plane_maps) : typeof(plane_maps))"))
    return plane_maps
end

"""
    frame_region(camera::BoundedCamera, width, height, rim_margin_px) -> region

Where a frame's sensor point counts, in camera pixels (a region answering `p ∈ region`): at least 2 px inside the
`width × height` frame (camera pixels; the bilinear taps) and inside what `camera` images — its `imaged_region`, the sensor ∩
its lens coverage ∩ its model's valid domain — less `rim_margin_px` at its lens coverage's edge (the image circle's soft
rim; the sensor's edge is not eroded, so a camera whose coverage is its whole sensor loses nothing to the margin). The
domain of [`sample_frames`](@ref), [`render_panorama`](@ref), [`score_seams`](@ref) and [`fit_vignetting`](@ref).
"""
frame_region(camera::VgtkCore.BoundedCamera, width, height, rim_margin_px) =
    VgtkCore.Rect(2.0, 2.0, width - 4.0, height - 4.0) ∩ _imaged_less_rim(camera, rim_margin_px)

# Whether a frame sees a direction: its sensor point `u` is finite and in the region and its ray `f` within the field angle of
# the camera frame's z axis (`f₃ > cos θ_max·‖f‖`) — the render's mask. A measurement (the sampling, the seam score) also
# needs the saturated mask's bilinear sample zero at every plane's sample position: no saturated pixel among those any plane's
# value blends (`sample_frames`).
struct _Sees{R, T}
    region::R
    cosmax::T
end
@inline (s::_Sees)(u, f) = all(isfinite, u) && VgtkCore.Point2(u[1], u[2]) ∈ s.region && f[3] > s.cosmax * LinearAlgebra.norm(f)

# A caller's per-point callable `g(u, f)` where the frame sees the point, NaN elsewhere: it is evaluated only at seen points (a
# grid's back-projection, say, has no ray off its domain). The vignetting keyword is read through it.
struct _WhereSeen{F, T}
    g::F
end
_WhereSeen(g::F, ::Type{T}) where {F, T} = _WhereSeen{F, T}(g)
# the callable moved to the frames' backend with it (a vignetting holding arrays, a calibration's map: Core's `OnBackend`)
Adapt.adapt_structure(to, v::_WhereSeen{F, T}) where {F, T} = _WhereSeen(Adapt.adapt(to, v.g), T)
@inline (v::_WhereSeen{F, T})(ok, u, f) where {F, T} = ok ? T(v.g(u, f)) : T(NaN)

"""
    FrameSamples

The frames sampled at a set of directions ([`sample_frames`](@ref)), N
directions × K frames: `directions` (N, the common frame), `pixels` (N × K:
the sensor point, camera pixels), `rays` (N × K: the ray in the frame's camera
frame, `R_kᵀ·d`), `values` (N × K × 3: the frame's linear value there, NaN
where unseen), `vignetting` (N × K: the vignetting model there, NaN where
unseen) and `seen` (N × K). The arrays live on the backend of the inputs.
"""
struct FrameSamples{D, P, F, V, W, S}
    directions::D
    pixels::P
    rays::F
    values::V
    vignetting::W
    seen::S
end

"""
    sample_frames(camera::BoundedCamera, frames, rotations, directions; saturated, frame_map, rim_margin_px, fov_deg,
                  vignetting, plane_maps = ntuple(_ -> frame_map, 3)) -> FrameSamples

The frames sampled at `directions` (unit 3-vectors of the common frame, any
array): for every frame `k` and direction `d`, the ray `f = R_kᵀ·d`
(`rotations[k]` with `d = R_k·f`), its exact sub-pixel sensor point
`u = π(f)` through the calibrated `camera` (a VisualGeometryToolkitCore
`BoundedCamera`, projected through its model's `imaging_transform`), the frame's value
there by bilinear sampling (Core's `image_sample!`, no smoothing) and the
vignetting model evaluated continuously there, `vignetting(u, f)` (a model
defined on the sensor reads `u`, one defined on the ray's direction reads
`f`; a calibration's vignetting, or `(u, f) -> 1.0` for none). These are the
observations of a vignetting validation (`I_k / V_k` must agree across the
frames that see a direction: their spread is the model's error;
[`score_seams`](@ref) scores them).

A frame sees a direction by [`render_panorama`](@ref)'s masks: the ray within
`fov_deg` of the camera frame's z axis, the sensor point inside the camera's
`imaged_region` (its sensor ∩ its lens coverage ∩ its model's valid domain)
less `rim_margin_px` at its lens coverage's edge, and at least 2 px inside the frame,
and no `saturated` pixel among those any plane's value blends (the four at
each plane's own sample position). `frames` are
`H × W × 3` linear images, `saturated` their `H × W` masks. `frame_map` is the
frames' map onto the camera's pixels (VisualGeometryToolkitCore's
`raw_frame_map(L)`), and `plane_maps` each plane's (`bayer_plane_map(L, c)`;
by default every plane's is the frame's): plane `c` is read at
`inv(plane_maps[c])(u)`.

Every step is a batch on the backend that owns `frames` and `directions`
(Core's `pointwise!`, `image_sample!`: one kernel each on a GPU array, the
same code on the host), so for device arrays `vignetting` must be
kernel-lawful (isbits, no allocation, no throw); it is evaluated only where
the frame sees the direction. The precision is the directions' element type.
"""
function sample_frames(camera::VgtkCore.BoundedCamera, frames::AbstractVector, rotations::AbstractVector,
                       directions::AbstractArray{<:StaticVector{3}}; saturated::AbstractVector, frame_map, rim_margin_px::Real, fov_deg::Real, vignetting,
                       plane_maps = ntuple(_ -> frame_map, 3))
    _checked_plane_maps("sample_frames", plane_maps)
    K = length(frames)
    length(rotations) == K && length(saturated) == K || throw(DimensionMismatch("sample_frames: frames, rotations and saturated disagree"))
    Hf, Wf, C = size(frames[1])
    C == 3 || throw(DimensionMismatch("sample_frames: frames are H × W × 3, got $(size(frames[1]))"))
    T = eltype(eltype(directions))
    fwd = _forward_map(camera, frames[1])
    sees = _Sees(frame_region(camera, _covered_extent(frame_map, Wf, Hf)..., rim_margin_px), T(cosd(fov_deg)))
    N = length(directions); D = reshape(directions, N)
    proto = frames[1]
    pixels = similar(proto, SVector{2, T}, N, K); rays = similar(proto, SVector{3, T}, N, K)
    values = similar(proto, T, N, K, 3); vig = similar(proto, T, N, K); seen = similar(proto, Bool, N, K)
    Uc = similar(proto, SVector{2, T}, N); w = similar(proto, T, N)
    for k in 1:K
        Rt = SMatrix{3, 3, T, 9}(rotations[k])'
        F = view(rays, :, k); U = view(pixels, :, k); ok = view(seen, :, k)
        VgtkCore.pointwise!(d -> Rt * d, F, D)
        VgtkCore.pointwise!(fwd, U, F)
        VgtkCore.pointwise!(sees, ok, U, F)
        for (c, P) in enumerate(plane_maps)                               # unsaturated at every plane's taps
            any(j -> plane_maps[j] == P, 1:(c - 1)) && continue           # a plane at another's samples: its taps are read
            VgtkCore.pointwise!(inv(P), Uc, U)                            # the plane's own COLMAP pixels
            VgtkCore.image_sample!(w, saturated[k], Uc)                   # the tap converts the mask: no float copy of it
            ok .&= iszero.(w)
        end
        VgtkCore.pointwise!(_WhereSeen(vignetting, T), view(vig, :, k), ok, U, F; exec = VgtkCore.OnBackend(KA.get_backend(U)))
        for c in 1:3
            v = view(values, :, k, c)
            VgtkCore.pointwise!(inv(plane_maps[c]), Uc, U)                # the plane's samples sit where its map says
            VgtkCore.image_sample!(w, view(frames[k], :, :, c), Uc)
            v .= ifelse.(ok, w, T(NaN))
        end
    end
    return FrameSamples(D, pixels, rays, values, vig, seen)
end

"""
    fibonacci_directions(n) -> Vector{SVector{3, Float64}}
    fibonacci_directions(rotations, n; fov_deg) -> Vector{SVector{3, Float64}}

`n` unit directions spread uniformly over the sphere, the Fibonacci (golden
angle) spiral: direction `i` at height `1 − (2i − 1)/n` along the up axis (−y)
and azimuth `i·π(3 − √5)` — equal area per direction, no pole bias. With
`rotations`, the directions of the `n` that lie within `fov_deg` of some
frame's optical axis (`R_k·z`): the sphere's set restricted to the frames'
coverage, the input of [`sample_frames`](@ref).
"""
function fibonacci_directions(n::Integer)
    n > 0 || throw(ArgumentError("fibonacci_directions: n must be positive, got $n"))
    ga = π * (3 - sqrt(5.0))
    return map(1:n) do i
        h = 1 - (2i - 1) / n; ρ = sqrt(max(0.0, 1 - h^2)); sφ, cφ = sincos(i * ga)
        SVector(ρ * sφ, -h, ρ * cφ)
    end
end
function fibonacci_directions(rotations::AbstractVector, n::Integer; fov_deg::Real)
    axes = [SMatrix{3, 3, Float64, 9}(R)[:, 3] for R in rotations]
    c = cosd(fov_deg)
    return filter(d -> any(a -> dot(a, d) > c, axes), fibonacci_directions(n))
end
