# =============================================================================
# The frames sampled at directions — the observations of the vignetting validation and fit
# =============================================================================
#
# The common frame is the first frame's camera frame (x right, y down, z forward); a direction
# `d` of it is seen by frame k as the ray `f = R_kᵀ·d` (`rotations[k]` with `d = R_k·f`), imaged
# at the sensor point `u = π(f)` of the calibrated camera's forward map (a parametric camera's
# `imaging_transform`, a grid camera's `SplineProjection`). Every per-direction step is a batch
# on the backend that owns the arrays (Core's `pointwise!` and `image_sample!`, KernelAbstractions
# kernels on a GPU, the same code on the host): the rays, the sensor points, the masks, the
# frame's value and the vignetting at the point.
# =============================================================================

"The unit direction of (latitude, longitude) in degrees: Core's `Equirectangular` back-projection (up = −y, longitude 0 = +z, +90° = +x)."
world_direction(lat, lon) = VgtkCore.backproject(VgtkCore.Equirectangular(), SVector(deg2rad(lon), deg2rad(lat)))

# The camera's forward map ray → camera pixel as a Transformation (a kernel-lawful value on a device): a parametric camera's
# `imaging_transform`, a `SplineProjection` itself. A grid model has no closed-form forward map (its projection is a pixel-space
# solve): the caller fits one, Estimation's `fit_camera(SplineProjection, grid, region; tolerance)`.
_forward_map(m::VgtkCore.CameraModel) = VgtkCore.imaging_transform(m)
_forward_map(b::VgtkCore.BoundedCamera) = _forward_map(VgtkCore.model(b))
_forward_map(sp::VgtkCore.SplineProjection) = sp
_forward_map(c) = throw(ArgumentError("PanoramaStitching: $(typeof(c).name.name) has no closed-form forward map; supply its " *
                                      "SplineProjection, VisualGeometryToolkitEstimation.fit_camera(SplineProjection, camera, region; tolerance)"))

# The image circle less a margin (camera pixels): a `Circle` or an `Ellipse` shrunk by `m` along its axes, the whole plane kept.
_rim(c::VgtkCore.Circle, m) = VgtkCore.Circle(c.center, c.r - m)
_rim(e::VgtkCore.Ellipse, m) = VgtkCore.Ellipse(e.center, e.a - m, e.b - m, e.θ)
_rim(w::VgtkCore.WholePlane, m) = w

# The image circle in the pixels of a frame binned by `s` (a COLMAP coordinate scales by exactly 1/s).
_binned(c::VgtkCore.Circle, s) = VgtkCore.Circle(c.center / s, c.r / s)
_binned(e::VgtkCore.Ellipse, s) = VgtkCore.Ellipse(e.center / s, e.a / s, e.b / s, e.θ)
_binned(w::VgtkCore.WholePlane, s) = w

# Where a frame's sensor point counts (camera pixels): at least 2 px inside the W × H sensor and inside the circle less the rim
# margin.
_frame_region(circle, W, H, rim_margin_px) = VgtkCore.Rect(2.0, 2.0, W - 4.0, H - 4.0) ∩ _rim(circle, rim_margin_px)

# Whether a frame sees a direction: its sensor point `u` is finite and in the region, its ray `f` within the field angle of the
# camera frame's z axis (`f₃ > cos θ_max·‖f‖`), and its saturation sample `w` (the saturated mask's bilinear sample at u) zero —
# no saturated pixel among the four the value blends.
struct _Sees{R, T}
    region::R
    cosmax::T
end
@inline (s::_Sees)(u, f, w) =
    all(isfinite, u) && VgtkCore.Point2(u[1], u[2]) ∈ s.region && f[3] > s.cosmax * LinearAlgebra.norm(f) && iszero(w)

# The vignetting model where the frame sees the point, NaN elsewhere: the model is evaluated only at seen points (a grid's
# back-projection, say, has no ray off its domain).
struct _Vignetting{F, T}
    model::F
end
_Vignetting(model::F, ::Type{T}) where {F, T} = _Vignetting{F, T}(model)
@inline (v::_Vignetting{F, T})(ok, u, f) where {F, T} = ok ? T(v.model(u, f)) : T(NaN)

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
    sample_frames(camera, frames, rotations, directions; saturated, circle, raw_scale, rim_margin_px, fov_deg, vignetting)
        -> FrameSamples

The frames sampled at `directions` (unit 3-vectors of the common frame, any
array): for every frame `k` and direction `d`, the ray `f = R_kᵀ·d`
(`rotations[k]` with `d = R_k·f`), its exact sub-pixel sensor point
`u = π(f)` through the calibrated `camera` (a `CameraModel` through its
`imaging_transform`, or a `SplineProjection` — a grid model's forward map,
`fit_camera(SplineProjection, grid, region; tolerance)`), the frame's value
there by bilinear sampling (Core's `image_sample!`, no smoothing) and the
vignetting model evaluated continuously there, `vignetting(u, f)` (a model
defined on the sensor reads `u`, one defined on the ray's direction reads
`f`). These are the observations of a vignetting validation (`I_k / V_k` must
agree across the frames that see a direction: their spread is the model's
error) and of a panorama vignetting fit (`I_k = L(d)·V(u_k)`).

A frame sees a direction by [`render_panorama`](@ref)'s masks: the ray within
`fov_deg` of the camera frame's z axis, the sensor point inside `circle` (a
Core `Circle` or `Ellipse` in camera pixels, or `WholePlane()` when the lens
has no image circle) less `rim_margin_px` and at least 2 px inside the sensor,
and no `saturated` pixel among the four its value blends. `frames` are
`H × W × 3` linear images with `raw_scale` camera pixels per frame pixel,
`saturated` their `H × W` masks.

Every step is a batch on the backend that owns `frames` and `directions`
(Core's `pointwise!`, `image_sample!`: one kernel each on a GPU array, the
same code on the host), so for device arrays `vignetting` must be
kernel-lawful (isbits, no allocation, no throw). The precision is the
directions' element type.
"""
function sample_frames(camera, frames::AbstractVector, rotations::AbstractVector, directions::AbstractArray{<:StaticVector{3}};
                       saturated::AbstractVector, circle, raw_scale::Integer, rim_margin_px::Real, fov_deg::Real, vignetting)
    K = length(frames)
    length(rotations) == K && length(saturated) == K || throw(DimensionMismatch("sample_frames: frames, rotations and saturated disagree"))
    Hf, Wf, C = size(frames[1])
    C == 3 || throw(DimensionMismatch("sample_frames: frames are H × W × 3, got $(size(frames[1]))"))
    T = eltype(eltype(directions)); s = raw_scale
    fwd = _forward_map(camera)
    sees = _Sees(_frame_region(circle, Wf * s, Hf * s, rim_margin_px), T(cosd(fov_deg)))
    N = length(directions); D = reshape(directions, N)
    proto = frames[1]
    pixels = similar(proto, SVector{2, T}, N, K); rays = similar(proto, SVector{3, T}, N, K)
    values = similar(proto, T, N, K, 3); vig = similar(proto, T, N, K); seen = similar(proto, Bool, N, K)
    Uf = similar(proto, SVector{2, T}, N); w = similar(proto, T, N)
    for k in 1:K
        Rt = SMatrix{3, 3, T, 9}(rotations[k])'
        F = view(rays, :, k); U = view(pixels, :, k); ok = view(seen, :, k)
        VgtkCore.pointwise!(d -> Rt * d, F, D)
        VgtkCore.pointwise!(fwd, U, F)
        Uf .= U ./ T(s)                                                   # the frame's COLMAP pixels
        VgtkCore.image_sample!(w, T.(saturated[k]), Uf)
        VgtkCore.pointwise!(sees, ok, U, F, w)
        VgtkCore.pointwise!(_Vignetting(vignetting, T), view(vig, :, k), ok, U, F)
        for c in 1:3
            v = view(values, :, k, c)
            VgtkCore.image_sample!(w, view(frames[k], :, :, c), Uf)
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
