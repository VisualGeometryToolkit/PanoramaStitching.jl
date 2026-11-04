# Getting started

PanoramaStitching turns a sweep of frames from a camera that only rotated into the rotation of every frame and a panorama.
The camera is calibrated and fixed; its lens may have any field of view, so a fisheye imaging past 90° off axis works the same
as a pinhole. All geometry is on unit bearings (VisualGeometryToolkitCore's `backproject`), never on the `z = 1` plane, and
every residual is measured in the camera's pixels.

The pipeline has four steps:

1. [`register_rotations`](@ref) — the rotation `R_k` of every frame (`d = R_k·f_k`, a bearing `f_k` of frame `k` to a world
   direction `d`; the first frame's camera frame is the common frame).
2. [`sample_frames`](@ref) — the frames sampled at directions spread uniformly over the covered sphere
   ([`fibonacci_directions`](@ref)): per direction, every frame that sees it with its sensor point, its value and the
   vignetting there — the observations of a vignetting validation or fit.
3. [`render_panorama`](@ref) — the panorama into a view ([`panorama_view`](@ref): a cylindrical-family projection or a
   virtual camera) by the reverse warp of the frames, unblended; [`render_cube_map`](@ref) renders the six faces of a cube map.
4. [`fit_vignetting`](@ref) — the camera's vignetting from the frames alone (every pixel of every frame, at its own pixel
   centre, is the scene's radiance times V), and [`score_seams`](@ref), how well a vignetting makes overlapping frames agree.

Steps 2 and 3 run on VisualGeometryToolkitCore's warping machinery — `image_warp`/`image_warp!`, `image_sample!` and
`pointwise!`, KernelAbstractions kernels — on the backend that owns the frames: GPU arrays (`CuArray`) run on the device,
host arrays run the same code on the CPU.

## Installation

PanoramaStitching is registered in [CauRegistry](https://github.com/prittjam/CauRegistry.jl); the VisualGeometryToolkit
stack it is built on is in General. Add CauRegistry once (Julia 1.12 or later), then the package and the two packages the
examples also name:

```julia-repl
pkg> registry add General https://github.com/prittjam/CauRegistry.jl
pkg> add PanoramaStitching VisualGeometryToolkitCore StaticArrays
```

## The inputs

- **Frames**: a `Vector` of `H × W × 3` linear (not gamma-encoded) images in capture order. They may be binned relative to the
  calibration: `raw_scale` is the number of camera pixels per frame pixel.
- **Camera**: a VisualGeometryToolkitCore camera. A parametric `CameraModel` serves every step. A grid model
  (`CentralGenericModel`) back-projects in the registration but has no closed-form forward map: supply its
  `SplineProjection`, fitted by VisualGeometryToolkitEstimation's `fit_camera`, as `projection` to the registration and as
  the camera of the sampling and the render.

```julia
using VisualGeometryToolkitCore: CameraModel, CentralGenericModel, KannalaBrandt, SplineProjection, Circle, Point2, Rect, px,
    estimate
import VisualGeometryToolkitEstimation as VgtkEstimation
using StaticArrays: SA

model = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
camera = CentralGenericModel(model, (1000, 1000); grid_size = (40, 40))
circle = Circle(Point2(500.0, 500.0), 480.0)          # the image circle, camera pixels (WholePlane() for a lens with none)
projection = estimate(VgtkEstimation.fit_camera(SplineProjection, camera, Rect(0, 0, 1000, 1000) ∩ circle; tolerance = 1e-2))
```

- **Image circle**: `circle`, a Core `Circle` or `Ellipse` in camera pixels, the part of the sensor the lens illuminates, or
  `WholePlane()` for a lens that fills the sensor. Features and samples outside it are ignored.

## 1. Registering the rotations

[`register_rotations`](@ref) composes the stack's estimators and does no estimation of its own:

1. **Features** ([`PanoramaStitching.detect_features`](@ref)): ASIFT keypoints inside the image circle, described by
   VisualGeometryToolkitFeatures' DeDoDe/LoMa descriptor (`descriptor = :b`), at most `max_detections` per frame.
   `tone_percentile` sets the exposure the detector sees; `asift_tilts = [(1.0, 0.0)]` is the untilted view only.
2. **Pairs**: `pairs = :all` matches every frame pair; `pairs = :lag_overlap` only those at most `max_lag` apart in capture
   order, plus those whose averaged optical axes are within `overlap_deg`. Matches come from the LoMa matcher
   (`matcher = :b128`) and become bearing correspondences through the camera
   ([`PanoramaStitching.pair_bearings`](@ref)).
3. **Pairwise rotations**: VisualGeometryToolkitEstimation's `find_rotation` (RANSAC on bearings, its sampler seeded from
   `seed`) verifies each pair; a pair with fewer than `min_inliers` inliers is dropped.
4. **Rotation averaging**: the verified pairs form a [`ViewGraph`](@ref); `fit_rotation_averaging` with `averaging_loss`
   gives every frame's rotation.
5. **Tracks**: [`chain_tracks`](@ref) links the inlier matches into tracks across frames; inconsistent tracks (two keypoints
   of one frame) are dropped.
6. **Bundle**: [`refine_rotations`](@ref) refines every rotation but the first, and one direction per track, minimizing the
   robust (`bundle_loss`) pixel reprojection error through `projection` with NLLSsolver.

```julia
import VisualGeometryToolkitCore as VgtkCore

reg = register_rotations(frames, camera; projection, raw_scale = 2, circle, tone_percentile = 99.5, detector_margin_px = 20.0,
                         asift_tilts = [(1.0, 0.0)], max_detections = 1024, descriptor = :b, matcher = :b128, min_score = 0.0,
                         pairs = :all, max_lag = 1, overlap_deg = 120.0, min_inliers = 40, seed = 1,
                         max_trials = 10_000, min_trials = 100,
                         averaging_loss = VgtkCore.HuberShape(), bundle_loss = VgtkCore.CauchyShape(),
                         bundle_scale_multiple = 3.0, bundle_iterations = 3, bundle_max_iter = 100,
                         bundle_ftol = 1e-10, bundle_xtol = 1e-10, say = println)
```

Every setting is a keyword without a default, so a run states its whole configuration. The result, a
[`Registration`](@ref), holds the rotations, the view graph, per-pair statistics (`reg.pairs[(i, j)]`: matches, inliers, the rms
Sampson distance in pixels and the rotation angle), the tracks and the bundle's result (`reg.bundle.rms` is its rms
reprojection error in pixels). On the package's synthetic fisheye sweep (four frames 35° apart) the rotations come back within
0.02° and the bundle's rms is below half a pixel.

## 2. Sampling the frames at directions

[`fibonacci_directions`](@ref) spreads directions uniformly over the sphere (the golden-angle spiral: equal area per direction,
no pole bias), here restricted to those within `fov_deg` of some frame's axis. [`sample_frames`](@ref) takes each into every
frame (`f = R_kᵀ·d`), projects it to its exact sub-pixel sensor point `u` through the camera, samples the frame there
bilinearly and evaluates the vignetting model continuously at the same point, `vignetting(u, f)`:

```julia
saturated = [falses(size(f, 1), size(f, 2)) for f in frames]     # mask clipped pixels here
D = fibonacci_directions(reg.rotations, 200_000; fov_deg = 89.0)
S = sample_frames(projection, frames, reg.rotations, D; saturated, circle, raw_scale = 2, rim_margin_px = 12.0,
                  fov_deg = 89.0, vignetting = (u, f) -> 1.0)
```

A frame sees a direction (`S.seen[i, k]`) when the ray is within `fov_deg` of the camera's z axis, its sensor point lies inside
the image circle less `rim_margin_px` and 2 px inside the sensor, and no saturated pixel enters its value. For every direction
seen by several frames, `S.values[i, k, c] / S.vignetting[i, k]` must agree across `k`: the spread is the vignetting model's
error, and `I_k = L(d)·V(u_k)` is the equation of a panorama vignetting fit.

## 3. Rendering the panorama

[`panorama_view`](@ref) builds the raster and its pixel → ray map; [`render_panorama`](@ref) warps every frame into it (Core's
`image_warp`, one fused map `camera ∘ R_kᵀ ∘ rays` per frame), divides by the vignetting and keeps, at each pixel, the LAST
frame that sees it — no gain, no seam, no blend. It renders the frames as they are: a saturated pixel keeps its clipped value
(the masks are for measurements — the sampling, the fit and the seam score — not for pictures):

```julia
cyl = panorama_view(VgtkCore.Cylindrical(), (-120.0, 120.0, -40.0, 40.0), 0.05)   # columns longitude, rows tan(latitude)
pano = render_panorama(projection, frames, reg.rotations, cyl; vignetting = (u, f) -> 1.0, circle, raw_scale = 2,
                       rim_margin_px = 12.0, fov_deg = 89.0)
pano.image      # rows × columns × 3, NaN where no frame sees the pixel
pano.owner      # the frame each pixel holds
pano.coverage   # how many frames see each pixel
```

The cylindrical family — Core's `Equirectangular()` (rows the latitude), `Cylindrical()` (`tan φ`), `CylindricalEqualArea()`
(`sin φ`: every pixel subtends the same solid angle) and `Mercator()` (`asinh(tan φ)`, conformal) — shares the longitude
columns. An azimuthal view needs no projection of its own: it is a virtual camera, a Core `CameraModel` with the lens of the
view, aimed by a rotation:

```julia
using VisualGeometryToolkitCore: Lens, Perspective, Stereographic
using StaticArrays: SMatrix
rect = panorama_view(CameraModel(1000.0px, SA[960.0px, 540.0px]; lens = Lens(Perspective())), 1920, 1080)
down = SMatrix{3, 3}(1.0, 0, 0, 0, 0, -1, 0, 1, 0)          # its axis along +y (down): a 'little planet'
planet = panorama_view(CameraModel(400.0px, SA[1000.0px, 1000.0px]; lens = Lens(Stereographic())), 2000, 2000; rotation = down)
```

[`render_cube_map`](@ref) renders the six 90° faces and their horizontal cross. To level the horizon, supply
`up = up_frame(pitch, roll)` to `panorama_view` (the first frame's pitch and roll against the true horizon, degrees).

## 4. The vignetting from the frames

A rotating camera sees each scene direction at several sensor points, so its frames determine its vignetting up to one
scale. [`fit_vignetting`](@ref) uses every pixel of every frame inside the domain, read at its own pixel centre — no
interpolation of the frames: `log I_k(p) = log L(R_k·f(p)) + log V(u(p))`, the pixel's ray `f(p)` the camera's
back-projection (computed once: every frame shares the pixel grid), the scene `log L` a cubic B-spline over an
[`AzimuthalChart`](@ref) of the covered sphere with a difference penalty, and `log V` a cubic B-spline in the distance from a
centre ([`RadialVignetting`](@ref)) or over the sensor ([`TensorVignetting`](@ref)). The least squares is weighted by I²;
its gauge is mean log V = 0 over the domain. With `frame_gains = true` every frame also has its own exposure gain
(`I_k = g_k·L·V`, gauge `g₁ = 1`): a fixed exposure under changing light needs it, or V absorbs the frames' differences.
The normal equations are accumulated by atomic scatter-add on the frames' backend (GPU arrays on the device) and solved by
conjugate gradients there, or by a host Cholesky factorization:

```julia
space = RadialVignetting(SA[500.0, 500.0], 490.0, 16)          # centre, radius (camera pixels), controls
res = fit_vignetting(space, camera, frames, reg.rotations; saturated, circle, raw_scale = 2, rim_margin_px = 12.0,
                     fov_deg = 89.0, plane = 2, plane_offsets = nothing, chart_spacing_deg = 1.0, penalty_order = 3,
                     penalty_weight = 0.01, vignetting_penalty_weight = 1.0e-6, frame_gains = true,
                     solver = ConjugateGradient(; tolerance = 1.0e-9, max_iter = 40_000))   # or SparseCholesky()
V = VgtkCore.estimate(res)        # a SplineVignetting: V(u, f), the vignetting keyword of render_panorama
g = res.metadata.gains            # the frames' gains g_k (g₁ = 1): the gains keyword of score_seams and render_panorama
```

A sweep about one axis does not observe a 2-D V's tilt across it (a field that varies only with the ray's height is unchanged
by a pure yaw, so L absorbs it): fit the tensor space only to frames with pitch and roll diversity. [`score_seams`](@ref)
measures the result where it matters, at the seams of the unblended panorama: per raster pixel measured by two frames or more,
the median log ratio of the other frames to the shown one (saturated pixels excluded), and its rms and 1.4826·MAD.
