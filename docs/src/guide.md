# Getting started

PanoramaStitching turns a sweep of frames from a camera that only rotated into the rotation of every frame and a panorama.
The registration holds the camera's calibration fixed; when the calibration is not known, the same frames refine it with the
rotations, with no target ([Autocalibration](#Autocalibration) below). Its lens may have any field of view, so a fisheye imaging
past 90° off axis works the same as a pinhole. All geometry is on unit bearings (VisualGeometryToolkitCore's `backproject`), never on the `z = 1` plane, and
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

This page continues from the [first example](index.md#A-first-example): its `camera`, `frames` and `reg` are
used below, and every block runs after it, in order, as the test suite runs them (the installation is the same).

## The inputs

- **Frames**: a `Vector` of `H × W × 3` linear (not gamma-encoded) images in capture order. They may be binned relative to the
  calibration: `frame_map` is their map onto the camera's pixels (a Core `SimilarityMap`; a RAW's `:bin2` frames',
  VisualGeometryToolkitFeatures' `raw_frame_map(L)`), and `plane_maps` each colour plane's (`bayer_plane_map(L, c)`: a
  Bayer binning's R and B sit ¼ px off the cell centre; by default every plane's is the frame's).
- **Camera**: a VisualGeometryToolkitCore `BoundedCamera` (a camera model and its sensor). A parametric `CameraModel` serves
  every step. A grid lens is a `CameraModel` with a `SplineLens`; it projects through its forward spline, fitted by
  VisualGeometryToolkitEstimation's `fit_camera`. The example's camera as a 40 × 40 grid, with its forward spline:

```julia
# Illustrative: VisualGeometryToolkitEstimation's fit_camera, one more package (pkg> add VisualGeometryToolkitEstimation)
using VisualGeometryToolkitCore: SplineLens, imaged_region, with_model, estimate
import VisualGeometryToolkitEstimation as VgtkEstimation

# the same sensor and lens coverage, the model a 40 × 40 SplineLens with its forward spline
grid = with_model(camera, estimate(VgtkEstimation.fit_camera(SplineLens, camera, imaged_region(camera); tolerance = 1e-2,
                                                             grid_size = (40, 40))))
# register_rotations(frames, grid; …), sample_frames(grid, …), render_panorama(grid, …)
```

- **Imaged region**: the camera is a VisualGeometryToolkitCore `BoundedCamera` (the model and its sensor), and its
  `imaged_region` — the sensor ∩ its lens coverage (the image circle, a `Circle` or `Ellipse` in camera pixels:
  `BoundedCamera(model, extent; lens_coverage = Circle(…))`; omitted, the whole sensor) ∩ its model's valid domain — is the
  part of the sensor every step reads. Features and samples outside it are ignored; there is no separate circle argument.

## 1. Registering the rotations

[`register_rotations`](@ref) composes the stack's estimators and does no estimation of its own:

1. **Features** ([`PanoramaStitching.detect_features`](@ref)): ASIFT keypoints inside the camera's imaged region less
   `detector_margin_px`, described by
   VisualGeometryToolkitFeatures' DeDoDe/LoMa descriptor (`descriptor = :b`), at most `max_detections` per frame.
   `tone_percentile` sets the exposure the detector sees; `asift_tilts = [(1.0, 0.0)]` is the untilted view only.
2. **Pairs**: `pairs = :all` matches every frame pair; `pairs = :lag_overlap` only those at most `max_lag` apart in capture
   order, plus those whose averaged optical axes are within `overlap_deg`. Matches come from the LoMa matcher
   (`matcher = :b128`) and become bearing correspondences through the camera
   ([`PanoramaStitching.pair_bearings`](@ref)).
3. **Pairwise rotations**: VisualGeometryToolkitEstimation's `find_rotation` (RANSAC on bearings, its sampler seeded from
   `seed`) verifies each pair; a pair with fewer than `min_inliers` inliers is dropped.
4. **Rotation averaging**: the verified pairs form VisualGeometryToolkitEstimation's `ViewGraph` (each edge the pair's rotation, weighted by its
   inlier count); Estimation's `fit_rotation_averaging` of it with `averaging_loss`
   gives every frame's rotation.
5. **Tracks**: [`chain_tracks`](@ref) links the inlier matches into tracks across frames (the components of the match graph,
   Estimation's `ViewGraph` on the observations); inconsistent tracks (two keypoints
   of one frame) are dropped.
6. **Bundle**: [`refine_rotations`](@ref) refines every rotation but the first, and one direction per track, minimizing the
   robust pixel reprojection error through the camera's forward map with NLLSsolver: the loss `bundle_loss` at the width
   `bundle_efficiency_constant`·σ̂, σ̂ the noise scale of the verified pairs pooled over the kept ones (`reg.bundle_sigma`,
   pixels), in one LM run. The bundle refines the rotations and the directions only: the camera's calibration is always
   held, and it takes no keyword that would free a camera parameter. Refining the camera too is a refine of its own,
   [`refine_camera_and_rotations`](@ref) ([Autocalibration](#Autocalibration) below).

Steps 1–3 are [`verify_pairs`](@ref), whose [`VerifiedPairs`](@ref) hold everything the rest reads;
`register_rotations(pairs::VerifiedPairs, camera; …)` runs steps 4–6 from them, so a caller that keeps the verified pairs
(the example caches them in `verified_pairs.json`) reruns the averaging and the bundle alone.

The example's call states every setting: each is a keyword without a default, so a run states its whole configuration
(`say = println` prints one line per frame, pair and step). The result, a [`Registration`](@ref), holds the rotations, the
view graph, per-pair statistics, the tracks and the bundle's result:

```julia
import VisualGeometryToolkitCore as VgtkCore

reg.pairs[(1, 2)]               # matches, inliers, rms_px (the inliers' rms Sampson distance) and rotation_deg
length(reg.tracks), reg.dropped # the tracks, and the inconsistent ones dropped
reg.bundle.rms                  # the bundle's rms reprojection error, pixels
reg.bundle.metadata.outcome     # how the bundle ended (RotationBundleMetadata)
```

On the example's sweep the rotations come back well within 0.05° and the bundle's rms is below a pixel.

### Autocalibration

A camera that only rotates determines its own calibration from its frames, with no target (Hartley 1994; Agapito, Hayman
and Reid 2001): the observations of one track must back-project to one direction in every frame that sees it, which
constrains the camera's parameters as well as the rotations. [`refine_camera_and_rotations`](@ref) is that refine. From a
start camera (a nominal lens, or an earlier calibration of the same one), start rotations and the tracks, it refines the
camera's parameters jointly with every rotation and one direction per track. Its one gauge is the first frame's rotation,
held exactly.

It is a refine of the stack's robust-estimation contract, so it never estimates a scale of its own:

- `optimization_strategy` is the robust strategy: a `Triggs` strategy on a loss shape (its loss as NLLSsolver's robustifier at
  one width; the default `Triggs{WelschShape}` at the shape's efficiency constant). Another strategy is an `ArgumentError`.
- A robust strategy needs exactly one of `scale = σ` (the noise scale, pixels) or `width = w` (the loss width). Here the
  scale is the registration's pooled noise scale. Neither, or both, is an `ArgumentError`.
- `fixed` names the camera parameters to hold, by name or by group (`:k` for every lens coefficient; VisualGeometryToolkitCore's
  `free_parameter_names`). The default `()` frees every parameter but the skew, which is held by construction.

The start must be close: like every refine, it ends in the local minimum its start leads to. Only a parametric camera can
be refined: a `SplineLens` camera is an `ArgumentError`. The tracks come from a registration run with the start camera, so
rerun [`register_rotations`](@ref) with the refined camera to verify the pairs through it.

```julia
# Illustrative: VisualGeometryToolkitCore's Triggs strategy
using VisualGeometryToolkitCore: Triggs

# the tracks as frame => pixel observations, and the registration's rotations as the start, keyed by frame
tracks = [[k => reg.features[k].pixels[kp] for (k, kp) in t] for t in reg.tracks]
start = Dict(enumerate(reg.rotations))
autocal = refine_camera_and_rotations(camera, start, tracks; optimization_strategy = Triggs{CauchyShape}(efficiency_constant = 2.0),
                                      scale = reg.bundle_sigma, fixed = (:aspect,), max_iter = 100, ftol = 1e-10,
                                      xtol = 1e-10)
```

## 2. Sampling the frames at directions

[`fibonacci_directions`](@ref) spreads directions uniformly over the sphere (the golden-angle spiral: equal area per direction,
no pole bias), here restricted to those within `fov_deg` of some frame's axis. [`sample_frames`](@ref) takes each into every
frame (`f = R_kᵀ·d`), projects it to its exact sub-pixel sensor point `u` through the camera, samples the frame there
bilinearly and evaluates the vignetting model continuously at the same point, `vignetting(u, f)`:

```julia
saturated = [falses(size(f, 1), size(f, 2)) for f in frames]     # mask clipped pixels here
D = fibonacci_directions(reg.rotations, 200_000; fov_deg = 89.0)
S = sample_frames(camera, frames, reg.rotations, D; saturated, frame_map, rim_margin_px = 12.0, fov_deg = 89.0,
                  vignetting = (u, f) -> 1.0)
```

A frame sees a direction (`S.seen[i, k]`) when the ray is within `fov_deg` of the camera's z axis, its sensor point lies inside
the camera's imaged region less `rim_margin_px` at its lens coverage's edge and 2 px inside the frame, and no saturated pixel enters its value. For every direction
seen by several frames, `S.values[i, k, c] / S.vignetting[i, k]` must agree across `k`: the spread is the vignetting model's
error, and `I_k = L(d)·V(u_k)` is the equation of a panorama vignetting fit.

## 3. Rendering the panorama

[`panorama_view`](@ref) builds the raster and its pixel → ray map; [`render_panorama`](@ref) warps every frame into it (Core's
`image_warp`, one fused map `camera ∘ R_kᵀ ∘ rays` per frame), divides by the vignetting and keeps, at each pixel, the LAST
frame that sees it — no gain, no seam, no blend. It renders the frames as they are: a saturated pixel keeps its clipped value
(the masks are for measurements — the sampling, the fit and the seam score — not for pictures):

```julia
cyl = panorama_view(VgtkCore.Cylindrical(), (-120.0, 120.0, -40.0, 40.0), 0.25)  # columns longitude, rows tan(latitude)
pano = render_panorama(camera, frames, reg.rotations, cyl; vignetting = (u, f) -> 1.0, frame_map, rim_margin_px = 12.0,
                       fov_deg = 89.0)
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
# a 1920 × 1080 rectilinear view, and a 'little planet': a stereographic view looking down
rect = panorama_view(CameraModel(1000.0px, SA[960.0px, 540.0px]; lens = Lens(Perspective())), 1920, 1080)
down = SMatrix{3, 3}(1.0, 0, 0, 0, 0, -1, 0, 1, 0)          # its axis along +y (down)
planet = panorama_view(CameraModel(400.0px, SA[1000.0px, 1000.0px]; lens = Lens(Stereographic())), 2000, 2000; rotation = down)
```

[`render_cube_map`](@ref) renders the six 90° faces and their horizontal cross. To level the horizon, supply
`up = up_frame(pitch, roll)` to `panorama_view` (the first frame's pitch and roll against the true horizon, degrees).

## 4. The vignetting from the frames

A rotating camera sees each scene direction at several sensor points, so its frames determine its vignetting up to one
scale. [`fit_vignetting`](@ref) uses every pixel of every frame inside the domain, read at its own pixel centre — no
interpolation of the frames: `log I_k(p) = log L(R_k·f(p)) + log V(u(p))`, the pixel's ray `f(p)` the camera's
back-projection (computed once: every frame shares the pixel grid), the scene `log L` a cubic B-spline over an
[`AzimuthalChart`](@ref) of the covered sphere with a difference penalty, and V VisualGeometryToolkitCore's
`LogVignetting`: V = exp(s), `s` a B-spline of log V in the distance from a centre (`RadialVignetting`) or over the sensor
(`TensorProductVignetting`). The model is linear in log L, log V and the log gains, so the fit is one weighted linear least squares;
its gauge is mean log V = 0 over the domain. Each row is the implicit constraint of its measurement (sensor point, log I),
weighted by its whitening by the measurement noise that `measurement_noise` chooses: [`IntensityNoise`](@ref) (the default:
noise in the value only, read noise of constant σ, weight I²) or [`IntensityAndPositionNoise`](@ref) (noise in the value and
the pixel's sensor position, the sensor point's from the registration's residual scale, at the I² solve's model, so a row
where the scene's log I changes fast across the sensor counts less). `gain_correction = IntensityGains()` adds an exposure
gain per frame shared by every colour plane, `I_k = g_k·L·V` (gauge `g₁ = 1`), and [`ChannelGains`](@ref)`()` one per
frame and colour plane, `I_kc = g_kc·L_c·V` (a white
balance or a light colour that drifts too; V stays one for all planes): a fixed exposure under changing light needs them, or V
absorbs the frames' differences. `nothing` takes the frames at one exposure. `free_directions` says how V's control points
may move: a radial profile's pole-regular subspace (smooth at its centre), every control point of a tensor surface, or none
(`zeros(n, 0)`) to hold a given V fixed, a calibration's — the same solve is then the frames' gain compensation.
The normal equations are accumulated by atomic scatter-add on the frames' backend (GPU arrays on the device) and solved by
conjugate gradients there, or by a host Cholesky factorization:

```julia
s = VgtkCore.RadialVignetting(16, SA[500.0, 500.0], 490.0)     # controls, centre, radius (camera pixels)
v0 = VgtkCore.LogVignetting(VgtkCore.with_control_points(s, zeros(16)))   # log V ≡ 0
res = fit_vignetting(v0, camera, frames, reg.rotations; free_directions = VgtkCore.pole_regular_basis(v0),
                     gain_correction = ChannelGains(), saturated, frame_map, rim_margin_px = 12.0, fov_deg = 89.0, plane = 2,
                     chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01,
                     vignetting_penalty_weight = 1.0e-6,
                     solver = ConjugateGradient(; tolerance = 1.0e-9, max_iter = 40_000))   # or SparseCholesky()
# or measurement_noise = IntensityAndPositionNoise(; point_covariance = VgtkCore.CovMat{2}(σ = σ_u)): σ_u the registration's
# pooled noise scale reg.bundle_sigma (the bundle's width over its efficiency constant), px
V = VgtkCore.estimate(res)        # the fitted Core model: the vignetting keyword of render_panorama
g = res.metadata.gains            # the frames' exposure gains, K × 3: g_kc per frame and colour plane (g_1c = 1)
```

A sweep about one axis does not observe a 2-D V's tilt across it (a field that varies only with the ray's height is unchanged
by a pure yaw, so L absorbs it): fit the tensor space only to frames with pitch and roll diversity. [`score_seams`](@ref)
measures the result where it matters, at the seams of the unblended panorama: per raster pixel measured by two frames or more,
the median log ratio of the other frames to the shown one (saturated pixels excluded), and its rms and 1.4826·MAD:

```julia
seams = score_seams(camera, frames, reg.rotations, cyl; vignetting = V, gains = g, saturated, frame_map,
                    rim_margin_px = 12.0, fov_deg = 89.0, plane = 2, chunk = 200_000)
seams.rms, seams.mad              # the spread of the log ratio over the n pixels two frames or more measure
corrected = render_panorama(camera, frames, reg.rotations, cyl; vignetting = V, gains = g, frame_map,
                            rim_margin_px = 12.0, fov_deg = 89.0)   # the panorama with V and the frames' gains divided out
```

The example's frames carry no vignetting and one exposure, so here V comes back close to 1 and every gain close to 1.
