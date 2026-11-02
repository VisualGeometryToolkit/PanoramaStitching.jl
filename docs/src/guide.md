# Getting started

PanoramaStitching turns a sweep of frames from a camera that only rotated into the rotation of every frame and an
equirectangular panorama. The camera is calibrated and fixed; its lens may have any field of view, so a fisheye imaging past
90° off axis works the same as a pinhole. All geometry is on unit bearings (VisualGeometryToolkitCore's `backproject`), never on
the `z = 1` plane, and every residual is measured in the camera's pixels.

The pipeline has three steps, one function each:

1. [`register_rotations`](@ref) — the rotation `R_k` of every frame (`d = R_k·f_k`, a bearing `f_k` of frame `k` to a world
   direction `d`; the first frame's camera frame is the common frame).
2. [`sample_cells`](@ref) — a near-equal-area grid of world cells, sampled in every frame that sees it.
3. [`render_panorama`](@ref) — the equirectangular image of per-frame cell values, composited in capture order.

## Installation

```julia
using Pkg
Pkg.add("PanoramaStitching")
```

## The inputs

- **Frames**: a `Vector` of `H × W × 3` linear (not gamma-encoded) images in capture order. They may be binned relative to the
  calibration: `raw_scale` is the number of camera pixels per frame pixel.
- **Camera**: any VisualGeometryToolkitCore camera with `project`, `backproject` and `backproject_with_jacobian`. A fisheye
  is best modelled as a `CentralGenericModel` over a lens model:

```julia
using VisualGeometryToolkitCore: CameraModel, CentralGenericModel, KannalaBrandt, px
using StaticArrays: SA

camera = CentralGenericModel(CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02)), (1000, 1000);
                             grid_size = (40, 40))
```

- **Image circle**: `circle = (cx, cy, r)` in camera pixels, the part of the sensor the lens illuminates. Features and samples
  outside it are ignored.

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
   robust (`bundle_loss`) pixel reprojection error with NLLSsolver.

```julia
import VisualGeometryToolkitCore as VgtkCore

reg = register_rotations(frames, camera; raw_scale = 2, circle, tone_percentile = 99.5, detector_margin_px = 20.0,
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

## 2. Sampling world cells

[`world_cells`](@ref) divides the sphere into latitude bands of `cell_deg` degrees, each split into longitude cells of nearly
equal area. [`sample_cells`](@ref) takes `sub × sub` jittered sample directions in every cell into each frame through the
camera and averages the (slightly blurred) frame there:

```julia
using Random: Xoshiro

saturated = [falses(size(f, 1), size(f, 2)) for f in frames]     # mask clipped pixels here
samples = sample_cells(frames, saturated, reg.rotations, camera; raw_scale = 2, circle, cell_deg = 2.0, sub = 4,
                       blur_px = 1.0, rim_margin_px = 12.0, fov_deg = 89.0, field_angle = :lens, rng = Xoshiro(1))
```

A cell counts in a frame (`samples.ok[k, i]`) only when the frame sees all of it: its centre within `fov_deg` of the optical
axis and every sample inside the image circle less `rim_margin_px`, unsaturated. The [`CellSamples`](@ref) also record each
cell's field angle per frame (`theta_model`), which is what a vignetting fit downstream needs.

## 3. Rendering the panorama

[`render_panorama`](@ref) composites per-frame cell values in capture order — each cell takes the last frame that saw it —
divided by that frame's gain (`gains` is `K × 3`, natural log; zeros for none):

```julia
K, N = size(samples.ok)
values = [samples.ok[k, i] ? samples.val[k, i, c] : NaN for k in 1:K, i in 1:N, c in 1:3]
image, extent, composite = render_panorama(samples.cells, values, samples.ok, zeros(K, 3))
```

`image` is the equirectangular raster (rows the elevation from the top, columns the azimuth, one pixel per cell, NaN where no
frame saw the cell) and `extent` its `(lon₀, lon₁, lat₀, lat₁)` in degrees. Corrections of the frames themselves, such as
dividing out a vignetting, are applied to `values` before rendering. To level the horizon, pass `up = up_frame(pitch, roll)`
(the first frame's pitch and roll against the true horizon, degrees).
