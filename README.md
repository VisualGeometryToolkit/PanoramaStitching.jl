# PanoramaStitching.jl

[![CI](https://github.com/VisualGeometryToolkit/PanoramaStitching.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/VisualGeometryToolkit/PanoramaStitching.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/PanoramaStitching.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/VisualGeometryToolkit/PanoramaStitching.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://visualgeometrytoolkit.github.io/PanoramaStitching.jl/dev/)

The panorama of a camera that only rotated — a sweep of a distant scene — through a calibrated lens of any field of view,
fisheyes past 90° included, on the [VisualGeometryToolkit](https://visualgeometrytoolkit.github.io) stack:

- `register_rotations(frames, camera; …)` — the rotations of the frames: ASIFT keypoints described by DeDoDe and matched by
  LoMa (VisualGeometryToolkitFeatures), the rotation of every frame pair by RANSAC on bearings
  (VisualGeometryToolkitEstimation's `find_rotation`), the view graph's robust rotation averaging
  (`fit_rotation_averaging`), the tracks, and the bundle of the rotations and the track directions (NLLSsolver, the pixel
  reprojection error through the fixed camera).
- `sample_cells(frames, saturated, rotations, camera; …)` — a near-equal-area grid of world cells sampled in every frame.
- `render_panorama(cells, values, ok, gains; up)` — the equirectangular panorama of per-frame cell values, composited in capture
  order.

## Installation

```julia
using Pkg
Pkg.add("PanoramaStitching")
```

## Usage

The frames are `H × W × 3` linear images in capture order; the camera is any calibrated VisualGeometryToolkitCore camera. Here a
Kannala–Brandt fisheye of 300 px focal length on a 1000 × 1000 sensor, imaging out to 100° off axis, with the frames binned
2 × 2 (`raw_scale = 2` camera pixels per frame pixel):

```julia
using PanoramaStitching
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: CameraModel, CentralGenericModel, KannalaBrandt, px
using StaticArrays: SA
using Random: Xoshiro

camera = CentralGenericModel(CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02)), (1000, 1000);
                             grid_size = (40, 40))
circle = (500.0, 500.0, 480.0)   # the image circle (cx, cy, r), camera pixels
frames = my_frames               # your Vector of 500 × 500 × 3 Float32 linear images, in capture order

# 1. the rotations: features, pairwise RANSAC, rotation averaging, tracks and the bundle
reg = register_rotations(frames, camera; raw_scale = 2, circle, tone_percentile = 99.5, detector_margin_px = 20.0,
                         asift_tilts = [(1.0, 0.0)], max_detections = 1024, descriptor = :b, matcher = :b128, min_score = 0.0,
                         pairs = :all, max_lag = 1, overlap_deg = 120.0, min_inliers = 40, seed = 1,
                         max_trials = 10_000, min_trials = 100,
                         averaging_loss = VgtkCore.HuberShape(), bundle_loss = VgtkCore.CauchyShape(),
                         bundle_scale_multiple = 3.0, bundle_iterations = 3, bundle_max_iter = 100,
                         bundle_ftol = 1e-10, bundle_xtol = 1e-10, say = println)
reg.rotations    # one rotation per frame (d = R_k·f_k; the first frame's camera frame is the common frame)
reg.bundle.rms   # the bundle's rms reprojection error, pixels

# 2. a 2° grid of world cells, sampled in every frame
saturated = [falses(size(f, 1), size(f, 2)) for f in frames]
samples = sample_cells(frames, saturated, reg.rotations, camera; raw_scale = 2, circle, cell_deg = 2.0, sub = 4,
                       blur_px = 1.0, rim_margin_px = 12.0, fov_deg = 89.0, field_angle = :lens, rng = Xoshiro(1))

# 3. the equirectangular panorama (zero log-gains: no exposure correction)
K, N = size(samples.ok)
values = [samples.ok[k, i] ? samples.val[k, i, c] : NaN for k in 1:K, i in 1:N, c in 1:3]
image, extent, _ = render_panorama(samples.cells, values, samples.ok, zeros(K, 3))
```

`image` is the panorama (rows the elevation from the top, columns the azimuth, NaN where no frame saw the cell) and `extent`
its `(lon₀, lon₁, lat₀, lat₁)` in degrees. The [documentation](https://visualgeometrytoolkit.github.io/PanoramaStitching.jl/dev/)
explains each step and its settings.

MIT licence.
