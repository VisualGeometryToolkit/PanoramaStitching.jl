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
- `sample_frames(camera, frames, rotations, directions; …)` — the frames sampled at directions of the sphere
  (`fibonacci_directions`): per direction, every frame that sees it, its sensor point, value and vignetting.
- `render_panorama(camera, frames, rotations, view; …)` — the panorama in a `panorama_view` (equirectangular, cylindrical,
  equal-area, Mercator, or a virtual camera), unblended; `render_cube_map` for a cube map.

The sampling and the render run on VisualGeometryToolkitCore's warping machinery (KernelAbstractions): GPU arrays in, the
device does the work; host arrays, the same code on the CPU.

## Installation

```julia
using Pkg
Pkg.add("PanoramaStitching")
```

## Usage

The frames are `H × W × 3` linear images in capture order; the camera is a calibrated VisualGeometryToolkitCore camera. Here a
Kannala–Brandt fisheye of 300 px focal length on a 1000 × 1000 sensor, imaging out to 100° off axis, with the frames binned
2 × 2 (`raw_scale = 2` camera pixels per frame pixel):

```julia
using PanoramaStitching
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, Circle, Point2, Cylindrical, px
using StaticArrays: SA

camera = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
circle = Circle(Point2(500.0, 500.0), 480.0)   # the image circle, camera pixels (WholePlane() for a lens with none)
frames = my_frames                             # your Vector of 500 × 500 × 3 Float32 linear images, in capture order

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

saturated = [falses(size(f, 1), size(f, 2)) for f in frames]
kw = (; saturated, circle, raw_scale = 2, rim_margin_px = 12.0, fov_deg = 89.0, vignetting = (u, f) -> 1.0)

# 2. the frames sampled at directions spread uniformly over the covered sphere
S = sample_frames(camera, frames, reg.rotations, fibonacci_directions(reg.rotations, 100_000; fov_deg = 89.0); kw...)

# 3. a cylindrical panorama, 0.1° per pixel
pano = render_panorama(camera, frames, reg.rotations, panorama_view(Cylindrical(), (-90.0, 90.0, -45.0, 45.0), 0.1); kw...)
```

`pano.image` is the panorama (rows from the top, columns the longitude, NaN where no frame sees the pixel), `pano.owner` the
frame each pixel holds and `pano.coverage` how many frames see it. The
[documentation](https://visualgeometrytoolkit.github.io/PanoramaStitching.jl/dev/) explains each step and its settings.

MIT licence.
