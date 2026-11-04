# Changelog

## Unreleased

The VisualGeometryToolkit 0.5 stack (Core 0.5, Estimation 0.6), and the sampling and the panorama on Core's warping machinery
(`image_warp`/`image_warp!`, `image_sample!`, `pointwise!`: KernelAbstractions kernels on GPU arrays, the same code on the
host).

### Breaking

- `circle` (of `register_rotations`, `detect_features`, `sample_frames`, `render_panorama`) is a Core `Circle` or `Ellipse` in
  camera pixels, or `WholePlane()` for a lens with no image circle, instead of a `(cx, cy, r)` tuple. Masks of binned frames
  are Core's `pixel_mask` of the circle scaled by `1/raw_scale`; a point is in it by Core's `∈` (the boundary included).
- A grid camera (`CentralGenericModel`) has no closed-form forward map: `register_rotations` and `refine_rotations` take its
  `SplineProjection` (VisualGeometryToolkitEstimation's `fit_camera(SplineProjection, grid, region; tolerance)`) as the
  keyword `projection`, and `sample_frames` and `render_panorama` take it as their camera. A grid without one is an
  `ArgumentError`. The private seeded projection helper is gone (Core's `project` no longer takes `init`).
- `render_panorama(camera, frames, rotations, view; vignetting, circle, raw_scale, rim_margin_px, fov_deg, saturated)`: the
  raster is a `PanoramaView` built by `panorama_view` — a Core panorama projection (`Equirectangular`, `Cylindrical`,
  `CylindricalEqualArea`, `Mercator`) over an `extent` at `deg_per_pixel` with an `up` frame, or a virtual camera (any Core
  `CameraModel`, aimed by a rotation) — instead of the `extent`, `deg_per_pixel` and `up` keywords. The field angle is measured
  from the camera frame's z axis. The render is Core's `image_warp`, and its masks are batches at the warp's own sensor points;
  a pixel next to a saturated one is now masked when the saturated pixel enters its bilinear blend.
- `render_panorama` and `render_cube_map` take no `saturated` keyword: a render shows the frames as they are, a saturated pixel
  with its clipped value (a window that clips in every frame comes out bright, not empty). Saturation masks measurements only:
  `sample_frames`, `fit_vignetting` and `score_seams`.
- Removed: `world_cells`, `sample_cells`, `CellSamples` and `cell_centres` (the binned sampling of the sphere). Use
  `sample_frames` at `fibonacci_directions`.

### New

- `fit_vignetting(space, camera, frames, rotations; …) -> FitResult`: the vignetting of a rotating camera from its frames
  alone, interpolation free — every pixel of every frame at its pixel centre is `L(R_k·f)·V(u)`, log L a cubic B-spline over
  an `AzimuthalChart` of the covered sphere with a Δ² or Δ³ penalty, log V a `RadialVignetting` or `TensorVignetting` spline,
  weighted by I², gauge mean log V = 0; the normal equations by atomic scatter-add on the frames' backend (KernelAbstractions,
  Atomix), solved by `ConjugateGradient(; tolerance, max_iter)` on the device or `SparseCholesky()` on the host
  (SparseArrays). The estimate is a `SplineVignetting`, callable as the render's
  `vignetting`; `log_vignetting`, `VignettingMetadata`.
- `fit_vignetting`'s required keyword `frame_gains::Bool` (no default): with `true` every frame has its own exposure gain,
  `I_k = g_k·L·V`, K − 1 more unknowns with the gauge `g₁ = 1`, returned as `VignettingMetadata.gains` (all 1 with `false`).
  A fixed exposure under changing light needs it: on the rooftop sweep the gains reached +18 % and V at the domain's edge
  came out 7 % too dark without them. `score_seams` and `render_panorama` take an optional `gains` (one per frame) and divide
  each frame by `g_k·V`; the example's configs state the key.
- `RotationBundleMetadata` (of `refine_rotations`, and so `register_rotations(…).bundle`) carries the bundle's `outcome` —
  `:converged`, `:nonfinite_start` (the start not finite: the retcode `:solve_failed`, the rotations the start's, and
  observations whose residual or Jacobian is not finite at the start, a start prediction outside the lens's domain),
  `:not_run` (no iteration) or the retcode — and `start_nonfinite`, that count. The example records both in rotations.json
  (its cache format v2) and summary.md.
- `refine_rotations` with a grid model: an observation predicted off the grid's imaged region now has a NaN residual, as its
  Jacobian already had (its `SplineProjection` extrapolates there). Before, such an iterate was accepted with a finite cost and
  NLLSsolver's Levenberg–Marquardt trial loop never returned on the NaN linear system it gave.
- `score_seams(camera, frames, rotations, view; …)`: per raster pixel measured by two frames or more, the median log ratio of
  the other frames to the shown one (saturated pixels excluded), and its rms and 1.4826·MAD.
- `sample_frames`, `render_panorama` (and so `score_seams`) move the camera's forward map to the frames' backend (Adapt): a
  grid's `SplineProjection` now samples GPU frames (its control grid was a host array inside a device kernel). Its device
  render still waits on Core: `image_warp`'s map wraps it in a `BoundedMap` and a `ComposedTransformation`, which have no
  Adapt rule yet.
- `examples/panorama_vignetting/`: the chain from RAW frames to corrected panoramas, two configs of record (the FX3 with the
  8 mm fisheye on a rooftop, with the Sigma 14 mm indoors).

- `sample_frames(camera, frames, rotations, directions; …) -> FrameSamples`: per direction and frame, the exact sub-pixel
  sensor point, the ray, the frame's bilinear value and the vignetting model evaluated there, and whether the frame sees it —
  the observations of a vignetting validation or a panorama vignetting fit.
- `fibonacci_directions(n)` and `fibonacci_directions(rotations, n; fov_deg)`: directions spread uniformly over the sphere,
  optionally restricted to the frames' coverage.
- `panorama_view`, `PanoramaView`, `render_cube_map` (the six 90° faces, `CUBE_FACES`, and their horizontal cross).
- `plane_offsets` (of `render_panorama` and `sample_frames`): the planes' sample positions in frame pixels — a 2 × 2 Bayer
  binning's R and B at `(±¼, ±¼)`, VisualGeometryToolkitFeatures' `load_raw(…; mode = :bin2).plane_offsets` — each plane
  read at the sensor point less its offset.
- The render's warp math is Float64 whatever the frames' element type: GPU and host renders agree to 1e-7 relative.
- `detect_features` resamples the descriptor's input with Core's `image_warp!` (the same values: a downscale never reaches the
  frame's edge).
