# Changelog

## 0.3.0

### Breaking

- The vignetting model is VisualGeometryToolkitCore's (0.5.5): `RadialVignetting`, `TensorVignetting` and `SplineVignetting`
  and `log_vignetting` are gone from PanoramaStitching. `fit_vignetting(vignetting, camera, frames, rotations; …)` takes a
  Core `LogVignetting` (V = exp(s), `s` a `RadialVignetting` or `TensorVignetting` spline of log V: the model of earlier
  releases, linear in log V) as V's model and start, and its estimate is the fitted Core model (callable `v(u)`, isbits: the
  `vignetting` keyword of `render_panorama`, `sample_frames` and `score_seams` on the host or a device, which still take any
  `vignetting(u, f)` as well). The fit is the one linear solve in log V of earlier releases. A V held fixed is any
  vignetting, a V-form `RadialVignetting` or `TensorVignetting` included. The model's knots are clamped (Core's), on the
  same breakpoints as before (a radial profile on `[0, radius]`, a tensor surface over the box). `VignettingMetadata` gains
  `weights` (the fit's) and `outer_iterations` and `log_change` (the solves, and the model's change in the last).
- The new required `free_directions` (`n_control_points × m`): how V's control points move. Core's `pole_regular_basis` for
  a radial profile (V′(0) = 0: smooth at its centre, which the earlier log-V profile did not impose), the identity for a
  tensor surface, `zeros(n, 0)` to hold a given V fixed — a calibration's — when the same solve is the frames' gain
  compensation.
- `frame_gains` is `gain_correction` (required): `nothing`, `FrameGains()` (one gain per frame, the release before's `true`)
  or `ChannelGains()` (new: one per frame and colour plane, `I_kc = g_kc·L_c·V`, the gauge `g_1c = 1` per plane: a frame's
  white balance or the light's colour drifting too). V stays one, shared by the planes: fitted (or held fixed) on `plane`
  with its gains, then each other plane's gains solved under that V. Both V modes take every gain mode. TOML: the tags of
  `PanoramaStitching.GAIN_CORRECTIONS` (a VisualGeometryToolkitCore `TagRegistry`: `"none"`, `"frame"`, `"channel"`; the
  example's `vignetting.gain_correction`, both configs `"channel"`). `VignettingMetadata.gains` is `K × 3`.
- `score_seams` and `render_panorama` keep `gains`, now the `K × 3` matrix of `metadata.gains` (one per frame and plane;
  `nothing` for one exposure): a vector is refused (`DimensionMismatch`).
- `fit_vignetting` back-projects the pixels through Core's batched `backproject` (a `CameraModel`, a `CentralGenericModel`
  or a `BoundedCamera` of one). `score_seams`'s per-direction ratio is a `pointwise!` functor of Core's.

### New

- `fit_vignetting`'s `weights` keyword: `IntensityWeights()` (the default, I²: every earlier call and its numbers stand)
  or `SampsonWeights(; point_scale | point_covariance, log_intensity_scale = 0.01, solves = 1)`. A row is the implicit
  constraint `g = log I − log L(R_k·f(u)) − log V(u) − γ_k` on the measurement `(u, log I)`, through VisualGeometryToolkitCore's
  `constraint_with_jacobians` protocol, and its weight Core's frozen whitening `whitening_factor(G_x, Σ_x)²` =
  `1/(G_x·Σ_x·G_xᵀ)`, `G_x = [−∇_u(log L∘R_k∘f + log V), 1]` the model's exact gradient (ForwardDiff through the two
  splines), `Σ_x = blockdiag(Σ_u, σ²)`. The weights are fixed per solve, at the previous solve's model: the I² solve, then
  `solves` Sampson solves. The example takes `Σ_u` from the registration's rotation bundle (its core scale, loss width /
  scale multiple, recorded in rotations.json as `bundle_loss_width_px`) and reads `vignetting.weights`,
  `log_intensity_scale` and `sampson_solves`. New dependency ForwardDiff.
- A noise model `var(I) = a·μ + b` (shot and read noise, measured on the frames) was tried as the weights during this
  release and is not added: its outer iterations did not settle and moved the gains by up to 15 %.

### Not added: a per-frame offset

An additive offset per frame (`I_k = g_k·L·V + b_k`, a veil of flare) was tested and rejected. The closed-form table, with V
held fixed at the radial fit's and the scene eliminated per direction (400k directions, the green plane, 1.4826·MAD of the
seams after a trimmed second solve), gave on the rooftop sweep log gains 0.0807, an offset alone 0.0967 and both 0.0799 (one
more parameter per frame for 1 %), and on the 14 mm indoor sweep log gains 0.0961, an offset 0.0936 and both 0.1676: an offset
is not supported on the rooftop and buys 3 % indoors. Fitted jointly with V by Gauss–Newton, the offset diverged on the
14 mm frames (b ≈ −2.8 after one iteration, then 10²–10⁵). The per-frame term is the exposure gain alone.

### Documentation

- The README's and the landing page's install block is the `pkg> add` line alone: the line
  `pkg> registry add General https://github.com/prittjam/CauRegistry.jl` named CauRegistry's URL "General" and is removed.

## 0.2.2

VisualGeometryToolkitCore 0.5.3 (compat `"0.5.3"`, for `stable_seed`); new dependency StableRNGs (Random dropped).

- `register_rotations(…; pairs = :lag_overlap)` no longer needs the lag pairs alone to connect the frames. When they leave
  them in more than one part (a lag pair dropped, for example for too few matches), the pair between two parts nearest in
  capture order is tried, and the next one, until the parts are joined; then the rotation averaging of every verified pair
  aims the overlap pairs as before. Before, that averaging ran on the lag pairs alone, and a dropped lag pair ended in a
  `KeyError` or an `ArgumentError` that the overlap pairs would have avoided. As with `pairs = :all`, only a graph that no
  pair joins is an `ArgumentError`.
- Each pair's RANSAC sampler is `StableRNG(stable_seed(seed, i, j))` (Core's SHA-256 seed keyed by the pair) instead of
  `Xoshiro(seed + 1000i + j)`, which gave one stream to different `(seed, i, j)`, for example `(1000, 1, 3)` and `(0, 2, 3)`.
  The streams are now also fixed across Julia versions. They change, so a registration's pairs, rotations and bundle move
  within the RANSAC's noise (the suite's tolerances hold). The example's registration record is `rotations/v3`, so a cached
  `rotations.json` from 0.2.1 is recomputed rather than reused.

## 0.2.1

VisualGeometryToolkitCore 0.5.1 (compat `"0.5.1"`).

- `render_panorama` (and so `render_cube_map`) of a grid lens's `SplineProjection` runs on GPU frames: the warp's sensor-point
  functor has an Adapt rule, and Core 0.5.1 adapts the `BoundedMap` and `ComposedTransformation` chain that wrap the projection
  in `image_warp`'s map. The device render equals the host render exactly (owner, coverage and image; a GPU test item). The
  example renders a grid lens on the device too.
- The vignetting splines' weights are Core's `cubic_bspline_weights` (the same formula at the same local coordinate: the same
  numbers), and `fit_vignetting`'s `residual_mad` and `score_seams`' `mad` are Core's `mad_scale`. Those two move by about
  1.5e-6 relative: `mad_scale` divides by the exact normal-consistency constant `Φ⁻¹(3/4)` (1/0.67449 = 1.482602…) where the
  private copies multiplied by 1.4826.

## 0.2.0

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
  grid's `SplineProjection` now samples GPU frames (its control grid was a host array inside a device kernel).
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
  read at the sensor point less its offset. `sample_frames` (and so `score_seams`) counts a direction as measured only when
  no saturated pixel enters ANY plane's bilinear blend at its own sample position (a first version checked the pixel
  centre's four alone, so R and B could blend a clipped neighbour by up to ¼ px of weight per axis).
- The render's warp math is Float64 whatever the frames' element type: GPU and host renders agree to 1e-7 relative.
- `detect_features` resamples the descriptor's input with Core's `image_warp!` (the same values: a downscale never reaches the
  frame's edge).
