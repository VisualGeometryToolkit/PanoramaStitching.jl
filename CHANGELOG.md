# Changelog

## Unreleased

### Breaking: the VisualGeometryToolkit 0.9 stack — the frames' and the planes' maps, Core's CG and penalty rows

- `raw_scale` (camera pixels per frame pixel) and `plane_offsets` (the planes' sample offsets) are gone from every entry point
  (`detect_features`, `verify_pairs`, `register_rotations`, `sample_frames`, `render_panorama`, `score_seams`,
  `fit_vignetting`): `frame_map` is the frames' map onto the camera's pixels (a Core `SimilarityMap`; a RAW's `:bin2`
  frames', VisualGeometryToolkitFeatures' `raw_frame_map(L)`), and `plane_maps` each plane's (`bayer_plane_map(L, c)`; by
  default every plane's is the frame's). Plane c is read at `inv(plane_maps[c])(u)`; the render reads the three planes in
  one Core `image_warp!(dst, warps, src)`. On the suite's fixtures the render and the seam score are bit-identical to
  0.5.1's (the BGGR plane maps against the offsets `(+¼, 0, −¼)`), the vignetting fit's control points within 2e-12
  (Cholesky) and 4e-9 (CG). A keypoint's covariance moves as `J Σ Jᵀ` through the frame map.
- `fit_vignetting`'s CG is VisualGeometryToolkitCore's `normal_cg` (Jacobi; cgls's stopping rule `‖g − N x‖ ≤
  tolerance·‖g‖`, so a warm start at the solution takes no iteration), and the scene's difference penalty is Core's
  `difference_rows` on the active controls (`λ DᵀD`, applied as a sparse operator beside the data's stencil); both equal
  the private copies they replace to rounding (the penalty to 6e-17). The private `_pcg`, `_penalty_kernel!`, `_upload`,
  `_zeros` and the docs gates' copy (`test/docs_gates.jl`: Core's `test_docs`) are deleted.
- The detector (VisualGeometryToolkitFeatures 0.9's ASIFT): `asift_tilts` is a `tilt_schedule` name (`:imas25`,
  `:untilted`) or its pairs (`nothing` is gone); the detections come in Features' canonical order (the package's own sort
  is deleted; the same keypoints in the same order on the suite's fixture); `detect_features`, `verify_pairs` and
  `register_rotations` take `saturated` (the frames' clip masks, a `LinearRaw`'s own by default): no keypoint within three
  of its scales of a clipped pixel (Features' detection mask). The example gives them its frames' saturation (verified pairs v5,
  rotations v9: the records of earlier versions are recomputed).
- The example: the decode cache (v3) holds no plane offsets (the maps are Features' of the set's first frame); the renders
  are encoded by Features' `srgb_image`; `compare_makie.jl` is Features' `compare_images`; the data root is
  VisualGeometryToolkitEval's `data_root` (Eval joins the example's environment). The README's example states its frame
  map, `SimilarityMap(LinearMap(2.0I))` (CoordinateTransformations joins its install line).
- The quality items run Core's `test_duplication` (the stack repositories from `VGTK_STACK_PATHS`).
- **Removed: `selection = :strongest`** (the global top `max_detections` by |response|, the private `_global_strongest`, a copy
  of the selection Features retired): `:spread`, Features' `spread_strongest` (SSC), is the one selection, and any other is an
  `ArgumentError`. Both configs of record select `"spread"` with `asift_tilts = "untilted"` (Features' schedule for a
  panorama's views); their registrations and panoramas change on a rerun (measured: the cascade's report).

### Breaking: the bundle's scale from the verified pairs — SIMFIT++ per pair, one LM run

- After its RANSAC each verified pair gets its noise scale by SIMFIT++ (Edstedt, "Less Biased Noise Scale Estimation for
  Threshold-Robust RANSAC", CVPRW 2025; VisualGeometryToolkitEstimation's `simfitpp_loss_width`, no new dependency): the rotation
  refitted on random train splits (`refine_rotation`, MSAC at the current width, the pair's local optimization), the
  held-out Sampson residuals' truncation-debiased median, χ²₂ (the residual's codimension) at coverage 0.99, seeded with
  the RANSAC's own σ̂, the sampler the pair's `(seed, i, j)` stream. `VerifiedPairs` gains `scales` (per pair `sigma`,
  `success`).
- `register_rotations` pools the kept pairs' scales into one σ̂ (`σ̂² = Σ nₚσₚ²/Σ nₚ`, `nₚ` the inliers, the pairs whose
  SIMFIT++ succeeded; none is an `ArgumentError`), `Registration.bundle_sigma`, and runs the bundle ONCE at the width
  `bundle_scale_multiple`·σ̂. `refine_rotations` takes `loss_width` (fixed) in place of `scale_multiple` and `iterations`:
  the outer loop that re-estimated the width from the Rayleigh median of all residuals is deleted, and `bundle_iterations`
  is gone (the example refuses the key by name). `max_iter = 0` is `:not_run`; the result's `iterations` are the LM's.
- `bundle_whitening = :keypoint_covariance` divides the keypoint covariances by their median `tr Σ / 2`, so the whitened
  residual is in pixels of the median keypoint and the width applies; an unrefined keypoint takes the identity.
- The example: `"welsch"` among the losses; the configs of record run the Welsch bundle at 2.5·σ̂, `bundle_max_iter = 1000`,
  `bundle_xtol = 1e-6` (radians); the records hold each pair's scale (verified pairs v4), σ̂ and the bundle's iterations
  (rotations v8). The self-calibration test's pinned numbers of 69471ff (a 3-run bundle) are retired.

### New: the view-graph filters, and the detector's variants

- `register_rotations` averages the rotations with VisualGeometryToolkitEstimation's `filtered_rotation_averaging`: the
  cycle filter at `cycle_tol` (default 2°) before the averaging, the orientation filter at `orientation_tol` (default 5°)
  after it, the averaging again when it removed a pair (radians; `nothing` skips that filter). A periodic scene (the hall's
  bays) verifies a pair one period off, which pairwise verification cannot reject. The removed pairs are
  `Registration.removed` (a new field); their matches make no track. A tolerance that is not positive is an
  `ArgumentError` before any feature is detected.
- `detect_features`, `verify_pairs` and `register_rotations` take `selection` (`:spread`, the default: about
  `max_detections` of the strongest spread over the frame, VisualGeometryToolkitFeatures' `spread_strongest`, which the
  LoMa descriptor applies; a global top-N baseline beside it, since removed) and `detection_image` (`:tone`, the
  default; `:vst`, the frame's variance-stabilized green cells, Features' `variance_stabilize`, at `noise_sigmas` noise σ,
  `detection_noise = (; electrons_per_dn, read_noise, noise_sigmas)`, the frame a Features `LinearRaw`). The kept
  keypoints are the described set's. A frame may be a `LinearRaw`.
- The example: `[registration]` keys `cycle_tol_deg`, `orientation_tol_deg`, `selection`, `detection_image` and, with
  `"vst"`, `electrons_per_dn`, `read_noise_dn`, `noise_sigmas` (refused with `"tone"`); `"vst"` reads the frames as
  `load_raw`'s `LinearRaw`. The records: verified pairs v3, rotations v7 (the tolerances and the removed pairs).
- The example: `[registration] local_optimization`, the pairs' RANSAC local optimization as an Estimation strategy table
  (read by Estimation's `local_optimization_from_toml`; `{ type = "irls", loss = "tls" }` is Estimation's rotation default,
  `{ type = "triggs", loss = "welsch", scale_multiple = 2.5 }` the robustified LM). `scoring_threshold_px` is MSAC's inlier
  threshold (its comment called it a half-width).

### New: the rotation bundle with the camera free — a rotating camera's self-calibration

- `refine_rotations(…; camera_free = (:f, :cx, :cy, :k1, …))` refines the named parameters of a parametric camera with the
  rotations and the track directions, in the same bundle (one more NLLSsolver variable: VisualGeometryToolkitCore's
  `ParameterManifold` of the camera's model restricted to those parameters, the camera block of Estimation's
  `refine_camera_and_poses`; its Jacobian ForwardDiff through the chart, the frame and direction blocks the closed form).
  The first frame's rotation stays the one gauge. `register_rotations` takes it as `bundle_camera_free`. A SplineLens
  camera is refused (its parameters are its lattice's controls, Estimation's own spline route).
- `RotationBundleMetadata` gains `camera` (the camera at the result: the given one when held) and `calibration`
  (`nothing` when held; else the free parameters' values, standard deviations and covariance — `s²·H⁻¹` of the robust
  Gauss–Newton information reduced to the rotations and the camera by the Schur complement of the directions — and that
  reduced information's scaled spectrum with and without the gauge frame: three zeros with it, none without).
- Every residual block of the bundle returns its cost through Core's `finite_cost` (non-finite is `+Inf`): an iterate past
  the model's valid domain is a rejected step also below `xtol`, where NLLSsolver accepted a NaN cost and failed its
  callback's check. The camera held, the bundle's numbers are unchanged (pinned by a test on 69471ff's numbers).

### Breaking: one view graph across the stack — VisualGeometryToolkitEstimation's `ViewGraph`

- **`PanoramaStitching.ViewGraph` is deleted** (no alias): the view graph of the verified pairs is
  VisualGeometryToolkitEstimation 0.9's `ViewGraph`, each edge the pair's rotation `R_ij` (`f_j = R_ij·f_i`) as
  VisualGeometryToolkitCore's `Attributed(R_ij, (; weight))`, its weight the pair's inlier count (read with Core's
  `value_of` and `weight_of`; the former `graph.rotations` and `graph.inliers` dictionaries are gone, and `graph.frames` is
  Estimation's `views(graph)`). `Registration.graph` holds it; `Registration` gains its type as a first parameter
  (`Registration{G, A, B, F}`).
- `register_rotations` averages the rotations with Estimation 0.9's `fit_rotation_averaging(graph; loss)` (the 0.8 form
  `fit_rotation_averaging(rotations, weights; loss)` is gone from Estimation; it was the `MethodError` of the example on
  the 0.9 stack). The algorithm, its spectral seed and its gauge (the first frame the identity) are 0.8's: the rotations
  are unchanged. A disconnected sequence is refused by the averaging (Estimation's `require_connected`, which names the
  components), as `ViewGraph` refused it before; `pairs = :lag_overlap` finds the parts with Estimation's `components`.
- `chain_tracks` (same signature, same tracks) finds the tracks as the components of the match graph, Estimation's
  `ViewGraph` on the observations, in place of its own union-find. Graphs.jl and DataStructures.jl are no longer
  dependencies.
- **New: `verify_pairs(frames, camera; …) -> VerifiedPairs`** — `register_rotations`' features, matching and pairwise
  rotations, stopping before the view graph (per frame its `FrameFeatures`; per verified pair its rotation, inlier count,
  inlier matches and statistics) — and **`register_rotations(pairs::VerifiedPairs, camera; averaging_loss, bundle_…)`**,
  the averaging, the tracks and the bundle from them. `register_rotations(frames, camera; …)` composes the two, its
  results unchanged.
- The example caches the verified pairs as a stage of their own, `verified_pairs.json`, written before the view graph is
  built and keyed by the matching and verification settings (every `[registration]` key but the averaging's and the
  bundle's; the averaging loss too with `pairs = "lag_overlap"`, whose averaging aims the pairs), the inputs, the decode
  settings and the RAW files' hashes: changing the averaging or the bundle reruns in seconds, and a failure past the pairs
  loses none of the matching.

### LoMa on the GPU wherever it runs: `execution_provider = :auto`

- `register_rotations`' `execution_provider` defaults to `:auto` (was `:cpu`): LoMa's descriptor and matcher run on ONNX
  Runtime's CUDA provider where VisualGeometryToolkitFeatures finds a CUDA device and the environment holds
  `CUDA_Runtime_jll` and `CUDNN_jll` (their CUDA 12 builds, loaded for ONNX Runtime alone, beside CUDA.jl's runtime), and on
  the CPU otherwise, warning once why. `:cpu` and `:cuda` stay; `:cuda` that cannot run is an error naming why. The provider
  is resolved once per registration by Features' `resolve_execution_provider` (in `verify_pairs`), and the `VerifiedPairs`
  and the `Registration` record what ran LoMa in their new `loma` field (Features' `loma_provenance` of the descriptor and of the matcher: the models, the provider used,
  ONNX Runtime's threads and, on CUDA, the driver, runtime and cuDNN versions). cuDNN.jl is no longer needed (cuDNN 1.4, the
  last for CUDA.jl 5, never resolved beside CUDA_Runtime_Discovery 2, so the configs had stated `"cpu"`).
- ONNX Runtime's threads are Julia's (`Threads.nthreads()`), not the machine's: a Slurm job's registration stays in its
  allotment (on dgx01 six 28-CPU jobs had driven a load of 673 at ~70 min of CPU registration each, the GPUs idle).
- The example: the configs of record state `execution_provider = "cuda"` (never the CPU); its environment holds `CUDA_Runtime_jll` and `CUDNN_jll`
  (lazy artifacts: downloaded on the first LoMa session where a device is visible), and `run.jl` no longer loads cuDNN. The
  keys of both cached stages hold the provider `"auto"` resolved to (a CPU record is not reused by a CUDA run: their matches
  differ at the network's threshold); `verified_pairs.json` (format v2) and `rotations.json` (format v6; older records of
  either are recomputed) record LoMa's provenance
  and `summary.md` states the provider, its library versions and the threads.
- TEMPORARY `[sources]` take Core, Estimation and Features (and the example's IO) from the monorepo snapshot
  `~/src/vgtk-bounded-camera-region` (the 0.9 release branch, a descendant of `~/src/vgtk-09-base`'s d2af91ef), in place of
  `~/src/vgtk-board-deform`; compat admits that tree's own version fields (Estimation 0.8.0, Features 0.6.1, unbumped) until
  the cascade moves it to the released versions.

### Breaking: the VisualGeometryToolkit 0.9 camera — `imaged_region` replaces every `circle`

- Requires VisualGeometryToolkitCore 0.9 (and Estimation and IO of the same stack). Until they register, TEMPORARY
  `[sources]` take them from a monorepo snapshot; never released while they are there.
- `register_rotations`, `detect_features`, `sample_frames`, `render_panorama` (and `render_cube_map`), `score_seams` and
  `fit_vignetting` take the camera as a VisualGeometryToolkitCore `BoundedCamera` and read where it images from the camera
  itself: its `imaged_region` (the sensor ∩ its lens coverage ∩ its model's valid domain), its coverage's edge less
  `detector_margin_px` for the features and `rim_margin_px` (and 2 px inside the frame) for the samples, the renders and the
  fit. The sensor's edge is not eroded, so the feature and sample masks of a camera with no stored coverage (its whole
  sensor) whose model's valid domain covers the sensor are 0.5.1's with no circle. The `circle` keyword is
  deleted everywhere (no alias): the image circle is the camera's lens coverage,
  `BoundedCamera(model, extent; lens_coverage = Circle(…))`; a camera without one images its whole sensor.
  `detect_features(camera, frame, descriptor; …)` takes the camera first.
- The `projection` keyword of `register_rotations` and `refine_rotations` is deleted (no alias): the camera is its own
  projection, its model's `imaging_transform`.
- A grid lens is a `SplineLens` camera (Core 0.9 retired `CentralGenericModel` and `SplineProjection`): it projects through
  its own forward spline, `fit_camera(SplineLens, camera, imaged_region(camera); tolerance)`, kept in the camera with Core's
  `with_model`. The rotation
  bundle's residual is NaN only off the camera model's `valid_domain` (where its projection and back-projection Jacobian
  are defined: a grid lens's lattice), never off its lens coverage: a parametric camera's bundle is 0.5.1's.
- The rotation bundle drops the observations whose start prediction is not imaged (off the camera model's valid domain,
  where it has no projection and no back-projection Jacobian) and the rest of any track left in fewer than two frames, and
  solves the rest: a partly kept track restarts from the mean bearing of the observations it keeps and is tested again
  until it loses none (a track left out keeps its start direction, held). `RotationBundleMetadata` reports
  them as `n_dropped_start` (a dropped observation's residual is NaN), replacing `start_nonfinite`; the outcome
  `:nonfinite_start` is retired. A start that leaves nothing to solve is an `ArgumentError`. A frame other than the gauge left with no kept
  observation is held at its start rotation and listed (`frames_unconstrained`; the metadata is
  `RotationBundleMetadata{K}`). The example's `rotations.json` (format v5: older records are recomputed) and `summary.md` record
  the count and the frames held.
- The example: `[inputs] circle` is refused with an `ArgumentError` naming the camera's `lens_coverage`; `load_lens` returns
  the `BoundedCamera` (a camera store's, or an old grid file read as a `SplineLens` camera whose coverage is the file's
  domain, its forward spline fitted over its imaged region). The configs of record no longer hold their circles; the
  14 mm config runs on the whole sensor until its camera store carries the measured circle as its lens coverage.

## 0.5.1 (2026-11-09)

On the VisualGeometryToolkit 0.8 stack, from General: VisualGeometryToolkitCore 0.8,
VisualGeometryToolkitEstimation 0.8 and VisualGeometryToolkitFeatures 0.6.1 (its compat floor, the first on
Core 0.8); the example also VisualGeometryToolkitIO 0.8. Nothing else changes.

## 0.5.0 (2026-11-07)

### Breaking: the VisualGeometryToolkit 0.7 stack

- Requires VisualGeometryToolkitCore 0.7, VisualGeometryToolkitEstimation 0.7 and VisualGeometryToolkitFeatures 0.6
  (the example also VisualGeometryToolkitIO 0.7), all from General; NLLSsolver 5.3.2.
- Core 0.7's renames: a keypoint's shape is `shape_matrix` (`moment_matrix` is retired); a grid model's `backproject` is
  NaN where it images nothing (the tests' synthetic sequence reads it so).

### Breaking: PanoramaStitching's copies of the stack deleted (no aliases)

- `AzimuthalChart` is Core's chart and lattice: fields `chart` (Core's `equidistant_chart(axis)`), `area` (the square of the
  chart the controls are laid over, radians) and `n`; the constructor `AzimuthalChart(axis, half_extent, spacing)` takes
  radians (was `(center, half_extent_deg, spacing_deg)`, with fields `e1, e2, e3, half_extent_deg, spacing_deg, n`). log L is
  Core's `spline_value` of the scene controls at the direction's chart point, and the normal equations scatter the 16
  controls of Core's `spline_window`; the private `_spline_span`, `_chart_point`, `_chart_span` and `_log_radiance` are gone.
  `fit_vignetting`'s `chart_spacing_deg` stays in degrees.
  - `VignettingMetadata.scene` is the `n × n` controls matrix in Core's layout (`[row, column]` along the chart's second and
    first coordinate: `spline_value(scene, chart.area, chart.chart(d))` is log L at `d`); it was an `n²` vector, first
    coordinate fastest.
  - Two numbers move. Core's chart basis is `e₁ = normalize(ŷ × axis)`, where PanoramaStitching's was the Gram–Schmidt of
    x̂: the scene's lattice turns about the axis. And a direction outside the area continues the outer cubic piece instead of
    clamping to the edge cell (the area holds the coverage plus one spacing, so no row reaches it). Measured on the test
    suite's synthetic sequences (`chart_spacing_deg = 1`): log V's control points move by at most 1.5e-7 (radial, CG),
    2.2e-8 (radial, Cholesky), 9.2e-9 (radial, value-and-position noise) and 1.9e-5 (16 × 16 tensor surface); the V errors
    against the truth move in the third significant digit (radial max 4.303e-5 → 4.295e-5; tensor max 6.3034e-3 →
    6.3033e-3), the gains by 7e-11, and the active scene controls are 42530 for 42523 (CG: 255 iterations for 248).
- `IntensityAndPositionNoise(; point_covariance, log_intensity_scale, iterations)`: `point_covariance` is Core's `CovMat{2}`
  (`CovMat{2}(σ = σ_u)` for the isotropic `σ_u²·I`), and the `point_scale` keyword is removed. A bare matrix is a
  `TypeError`; the point's noise missing is an `UndefKeywordError` (was an `ArgumentError`).
- `PanoramaStitching.LOMA_SIZE` is removed: Features' `LoMaDescriptor` takes the frame at its own size (it resamples to its
  network's input, `VisualGeometryToolkitFeatures.LOMA_IMAGE_SIZE`, by the same bilinear warp), so the detections are no
  longer rescaled to 784² by hand and the described features carry them in the frame's pixels; the matcher takes each
  frame's `(H, W)`. The `max_detections` kept are Features' top-N selection of then. The registration's matches, inliers and
  rotations are unchanged on the synthetic sweep (rotations to 4e-14°, bundle rms to 4e-13 px).
- A rotation bundle's track direction is Core's `Eliminable{E}` of its `SphereManifold` (the private `_TrackDirection` is
  gone); the loss width's scale is Core's `noise_sigma(norms, 2)`; the image circle in a binned frame is Core's
  `SimilarityMap` image of the `Circle` or `Ellipse`; an array index's COLMAP point is Core's `pixel_centre`; V's weak
  penalty is Core's `difference_rows(size, 2)`. Each is the same number as before (bit for bit, or to rounding).

### The example

- `vignetting.model` and `compare_models` are tags of Core's `VIGNETTING_TAGS`: `"radial"` and `"tensor_product"` (the
  retired `"tensor"` is refused by name). `config_fx3_14mm.toml`'s `compare_models = ["tensor_product"]` (was `["tensor"]`),
  so its comparison files are `vignetting_compare_tensor_product.*`, and the records' `model.kind` is the tag. The edited
  configs move the arms runs' keys (`compare_arms.jl`, the content of `arms.toml`, the set's config and its
  `rotations.json`): `fx3_14mm_indoor` `1fe3af91b1e02a7a` → `bcd50f33a2b18240`, `fx3_8mm_rooftop` `248be13ee1c0ad3f` →
  `a946d2c76bcfb407`. The registration's key (its own config sections) does not move.
- The decode cache's, the registration's and the arms' hashes are Core's `sha256_hex` (the same digests); the example no
  longer depends on SHA.

## 0.4.0

### Breaking: the VisualGeometryToolkit 0.6 stack

- Compat VisualGeometryToolkitCore "0.6", VisualGeometryToolkitEstimation "0.6.1", VisualGeometryToolkitFeatures "0.5" (the
  package, test and docs environments) and VisualGeometryToolkitIO "0.6" (the example). PanoramaStitching's API carries Core's
  types (`fit_vignetting` takes and returns a Core `LogVignetting`; the bundle's observations are Core `Uncertain` pixels), so
  their 0.6 changes are PanoramaStitching's too.
- Core's tensor-product vignetting surface is `TensorProductVignetting` (Core 0.6 retired `TensorVignetting`): a 2-D V's model
  is `LogVignetting(TensorProductVignetting(…))`. The example builds it under the new name; its config value
  `vignetting.model = "tensor"` (and `compare_models = ["tensor"]`) is unchanged.
- Core 0.6's `CovMat` holds a static matrix (full, diagonal or isotropic); the keypoints' `CovMat(Σ)` of a full `SMatrix` is
  the full form, as before, so the bundle's whitening is unchanged. `moment_matrix` (the keypoints' affine shape) is unchanged
  in Core 0.6.

### Breaking: renames (no aliases; every number unchanged for the same settings)

- Gain correction: `FrameGains` → `IntensityGains` (one gain per frame shared by every colour plane: a gain on the intensity);
  its tag in `GAIN_CORRECTIONS` (`vignetting.gain_correction`) `"frame"` → `"intensity"`. `ChannelGains` / `"channel"` and
  `"none"` are unchanged.
- Weights → measurement noise. The option chooses the measurement-noise model; the row weights are always the whitening of the
  log-I constraint by it (VisualGeometryToolkitCore's `whitening_factor`). "Sampson" is gone from the option's names: in the
  stack's vocabulary `sampson_` is the GEOMETRIC distance, while these weights are in σ units.
  - `IntensityWeights` → `IntensityNoise` (noise in the pixel value only, read noise of constant σ: weight I²).
  - `SampsonWeights(; point_covariance | point_scale, log_intensity_scale, iterations)` →
    `IntensityAndPositionNoise(; the same keywords)` (noise in the value and the pixel's sensor position).
  - `fit_vignetting`'s keyword `weights` → `measurement_noise`; `VignettingMetadata.weights` → `measurement_noise`.
  - The registry `PanoramaStitching.WEIGHTS` (family `:weights`, suffix `Weights`) → `PanoramaStitching.MEASUREMENT_NOISES`
    (family `:measurement_noise`, suffix `Noise`), tags `"intensity"` and `"intensity_and_position"` (was `"sampson"`), TOML
    path `vignetting.measurement_noise`.
- Retired tags, refused with an `ArgumentError` that names the new tag (the registries' `retired`): `gain_correction = "frame"`
  (→ `"intensity"`), `measurement_noise = "sampson"` (→ `"intensity_and_position"`).
- The example's TOML keys: `vignetting.weights` → `vignetting.measurement_noise`, `vignetting.sampson_iterations` →
  `vignetting.measurement_noise_iterations` (`log_intensity_scale` unchanged); in `arms.toml` the same two keys (top level and per
  arm). The example's loaders (`run.jl`'s `load_config`, `compare_arms.jl`) refuse each old key by name, `ArgumentError`
  "the key … was renamed to …": an old config fails, it is never read as the new setting.
- The example's arms (`arms.toml`) carry labels of their own instead of their tags: `free_frame_intensity` →
  `selfcal_shared_gain_value` (the `fixed_from` V of record), `free_none_intensity` → `selfcal_no_gain_value`,
  `free_channel_intensity` → `selfcal_plane_gain_value`, `fixed_none_intensity` → `given_v_no_gain_value`, `fixed_frame_intensity` →
  `given_v_shared_gain_value`, `fixed_channel_intensity` → `given_v_plane_gain_value`, and the `…_sampson` arms the same with
  `…_value_point` (`free_frame_sampson` → `selfcal_shared_gain_value_point`, …). `results.json` and `results.md` name the column
  `measurement_noise`. A comparison's key is the content of `arms.toml`, so earlier results are not reused.

## 0.3.0

### Breaking

- The vignetting model is VisualGeometryToolkitCore's (0.5.6, compat "0.5.6"): `RadialVignetting`, `TensorVignetting` and `SplineVignetting`
  and `log_vignetting` are gone from PanoramaStitching. `fit_vignetting(vignetting, camera, frames, rotations; …)` takes a
  Core `LogVignetting` (V = exp(s), `s` a `RadialVignetting` or `TensorVignetting` spline of log V: the model of earlier
  releases, linear in log V) as V's model and start, and its estimate is the fitted Core model (callable `v(u)`, isbits: the
  `vignetting` keyword of `render_panorama`, `sample_frames` and `score_seams` on the host or a device, which still take any
  `vignetting(u, f)` as well). The fit is the one linear solve in log V of earlier releases. A V held fixed is any
  vignetting, a V-form `RadialVignetting` or `TensorVignetting` included. The model's knots are clamped (Core's), on the
  same breakpoints as before (a radial profile on `[0, radius]`, a tensor surface over the box). `VignettingMetadata` gains
  `weights` (the fit's: its Sampson iterations give the linear solves) and `log_change` (the model's change in the last solve).
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

- **A precompile workload** (PrecompileTools, `src/precompile.jl`): `fit_vignetting` (radial and tensor, frame and channel
  gains), `render_panorama`, `render_cube_map`, `score_seams` and `sample_frames` on two synthetic 40 × 40 frames are compiled
  into the package image (CPU only, no data files, no output): a fresh process's first calls take 3.8 s instead of 10.0 s, and
  the package precompiles in 16 s instead of 4 s. `refine_rotations` is not in it (an NLLSsolver solve evaluates into
  NLLSsolver, which a precompiling package may not do). Every result is unchanged.
- `fit_vignetting`'s `weights` keyword: `IntensityWeights()` (the default, I²: every earlier call and its numbers stand)
  or `SampsonWeights(; point_scale | point_covariance, log_intensity_scale = 0.01, iterations = 1)`. A row is the implicit
  constraint `g = log I − log L(R_k·f(u)) − log V(u) − γ_k` on the measurement `(u, log I)`, through VisualGeometryToolkitCore's
  `constraint_with_jacobians` protocol, and its weight Core's frozen whitening `whitening_factor(G_x, Σ_x)²` =
  `1/(G_x·Σ_x·G_xᵀ)`, `G_x = [−∇_u(log L∘R_k∘f + log V), 1]` the model's exact gradient (ForwardDiff through the two
  splines), `Σ_x = blockdiag(Σ_u, σ²)`. The weights are fixed per solve, at the previous solve's model: the I² solve, then
  `iterations` Sampson reweight-and-solve cycles. The example takes `Σ_u` from the registration's rotation bundle (its core scale, loss width /
  scale multiple, recorded in rotations.json as `bundle_loss_width_px`) and reads `vignetting.weights`,
  `log_intensity_scale` and `sampson_iterations`. New dependency ForwardDiff. The weights' tags are
  `PanoramaStitching.WEIGHTS` (a `TagRegistry`: `"intensity"`, `"sampson"`).
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
