# Changelog

## 0.8.2 (unreleased)

### Changes

- `refine_rotations` / `refine_camera_and_rotations`: `eliminate = true` is refused by name (an `ArgumentError` naming
  NLLSsolver's missing `MultiVariateLSschur`) where the loaded NLLSsolver has no Schur system, which no installable
  NLLSsolver has; there it was silently the joint solve. `eliminate` defaults to whether the Schur system is there (`false`
  today, the only path). Migrate: drop `eliminate = true`. The keyword is to be removed in wave 11
  (`.release/wave11-removals.md`).
- `register_rotations` / `verify_pairs`: a camera whose footprint VisualGeometryToolkitCore's `view_overlap` refuses (a
  region the lens cannot back-project, as a bare sensor's corners past a fisheye's fold; a model that is not a `CameraModel`;
  a field of view within 3° of the backward axis) is refused at the entry, before any feature is detected, wherever the
  registration may read `view_overlap` (the near-duplicate check, unless `near_duplicate_overlap = 0`; `:lag_overlap` with a
  pair past the lag). 0.8.0 and 0.8.1 refused it only after every pair was matched. Migrate: give the camera its lens coverage.

### Tests

- The references are derived and independent of the solver (0.8.1's eliminated-versus-joint comparison compared a solve with
  itself: without the Schur system both run the same path):
  - the held noisy sweep: its rotations against the truth, within the Cauchy M-estimator's asymptotic covariance
    `κσ²(JᵢₙᵀJᵢₙ)⁻¹` (J the bundle's Jacobian at the truth by ForwardDiff, κ the Cauchy efficiency at the loss width under
    Gaussian noise) less the outliers' first-order bias, at `χ²₁₂(0.999)`;
  - the sweep's inlier observations: two starts reach one minimum, within the sum of their stationarity radii
    `√(gᵀH⁻¹g)` (the robust cost's gradient and Hessian at each estimate, one metric, the gradient's rounding added);
  - the KB scene held at the wrong camera: no minimum is asserted (its robust cost has many shallow minima: starts 0.3°–1°
    off stop at rms 13.49–13.63 px), the camera held and the fit not exact.
- A camera refused by the footprint preflight is tested with a `say` callback that hears no detection.

## 0.8.1 (2026-11-14)

A test-only patch: no source file under src/ changed, no API change.

- The bundle tests no longer pin a count of LM iterations or the last digits of a minimum recorded on one machine (GitHub's
  runner failed 0.7.0's `test/autocalibration_tests.jl:173`, `(res.iterations, res.n) == (19, 1489)`: the iterations and
  the last digits follow how each machine's compiled code rounds). Each bundle asserts it converged by a tolerance within
  its iteration cap, with its fixture's observation count; the exact-data autocalibration keeps its derived rms bound
  (`2eps(1200.0)`) and its checks against the true camera and rotations; the camera-held bundles are compared with the same
  bundle solved jointly (`eliminate = false`) at the stop tolerances' own bound (the rms to `rtol = ftol`, the rotations and
  directions to `rtol = 2xtol`). Corrected in 0.8.2: that is not another rounding path; without NLLSsolver's Schur system
  (no installable NLLSsolver has it) `eliminate = false` and `true` run the same solve, so the comparison was vacuous.
- The example's `write_results_check.jl` compares the bundle record it reads back with the fixture's own record instead of
  restating its numbers.
- The suite fits a 16 GB CI runner: each finished test item's globals (its scenes, frames and LoMa's ONNX Runtime sessions)
  are released and the memory handed back to the OS before the next item (`test/runtests.jl`'s `test_end_expr`).
  ReTestItems keeps every item's module bound, so the one worker of GitHub's 4-core runner grew to its 16 GB and the
  runner died. Under a 16 GiB limit on dgx01, with GitHub CI's command, the worker went from 15.2 GiB after item 22 (killed
  in item 24) to at most 3.95 GiB after any item, 7.6 GiB peak. Registration itself holds nothing after it returns.
- 0.8.0's CHANGELOG entry dated.

## 0.8.0 (2026-11-14)

One notion of view overlap across the stack (owner ruling 2026-11-14): PanoramaStitching decides near-duplicates and pair
selection by VisualGeometryToolkitCore's `view_overlap` — the solid angle two frames of a rotating camera both image, as a
fraction of each footprint — and no longer by the angle between optical axes (0.7's `_axis_angle_deg`, a second overlap
estimate beside Core's).

## Breaking changes

- **`near_duplicate_deg` is retired**, refused by name (an `ArgumentError` naming `near_duplicate_overlap`), with no alias.
  A near-duplicate pair is now a kept pair whose frames share at least `near_duplicate_overlap` of the SMALLER footprint
  (`view_overlap` under the averaged rotations, `max(fraction_i, fraction_j)`). Migrate: drop `near_duplicate_deg` (and
  `near_duplicate_deg = nothing`); the default `near_duplicate_overlap = 0.90` is the owner's value, and
  `near_duplicate_overlap = 0` turns the check off. A value outside `[0, 1]` is an `ArgumentError`.
- **Near-duplicates are always on.** 0.7's default (`near_duplicate_deg = nothing`) chained every kept pair; 0.8's default
  leaves out of the chaining every kept pair that shares ≥ 0.90 of the smaller footprint (it is still verified and
  averaged; `Registration.near_duplicates`). A registration that set no `near_duplicate_deg` changes wherever two frames
  share that much view: fewer tracks, and the bundle's rotations differ. `near_duplicate_overlap = 0` restores 0.7's default.
- **`overlap_deg` is retired**, refused by name (an `ArgumentError` saying what replaced it), with no successor keyword.
  `pairs = :lag_overlap` now tries, past the lag, every pair whose frames share ANY view under the rotation averaging of the
  verified pairs (`view_overlap`'s solid angle above zero), and `min_inliers` decides which are kept. Migrate: remove
  `overlap_deg`. A `:lag_overlap` registration tries a different set of pairs past the lag (every overlapping pair, where 0.7
  tried those within the axis angle), so its verified pairs, tracks and rotations differ. `pairs = :all` is unchanged.
- `max_lag` alone is `:lag_overlap`'s setting (needed there, refused with `:all`).
- Every registration now evaluates `view_overlap` on the registration's camera (for the near-duplicates; and, with
  `:lag_overlap`, for the pairs past the lag), so its refusals apply: the camera's model a `CameraModel` (a `SplineLens`
  lens imaging ẑ), its footprint its `imaged_region` (its lens coverage), no field of view within 3° of the backward axis.
  **A registration whose camera's footprint the lens cannot back-project is refused** (`view_overlap`'s `ArgumentError`:
  "the lens does not back-project the region's boundary pixel …"), as when a bare sensor's corners lie past a fisheye's
  fold. Set the camera's valid region: its lens coverage, the image circle where light reaches the sensor
  (`BoundedCamera(model, extent; lens_coverage = Circle(centre, radius))`, or `with_lens_coverage`), inside where the lens
  back-projects. 0.7 accepted such a camera.
- The example: `[registration] near_duplicate_deg` and `overlap_deg` are refused by name when a config is read; the configs
  of record state `near_duplicate_overlap = 0.90` (were `near_duplicate_deg = 4.0`, 14 mm, and `9.0`, 8 mm). Its registration
  record is format v17 and its verified pairs v12, so 0.7's records are recomputed, not reused.
- A retired keyword's refusal reads "`<key>` is retired: …" (was "the keyword `<key>` is retired: …"; the example's
  "the key <key> is retired: …"), one message for a keyword and a config key alike.

## Added

- `PanoramaStitching.refuse_retired_registration(table, where)` (public, not exported, as `detect_features` and
  `pair_bearings` are): the ONE list of the retired registration settings, refused by name with each one's successor, for
  a config's `[registration]` table (`String` keys) and keywords (`Symbol` keys) alike. `register_rotations` and
  `verify_pairs` refuse their retired keywords through it, and the example's config loader calls it (its own list,
  `RETIRED_REGISTRATION_KEYS`, and function are deleted); a downstream loader (RC's cvpr2027 configs) calls it on the table
  it reads.

## 0.7.0 (2026-11-13)

The breaking wave of the stack (VisualGeometryToolkitCore 0.10, IO 0.10, Features 0.9, Estimation 0.11): the terminology
taxonomy — a detector's output is a **detection**, a hypothesised pairing a **correspondence**, a detection associated with a
model entity an **observation** (Core's `conventions/naming.md`).

### Breaking

- `[compat]`: VisualGeometryToolkitCore "0.10", VisualGeometryToolkitFeatures "0.9", VisualGeometryToolkitEstimation "0.11"
  (the example: VisualGeometryToolkitEval "0.8", VisualGeometryToolkitIO "0.10"); floors raised to the gated resolution at
  release.
- `fit_vignetting`'s keyword `measurement_noise` → `observation_noise`, and `VignettingMetadata`'s field of the same name; the
  old keyword is refused naming the new one (`ArgumentError`), no alias.
- `MEASUREMENT_NOISES` → `OBSERVATION_NOISES` (the `TagRegistry` family `:observation_noise`; its tags unchanged:
  `"intensity"`, `"intensity_and_position"`).
- The example's TOML keys `vignetting.measurement_noise` → `vignetting.observation_noise` and `measurement_noise_iterations`
  → `observation_noise_iterations` (in `arms.toml` too); a config holding an old key is refused by name.
- On VisualGeometryToolkitCore 0.10's names: `correspondences` (was `csponds`) and `bearing_correspondences` (was
  `bearing_csponds`) in `pair_bearings`; no API change of PanoramaStitching's own from them.
- `verify_pairs` / `register_rotations`: `max_lag` and `overlap_deg` are read only by `pairs = :lag_overlap`, which needs
  both; they now default to `nothing` and giving either with `pairs = :all` is an `ArgumentError` (no inert setting). So is
  `verify_pairs`' `averaging_optimization_strategy` with `pairs = :all` (`register_rotations` still takes it for its averaging).
  The example's configs of record drop the two keys (they use `pairs = "all"`), which re-keys their verified-pairs stage;
  the example refuses the two keys with `pairs = "all"` when it reads a config. Its registration record is format v15
  (`pairs_removed` the filters' only), so records of 0.6 are recomputed, not reused.
- The results of record through VisualGeometryToolkitIO 0.10's `vgio/panorama/v2.0` (0.6.4's v1, reshaped by IO 0.10: the
  envelope's units and conventions, the rotations' `frame_index`; IO reads a 0.6.4 v1 file through its step). The example
  and the test environment require VisualGeometryToolkitIO "0.10"; `save_panorama` and `load_panorama` keep their
  signatures (IO 0.10 adds the optional `fit`: the example stores the registration bundle's termination record, kept in its
  registration record as Core's values, `bundle_fit`, format v15), and the example names PanoramaStitching as the file's
  generator (`VgtkIO.with_caller`). V stays inside the
  panorama file (its `vignetting` is the `vgio/vignetting` body): no separate vignetting file.
- The detector's noise is the frame's sensor model (VisualGeometryToolkitCore's `SensorModel`, its ONE resolver `sensor_model`
  of the body's characterisation): `detect_features(...; detection_image = :vst, sensor_model, noise_sigmas)`, and
  `verify_pairs` / `register_rotations(...; sensor_models, noise_sigmas)` with one model per frame (Features'
  `variance_stabilize(L, sensor_model)`); `:vst` needs a gain (a `PoissonGaussian`), any other noise refused naming the
  model's reason. The 0.6 keyword `detection_noise` is refused naming them; with `:tone` neither is given. The example
  resolves each frame's model (`frame_models`: the RAW files' capture settings, the body's `[registration]
  sensor_characterisation` — a `vgio/sensor_characterisation` file or `"none"` — and Core's `sensor_model`), decodes the
  LinearRaw with it, and refuses the 0.6 keys `electrons_per_dn` and `read_noise_dn` by name.
- New public API, hoisted from the example (RC's cvpr2027 called them from run.jl): `sweep_up(rotations) -> (; up, eigenvalues)`
  (the up of a roll-free sweep: the least eigenvector of Σ x_k x_kᵀ over the frames' image x axes, signed against their down
  axes; the eigenvalues its roll-free diagnostic), `up_frame(up::AbstractVector)` (the true-up frame of an up direction,
  VisualGeometryToolkitCore's `lookat_pose` frame; `up_frame(pitch, roll) = up_frame(true_up(pitch, roll))`, the one frame
  construction the example's `level_up` duplicated), `coverage_extent(camera, frames, rotations, up, frame_map; δ)` (the
  sweep's longitude/latitude box) and `vignetting_summary(v, center, frame_map, frame_size; mask)` (a V's range, maximum and
  antipodal asymmetry; it took an unused camera and rim before). The example's `level_up`, `coverage_extent`,
  `vignetting_summary` and `principal_point(lens)` are deleted; it calls these and Core's `principal_point(intrinsics)`.
  `up_frame` now goes through Core's look-at frame: equal to 0.6's to rounding (≤ 4 eps), not bit for bit.
- The example's decoded-RAW cache is opt-in, `[decode] cache = true | false` (the configs of record set `true`), and the example
  loads no JLD2 itself: VisualGeometryToolkitEval brings it, and with it Features' FileCache extension. A caller that includes
  run.jl (RC's cvpr2027) needs no JLD2 of its own; the example's environment drops its direct JLD2 entry.
- `fit_vignetting`'s value noise is the frames' own (owner ruling D10), by dispatch on the image type: an uncertain image —
  VisualGeometryToolkitCore's `LinearRaw` under a sensor model whose noise feeds each value's variance (a `PoissonGaussian`, or a
  specified `RelativeNoise`), a `StructArray` of `Uncertain` values — weights a row by I²/var(I); a plain image keeps the
  read-noise fallback, weight I² (bit for bit the fits before 0.7; the configs of record decode plain images). The metadata
  gains `value_variance`. `IntensityNoise` is deleted (its weights are the plain image's fallback), and
  `IntensityAndPositionNoise` becomes `PositionNoise(; point_covariance, iterations)`: its constant σ of log I
  (`log_intensity_scale`) is a specified `RelativeNoise` of the frames' sensor model, and it needs the frames' variances.
  `observation_noise` is required (`nothing` or a `PositionNoise`); `OBSERVATION_NOISES` tags `"none"` and `"position"`
  (`"intensity"`, `"intensity_and_position"` refused naming them). The example's `[vignetting] log_intensity_scale` is
  refused; `arms.toml` states the position arms' `relative_noise = 0.01`.
- The example's V map is a data result (owner ruling "RAW + results only"): `<stem>_map.json` + `.tiff`, a Float32
  `vgio/raster` of V over the frame's pixels (IO's `save_raster`; read with `load_raster`), in place of the normalised 8-bit
  `<stem>_map.png`. The panorama renders and the seam maps stay PNG: display figures, said so in the example's README.
- Prose, docstrings and test names: "keypoint" → "detection", "measurement" → "observation", the example's cache formats
  and their version history included (owner, 2026-11-13): its verified-pairs record is format v11, so 0.6 caches are
  recomputed, not reused.
- `register_rotations` checks the tracks by default (`check_tracks = true`): after the bundle, the observations whose
  residual exceeds 3.035 σ̂ (√χ²₂(0.99), σ̂ the pooled scale) are removed and the bundle re-solved from its rotations, until
  none is removed (at most 10 solves); `Registration.track_check` holds each solve's removals, rms and inlier rms. So the
  rotations of a registration differ from 0.6's; `check_tracks = false` keeps the single bundle. New keyword
  `near_duplicate_deg` (default `nothing`, every kept pair chained): a kept pair whose optical axes are within it is
  averaged but seeds no track (`Registration.near_duplicates`). `RotationBundleMetadata` gains `inlier_rms` (the rms over
  the observations within the loss width). The example's configs of record set both; its registration record is format
  v16, so v15 records are recomputed, and `provider_agreement_check.jl` (LoMa on the CPU and on CUDA agree) joins its checks.

### Fixed

- `refine_camera_and_rotations`' covariance (`metadata.calibration`) is accumulated block by block — the frames' and the
  camera's information, per track its 2 × 2 direction block and coupling — and the directions eliminated by their Schur
  complement, instead of dense rows of every observation over every unknown (30 frames × 3000 tracks: about 0.9 GB and a dense
  inverse of the ~6100 unknowns, on every call). The same quantities, to rounding.
- The autocalibration bundle computes the camera model's `valid_domain` once per value of the camera variable, not in every
  residual and Jacobian (a lens with its own fold searches its view-angle limit for it).
- `Registration.removed` lists only the pairs the view-graph filters removed; a pair with no determined rotation covariance
  is in `no_covariance` only (it was in both).
- `pairs = :lag_overlap`: a lag pair with no determined covariance that splits the frames no longer stops the registration
  before the pairs past the lag are tried; the joining pair across the split is tried, and the final graph is checked.
- Core's `with_model` replaces PanoramaStitching's private copy; `docs/Project.toml`'s Core compat aligned with the package's.

## 0.6.5 (unreleased)

A patch: no API change, the same results. The example writes its results of record again.

- **The example's `write_results` failed on every run** (0.6.4): after the vignetting fit, `run.jl` built each frame's pose
  as `EuclideanMap(R, zeros(3) .* m)` with `R` the registration's `SMatrix{3, 3}`. VisualGeometryToolkitCore takes a raw
  3 × 3 matrix only with a static translation (its boundary constructor `EuclideanMap(::SMatrix{3, 3}, ::StaticVector{3})`),
  so the call was a `MethodError` and no `panorama.json` was written. The zero translation is now a static length,
  `zero(SVector{3, Float64}) .* m`, through Core's existing constructor (no new Core method). The guide's section 5 had
  the same call; corrected, and the test's pose written the same way.
- `examples/panorama_vignetting/write_results_check.jl`, run by `run_checks.jl` (the example's checks): `write_results` on a
  tiny fixture writes `panorama.json` and the map image, and the record reads back with the frames' rotations, zero
  translations, V and the gains. It fails on 0.6.4's code.

## 0.6.4 (unreleased)

A patch: no API change, the same results. The results are saved and loaded through VisualGeometryToolkitIO.

- **The results through `vgio/panorama/v1`** (VisualGeometryToolkitIO 0.9.4, the owner's ruling of 2026-11-13: everything
  an experiment produces or consumes is saved and loaded through IO). The example (`examples/panorama_vignetting/run.jl`)
  writes its results of record as one `panorama.json` (and `panorama_compare_<model>.json` per comparison fit) with
  `VgtkIO.save_panorama`: the camera, the frames' rotations camera → world (`d = R_k·f_k`, the world the first frame's
  camera frame, zero translations, each frame's RAW file its image), the fitted V and the frames' gains. Its hand-made
  serialisation of the same results is deleted, with no parallel path: `vignetting.json` (the spline space and
  coefficients, written field by field), `vignetting_profile.csv` and `vignetting_map.csv` (V sampled on a radius and on
  the sensor). `<stem>_map.png`, V as an image, stays: a picture, not a record. The example's environment requires
  VisualGeometryToolkitIO "0.9.4".
- The guide's new section 5, "Saving the results", and a test item (`results_io_tests.jl`) that saves a fit's results,
  loads them and renders and scores the panorama with the loaded values as with the fit's own; the test environment
  gains VisualGeometryToolkitIO "0.9.4".
- The package's own `[deps]` does not gain VisualGeometryToolkitIO: its source writes no file, and VisualGeometryToolkitCore
  0.9.5's stack allow-list (`configs/stack_dependencies.toml`) does not list IO for PanoramaStitching, which the quality
  gate enforces.

## 0.6.3 (unreleased)

A patch: requires Julia 1.13 (`julia = "1.13"` in the `[compat]` of the package, of its workspace
members test and docs, and of the example environment examples/panorama_vignetting). CI runs Julia 1.13 only. No API change.

## 0.6.2 (2026-11-13)

A patch: no API change, the same results. On VisualGeometryToolkitEstimation 0.10 (`[compat]`
VisualGeometryToolkitEstimation "0.10" in the package, test and example environments), whose `fit_forward_spline` the
docs now name. Estimation 0.10.0 retired the image-space refine of a free SplineLens camera's controls;
PanoramaStitching never called it (`refine_rotations` holds its camera, and `refine_camera_and_rotations` refuses a
SplineLens camera).

- Docs: `refine_rotations`' docstring (and the forward map's comment in `sampling.jl`) name
  VisualGeometryToolkitEstimation's `fit_forward_spline(camera; tolerance)` as the way to give a held SplineLens camera its
  forward spline. They named `fit_camera(SplineLens, camera, imaged_region(camera); tolerance)`, which turns a gauge-K
  camera onto its optical axis (a new frame), so rotations estimated against the camera no longer fit it.

## 0.6.1 (unreleased)

A patch: no API change, the same results.

- On VisualGeometryToolkitFeatures 0.8 and VisualGeometryToolkitCore 0.9.4 (`[compat]` VisualGeometryToolkitFeatures
  "0.8", VisualGeometryToolkitCore "0.9.4"): the sensor image's names (`LinearRaw`, `RawMosaic`, `raw_frame_map`, `bayer_plane_map`) are
  VisualGeometryToolkitCore's, which Features 0.8 no longer re-exports. The package, its tests and the example call
  Core's; the docstrings and guides name Core as their owner.
- No hard-coded ×2 from a frame's pixels to the camera's: the frames' map (`frame_map`) places every frame pixel. The
  precompile workload's synthetic frames, the example's vignetting map image (`vignetting_map.png`), its summary's sample
  points and its camera extent (`W`, `H`), and the tests' frame-pixel centres go through it. For `:bin2` frames
  (x ↦ 2x) every value is the same, bit for bit (a doubling and the map's identity rotation and zero translation are
  exact).
- Frames with no RAW behind them take the mode's map, Core 0.9.4's `raw_frame_map(LinearRaw{:bin2})`: the precompile
  workload no longer builds a one-cell synthetic mosaic to read it, and the README's example states
  `frame_map = raw_frame_map(LinearRaw{:bin2})` (was `SimilarityMap(LinearMap(2.0I))`) and places its frame pixels through
  it; its install line no longer adds CoordinateTransformations. The README and the documentation's first page say
  PanoramaStitching is registered in General (they still named CauRegistry).
- The suite and the example's checks succeed on Julia 1.12 as on 1.13. The autocalibration items no longer pin the bundles'
  numbers bit for bit: StaticArrays' 3 × 3 products are `muladd` chains, which LLVM contracts into FMAs as each compiled
  copy and each LLVM decides (LLVM 18 and 20 one ulp apart), so the rms is checked to the LM stop's `ftol`, the estimate to
  its `xtol`, an exact scene's rms to its rounding (2eps(1200.0)), and the counts exactly. The example's stage-cache check
  no longer calls `Test.anynonpass` (not in Julia 1.12's Test).

## 0.6.0 (2026-11-12)

### Breaking: the robust-estimation contract's names (VisualGeometryToolkit 0.9)

- The feature store is VisualGeometryToolkitFeatures' `StructArray{DescribedFeature}` as the descriptor returns it
  (`detect_features`' result; `VerifiedPairs.features`, `Registration.features`): `FrameFeatures` (its `pixels`,
  `pixel_covariances` and `n_detected` kept beside the store) is gone, and so are `keypoint_covariance_scale`,
  `keypoint_uncertain` and the `bundle_whitening` keyword (refused by name). **Keypoint covariances are OFF in this
  release** (owner ruling 2026-11-12): the detector runs at its default (no `FeatureCovariance`), the pairs'
  correspondences and the bundle's observations are the keypoints' centres in the camera's pixels (through the frames'
  `frame_map`), and every residual, scale and width is in pixels. Per-keypoint covariances return with the
  heterogeneous-covariance Student-t + uniform score of the next VisualGeometryToolkitCore release.
  `pair_bearings(camera, fa, fb, matches; frame_map)` takes the two frames' described features and Features' `NNMatch`es
  (Core/Features' `csponds`, Core's `bearing_csponds`); `register_rotations(::VerifiedPairs, camera; frame_map, …)` takes
  the frames' map. A pair's `rms_px` statistic is `rms`.
- SIMFIT++ is removed (owner ruling). Each verified pair's noise scale is its RANSAC scorer's own ŝ (Core's `noise_scale`
  of the result) with the result's degrees of freedom (`VerifiedPairs.scales` holds `(sigma, dof)`, was
  `(sigma, success)`); the bundle's scale is the kept pairs' scales pooled by VisualGeometryToolkitEstimation's
  `pooled_noise_scale`, recorded as `ScaleUpstream`.
- examples/panorama_vignetting: the pairs are scored by VisualGeometryToolkitCore's `StudentTUniformScoring` (Jeffreys
  scale prior, flat label prior) at the `[registration]` key `outlier_halfwidth_px` (2120 px, half the FX3's long side);
  the MSAC `scoring_threshold` (and `scoring_threshold_px`) are refused by name. Records: verified pairs v10, rotations
  v14. The `intensity_and_position` measurement-noise arms are refused in this release (they await each pixel's position
  noise from the rotation covariance).
- The stack's one implementation replaces PanoramaStitching's copies: Core's `is_stationary(blocks, variables, free)`
  (the bundle's stall postcondition), `lens_coverage(camera, rim)` (the coverage less a rim margin), `bearing_csponds`
  and `implicit_covariance` (`refine_camera_and_rotations`' calibration covariance; its gauge-included
  `reduced_eigenvalues` is gone, `reduced_eigenvalues_gauge_fixed` stays), Core's `restrict` of a `ParameterManifold`
  (`fit_vignetting`'s V increment) and `propagate`; Estimation's `bundle_loss` refuses an `Irls` bundle strategy.
- **Accuracy tolerance (owner ruling 2026-11-12).** Against 0.5.1's line (f0c711c) on the suite's and the README's
  synthetic sweeps, with ground truth: every kept observation is a true inlier (0.99 before) and the true inliers' pixel
  rms is 0.23–0.26 px (0.35 px before), while the rotation error moved from at most 0.0037° / 0.0067° to at most 0.0055°
  / 0.0078° (Student-t pairs). The release gate accepts rotation errors up to 0.01° on both scenes. On the example's real
  sets the rotations move by up to 0.18° (14 mm) and 0.06° (rooftop) from 0.5.1's.
- The termination record (the contract's §14): `RotationBundleMetadata`'s `outcome` is gone; the bundles' and
  `fit_vignetting`'s results carry VisualGeometryToolkitCore's termination record (the `FitResult`'s `ReturnCode.T` and
  `StopRule`, read by `retcode`/`stop_rule`; the legacy Symbol `retcode_symbol(result)`). NLLSsolver's step-size or rejection stop is `Success` only where the robust cost is stationary
  at the result (Core's `is_stationary` at Estimation's `[chart_lm] stationarity_tol`), `Stalled` (retcode `:stalled`)
  otherwise (`:stalled`), where it read `:converged` before; `max_iter = 0` (was outcome `:not_run`) is `MaxIters` with `IterationCap`.
  `fit_vignetting` records the worst termination over every linear solve of every plane (Core's `worst_retcode`) instead
  of the last solve's: a CG capped in an earlier reweighted solve now reads `MaxIters` (`:max_iterations`). A pair's
  RANSAC, and its SIMFIT++ refits, are kept when they found a model (Core's `found_model`), Core's `converged` being
  strict now. The example's rotations.json is format v11 (`bundle_return_code`, `bundle_stop_rule`; `bundle_outcome`
  retired).
- The rotation bundles take their loss from VisualGeometryToolkitEstimation's `bundle_loss` (the one implementation the
  stack's bundles share; PanoramaStitching's own copy is gone): its refusals are theirs, so a `width` beside a strategy
  that states an efficiency constant is an `ArgumentError` (give `Triggs{Shape}()` with a width), and the width is the
  deployed loss's (`sqrt(sq_loss_width(loss))`). An `Irls` strategy is refused by the bundles (one LM run).
- The view graph's averaging calls VisualGeometryToolkitEstimation's `fit_filtered_rotation_averaging` (was
  `filtered_rotation_averaging`, renamed by the stack: a fit verb).
- The bundles' default `optimization_strategy` is VisualGeometryToolkitEstimation's pixel bundle's, read from its
  configs/default.toml `[refine_camera_and_poses]` (compliance audit #48; was a literal of the same strategy):
  `Triggs{WelschShape}` stating no efficiency constant, its width Core's `efficiency_constant(Welsch, 2)` × `scale`, as
  every refine reads it. Same numbers.
- examples/panorama_vignetting runs on VisualGeometryToolkitEval's Experiments (compliance audit #21): everything it writes
  goes under the results root (`<results>/panorama_vignetting/<name>/`, the run's outputs in `run-<run_key(config)>/`
  with Eval's run records) instead of the config's `output` folder (the key is refused by name); the stages' records are
  Eval's keyed result cache (`cached_result` at `run_key`: `verified_pairs_<key>.jld2`, `registration_<key>.jld2`) instead of a
  private key and JSON cache; the READY_14mm.txt marker is gone; `reproduce.toml` declares the stage graph. Its checks
  (`stage_cache_check.jl`, `autocalibration_check.jl`) run through `run_checks.jl` in the example's environment.
- The view graph's averaging runs on Uncertain edges (the owner's ruling): each verified pair's edge is
  VisualGeometryToolkitEstimation's `uncertain_rotation` (its scorer's `noise_scale`² × the implicit-function covariance:
  absolute, rad²), so the averaging takes no scale and records scale 1, `ScaleUpstream`; edges sharing a keypoint are
  treated as independent (first order). A pair whose covariance is not determined is no edge, listed in
  `Registration.no_covariance` and the log; a drop that splits the frames is an `ArgumentError` naming the pairs.
  `VerifiedPairs.verified` entries carry `covariance`. The example's verified pairs are format v6, rotations.json records
  the pairs with no covariance and the averaging's scale record.
- `refine_rotations` takes the robust-estimation contract's `optimization_strategy` (a `Triggs{Shape}`, by default every
  refine's `Triggs{WelschShape}`; `nothing` the plain least squares) and exactly one of `scale` or `width`; its `loss` and
  `loss_width` are refused by name. `RotationBundleMetadata` records the scale (`scale`, `scale_source`: read through
  Core's `noise_scale`/`scale_source`). `register_rotations`' and `verify_pairs`' `averaging_loss`, `bundle_loss` and
  `bundle_efficiency_constant` are `averaging_optimization_strategy` (an `Irls`; by default Estimation's
  `[fit_rotation_averaging]` strategy) and `bundle_optimization_strategy` (a `Triggs`, run at the pooled pair scale);
  the old keywords are refused by name. Same numbers: the bundle runs at the same width.
- examples/panorama_vignetting: the `[registration]` keys `averaging_optimization_strategy` and
  `bundle_optimization_strategy` are strategy tables read by Estimation's `optimization_strategy_from_toml` (the private
  string → loss table is gone); `averaging_loss`, `bundle_loss` and `bundle_efficiency_constant` are refused by name.
- `RotationBundleMetadata.calibration.free` is `free_parameters`.
- `register_rotations`' `bundle_scale_multiple` is the efficiency constant of its `bundle_optimization_strategy` (the same
  plain multiple of the pooled scale; results unchanged, above); the pairs' RANSAC strategy keyword of `verify_pairs`/`register_rotations` is
  `local_optimization_strategy` (was `local_optimization`), the stack's find_* keyword. The old keywords are an
  `ArgumentError` naming the new ones. The view graph's averaging and the pairs' SIMFIT++ refits take Estimation's
  `optimization_strategy` (`nothing` for the plain least squares, no `NoLocalOptimization`).
- examples/panorama_vignetting: the `[registration]` key `local_optimization_strategy` (the old one refused by name, as are
  `bundle_scale_multiple` and `bundle_camera_free`); `rotations.json` format v10 (the bundle's strategy and its scale
  record, `bundle_scale_px`, in place of `bundle_scale_multiple`).

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
- The example's decode cache is VisualGeometryToolkitFeatures' (`load_raw(path; cache)`, a JLD2 `FileCache` at
  `out/<name>/decoded/linear_raws.jld2`, one entry per RAW content and decode settings); its code-sum PNGs, `decode.json`
  and their helpers are deleted (a cache of an earlier run is not read: the frames decode once more). JLD2 joins the
  example's environment. `decoded_frames` no longer returns `levels`.
- The fit's zero arrays on its backend are VisualGeometryToolkitCore's `device_zeros` (the private `_device_zeros` is deleted).
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

### New: `refine_camera_and_rotations` — the autocalibration of a rotating camera

- `refine_camera_and_rotations(camera, rotations, tracks; optimization_strategy = Triggs{WelschShape}(…), scale | width,
  fixed = ())` refines a parametric camera with the rotations and the track directions, in the same bundle as
  `refine_rotations` (one internal bundle, the camera variable present or absent: VisualGeometryToolkitCore's
  `ParameterManifold` of the camera's model restricted to its free parameters, the camera block of Estimation's
  `refine_camera_and_poses`; its Jacobian ForwardDiff through the chart, the frame and direction blocks the closed form).
  No target: the first frame's rotation is the one gauge. A contract refine: a `Triggs{K}` strategy, exactly one of
  `scale`/`width`, and `fixed` naming the HELD parameters (names or groups, VisualGeometryToolkitCore's
  `free_parameter_names`; the skew always held). A SplineLens camera is refused (its parameters are its lattice's
  controls, Estimation's own spline route).
- `refine_rotations` is the rotation-only bundle: it holds the calibration and takes no parameter keyword; its
  `camera_free` (briefly unreleased) is refused, pointing at `refine_camera_and_rotations`, and `register_rotations`'
  `bundle_camera_free` likewise. Its numbers are unchanged, bit for bit.
- "Self-calibration" (of the camera, no target) is "autocalibration" throughout (the tests: `autocalibration_tests.jl`);
  the vignetting's self-calibration (V from the frames) keeps its name.
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
