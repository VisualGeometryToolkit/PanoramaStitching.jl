# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

PanoramaStitching: the panorama of a camera that only rotated, through a calibrated lens of any field of view — the rotations of
its frames (`register_rotations`: ASIFT + DeDoDe + LoMa features, Estimation's `Ransac` on each frame pair's `RotationProblem`, the view graph's robust rotation
averaging, the tracks and the rotation and direction bundle), the frames sampled at directions of the sphere
(`sample_frames`) and the panorama in any view (`render_panorama`, `render_cube_map`), both on Core's warping machinery
(KernelAbstractions: GPU arrays or the host, one code path; no per-pixel or per-direction host loops), and the camera's
vignetting from the frames alone (`VignettingSelfCalibration`, interpolation free: every pixel at its centre, the scene a spline on the
sphere fitted with V) with the seams it leaves (`score_seams`). A package of the VisualGeometryToolkit stack; its only
radiometry is that vignetting fit, on VisualRadiometryToolkitCore's vignetting models. Saturation masks observations (`sample_frames`, `VignettingSelfCalibration`, `score_seams`), never a
render.

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'                                   # every item (ReTestItems)
julia --project=. -e 'using Pkg; Pkg.test(; test_args = ["bundle|sample_frames"])'   # items whose name matches
julia --project=docs --startup-file=no docs/make.jl
```

One Pkg workspace (`[workspace] projects = ["test", "docs"]`), one untracked Manifest at the root. The stack packages resolve
from the registries (CauRegistry; General for those it holds).

## Layout

- `src/features.jl` — per frame ASIFT detections and DeDoDe descriptions, per pair the LoMa matches as bearings with their
  back-projection Jacobians.
- `src/registration.jl` — `register_rotations` and its `Registration`.
- `src/tracks.jl` — `chain_tracks` (the components of the match graph, VisualGeometryToolkitEstimation's `ViewGraph`). The view
  graph of the verified pairs is Estimation's `ViewGraph` (registration.jl's `_view_graph`): PanoramaStitching has none of its own.
- `src/rotation_bundle.jl` — `RotationBundleProblem`, a block problem (Core's `AbstractBlockProblem`) refined by Core's
  `LevenbergMarquardt`: `RotationBundleProblem(camera, tracks)` refining the rotations (the calibration held) and
  `RotationBundleProblem(tracks; fixed)` refining `(; camera, rotations)` (autocalibration: the camera refined too, `fixed` the
  held parameters), one shared bundle (`block_refit`): NLLSsolver, one `SO3Manifold` per frame and one
  `SphereManifold` per track.
- `src/sampling.jl` — `sample_frames` and `fibonacci_directions`; the camera's forward map, the frame region and masks the
  render shares.
- `src/panorama.jl` — `panorama_view`, `render_panorama`, `render_cube_map`; a sweep's up direction, `RollFreeUp` (a closed form on
  `EstimationProblem(SweepUp, rotations)`).
- `src/vignetting.jl` — `VignettingSelfCalibration` (a fit on `EstimationProblem(AbstractVignetting, frames, p)`): the spline spaces (VisualRadiometryToolkitCore's `RadialVignetting`, `TensorProductVignetting`), the scene's
  `AzimuthalChart`, the scatter-add kernels of the normal equations (KernelAbstractions + Atomix), the scene penalty Core's
  `difference_rows`, device CG (Core's `normal_cg`) or host
  Cholesky (SparseArrays).
- `src/seams.jl` — `score_seams`: the seam ratio per raster pixel through `sample_frames`.
- `src/pair_vignetting.jl` — the model-free vignetting of a sweep, the pair equation on rings of a field angle:
  `ring_correspondences`, `RingPairMedians` (on `RingVignetting`), `LevelBisection` (on `AdditiveLevel`). The saturation test it
  reads, `VignettingSelfCalibration`'s and
  `score_seams`' alike, is one (`_sweep_saturation` in vignetting.jl: the masks dilated and symmetric over the frames).
- `src/panorama_record.jl` — `save_panorama`, `load_panorama`: the results as a `vgio/panorama/v2.0` file. The record is this
  package's (`PANORAMA_KIND`, a VisualGeometryToolkitIO `KindSpec`) and holds only the parts' list, the frames' gains and the
  rules that tie them; every part is written by its owner's kind (VisualGeometryToolkitIO's camera, trajectory and fit records,
  VisualRadiometryToolkitIO's vignetting body). Its JSON Schema is docs/src/schemas/vgio/panorama/v2.0.json, held to the
  generator's output by a test.
- `examples/panorama_vignetting/` — the RAW-to-corrected-panorama driver (`run.jl`) and its configs of record; its own
  environment (`Project.toml`, not a workspace member: CUDA and LibRaw_jll).

## Git Commit Conventions

- Never add "Co-Authored-By: Claude" or any other AI attribution to a commit message.
- Never add a "Generated with Claude Code" footer.

## Stack Conventions

The conventions every stack package follows — naming and the estimator verbs, signatures, data arguments, frames, pixel origins,
bearings, Jacobians, units and the stack contracts — live once, in VisualGeometryToolkitCore's `conventions/` folder. Read them
before changing code here; this file carries no copy. Where the folder is:

- in a checkout of the stack's monorepo: `VisualGeometryToolkitCore/conventions/`;
- from Julia, the installed copy: `VisualGeometryToolkitCore.conventions_dir()`.

How they land here: every estimator is `solve(problem, algorithm)` (Core's front end; this package adds `__solve` methods for
its own algorithms and `block_refit` for its block problem, never a `solve` method); `register_rotations` is a task verb on
images (it composes the estimators — `Ransac`, `FilteredRotationAveraging`, `PooledScale`, the rotation bundle's
`LevenbergMarquardt` — and does no estimation of its own); every residual is in the camera's pixels
(bearings carry their back-projection Jacobians), whitened by the detections' covariances (the detector's, Features'
described features the one feature store); the common frame is the first frame's camera frame.
