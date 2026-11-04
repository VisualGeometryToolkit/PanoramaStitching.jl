# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

PanoramaStitching: the panorama of a camera that only rotated, through a calibrated lens of any field of view — the rotations of
its frames (`register_rotations`: ASIFT + DeDoDe + LoMa features, `find_rotation` per frame pair, the view graph's robust rotation
averaging, the tracks and the rotation and direction bundle), the frames sampled at directions of the sphere
(`sample_frames`) and the panorama in any view (`render_panorama`, `render_cube_map`), both on Core's warping machinery
(KernelAbstractions: GPU arrays or the host, one code path; no per-pixel or per-direction host loops), and the camera's
vignetting from the frames alone (`fit_vignetting`, interpolation free: every pixel at its centre, the scene a spline on the
sphere fitted with V) with the seams it leaves (`score_seams`). A package of the VisualGeometryToolkit stack; its only
radiometry is that vignetting fit. Saturation masks measurements (`sample_frames`, `fit_vignetting`, `score_seams`), never a
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

- `src/features.jl` — per frame ASIFT keypoints and DeDoDe descriptions, per pair the LoMa matches as bearings with their
  back-projection Jacobians.
- `src/registration.jl` — `register_rotations` and its `Registration`.
- `src/view_graph.jl` — `ViewGraph` (Graphs.jl) and `chain_tracks` (DataStructures.jl union-find).
- `src/rotation_bundle.jl` — `refine_rotations`: NLLSsolver, one `SO3Manifold` per frame and one `SphereManifold` per track.
- `src/sampling.jl` — `sample_frames` and `fibonacci_directions`; the camera's forward map, the frame region and masks the
  render shares.
- `src/panorama.jl` — `panorama_view`, `render_panorama`, `render_cube_map`.
- `src/vignetting.jl` — `fit_vignetting`: the spline spaces (`RadialVignetting`, `TensorVignetting`), the scene's
  `AzimuthalChart`, the scatter-add kernels of the normal equations (KernelAbstractions + Atomix), device CG or host
  Cholesky (SparseArrays).
- `src/seams.jl` — `score_seams`: the seam ratio per raster pixel through `sample_frames`.
- `examples/panorama_vignetting/` — the RAW-to-corrected-panorama driver (`run.jl`) and its configs of record; its own
  environment (`Project.toml`, not a workspace member: CUDA and LibRaw_jll).

## Git Commit Conventions

- Commit messages end with the co-author line the owner's repositories use; no "Generated with" footer.

## Stack Conventions

The conventions every stack package follows — naming and the estimator verbs, signatures, data arguments, frames, pixel origins,
bearings, Jacobians, units and the stack contracts — live once, in VisualGeometryToolkitCore's `conventions/` folder. Read them
before changing code here; this file carries no copy. Where the folder is:

- in a checkout of the stack's monorepo: `VisualGeometryToolkitCore/conventions/`;
- from Julia, the installed copy: `VisualGeometryToolkitCore.conventions_dir()`.

How they land here: `register_rotations` is a task verb on images (it composes the estimators — `find_rotation`,
`fit_rotation_averaging`, `refine_rotations` — and does no estimation of its own); every residual is in the camera's pixels
(bearings carry their back-projection Jacobians); the common frame is the first frame's camera frame.
