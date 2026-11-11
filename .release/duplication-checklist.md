# PanoramaStitching: duplicate removals of the 0.9 cascade

Owner decision 12 (2026-11-11): each item is REQUIRED in the release leg; it is closed by deleting PanoramaStitching's copy in
favour of the stack's one implementation, and ticked with the commit that closes it. Source:
cascade-duplicate-removals.md ("PanoramaStitching"), duplication-audit.md, plane-maps-report.md; the owner's ruling (all
duplicates gone, all common code hoisted) and VisualGeometryToolkitCore's `test_duplication` over the package.

## The cascade's list

- [x] _pcg → Core's normal-form CG (`normal_cg`, monorepo 5305c2e4) (closed by 8f2538b)
- [x] _penalty_kernel! → Core difference_rows (closed by 8f2538b)
- [x] its ViewGraph → Estimation's ViewGraph (closed by 69471ff, already in cda8bf7: src/view_graph.jl deleted, registration.jl's `_view_graph` and tracks.jl build Estimation's `ViewGraph`)

## Further copies (duplication-audit.md, plane-maps-report.md, test_duplication)

- [x] _upload → Core's `ka_upload` / `on_device` (audit finding 17) (closed by 8f2538b)
- [x] the planes' sample offsets and the frames' binning scale (`_plane_offsets`, the scale keyword, the hand-written offset subtraction, `_SensorPoint`'s scale) → Features' `raw_frame_map` / `bayer_plane_map` and Core's multi-plane `image_warp!` (closed by 8f2538b; its device fix 7e1ed90)
- [x] the example's sRGB encoder (`srgb`, `save_srgb`'s clamp and rounding) → Features' `srgb_image` (audit finding 16) (closed by 8f2538b)
- [x] the example's image viewer (compare_makie.jl's menu, linked axes, `rotr90` orientation) → Features' `compare_images` (closed by 8f2538b)
- [x] the detections' own sort in `detect_features` → Features' canonical detection order (ssc-anms 05c91203) (closed by 8f2538b)
- [x] test/docs_gates.jl (the docstring gates, copied) → Core's `test_docs` (closed by 8f2538b)
- [x] the example's `data_root` → VisualGeometryToolkitEval's `data_root` (closed by 8f2538b)
- [x] the names an upstream package owns: `_PlanePoint` (Estimation) folded into `_AtPixelCentre` with `_RayAt`; `_measurement_covariance` (Core) renamed `_noise_covariance` (a different concept: Σ_x of `IntensityAndPositionNoise`); `_zeros` (Core.Cameras' `_zeros(n)`, a tuple of zeros: another concept) renamed `_device_zeros` (closed by 8f2538b)

## Hoists into the stack

- [x] hoist the decoded-RAW cache (examples/panorama_vignetting/run.jl `decoded_frames`, `encode_codes`, `decode_codes`, `save_codes`, `load_codes`, `save_mask`, `load_mask`, `plane_blacks`; audit finding 4) into the stack: owner VisualGeometryToolkitFeatures' raw module (the "one RAW image path" rule) with the file writing in VisualGeometryToolkitIO. Proposed: `VgtkFeatures.cached_linear_raws(raw_paths, cache_dir; mode, gain) -> (; images, saturated, raws, levels, daylight_wb)` (one decode per RAW keyed by its SHA-256 and the decode settings; the lossless per-quad code-sum PNG layout, or RC's linear-PNG layout once one is agreed), its first frame's `LinearRaw` (or a mosaic-free header) for `raw_frame_map`/`bayer_plane_map`. Callers: run.jl `decoded_frames` (run_example, compare_arms.jl). The stack lacks it today, so the example keeps its cache until the hoist lands; the switch is the body of `decoded_frames`. test_duplication does not flag it (no restated body: RC's copy, cvpr2027/src/drivers.jl, is another layout). — CLOSED: the stack had the concept, Features' one persistent cache (`FileCache`, JLD2) and its one RAW path: `load_raw(path; mode, gain, decoder, cache)` decodes each RAW once per content (SHA-256) and settings (monorepo ca3c6f63); the example's `decoded_frames` calls it and its code-sum PNG cache and helpers are deleted (663c30a). Nothing new in IO: the FileCache writes the file.
- [x] hoist a zero array on a KernelAbstractions backend, `fill!(KA.allocate(dev, T, dims...), zero(T))` (KernelAbstractions' own `KA.zeros` is not public, so ExplicitImports refuses it across the package boundary): owner VisualGeometryToolkitCore, src/base/device.jl beside `ka_upload`/`on_device`. Proposed: `VgtkCore.device_zeros(T, device, dims...) -> array` (public). Callers here: src/vignetting.jl `_device_zeros` (the fit's accumulators); in the stack the same `fill!(KA.allocate(…))` form is written in Core's `cgls` (penalized_least_squares.jl:170) and Features' matching_ka.jl:249–250. The switch is the one definition line: `const _device_zeros = (be, T, dims...) -> VgtkCore.device_zeros(T, be, dims...)`, or the call sites renamed. — CLOSED: VisualGeometryToolkitCore `device_zeros(T, device, dims...)` (public, base/device.jl; Core's cgls, B-spline evaluation and ChainRulesCore extension and Features' spread selection call it too) (monorepo 8de8e2a6); `_device_zeros` deleted, the call sites renamed (663c30a)

## CPU/GPU twins (the owner's one-kernel rule: a GPU branch's CPU branch is the same KernelAbstractions kernel on the CPU backend)

Source: cpu-gpu-twin-audit.md ("PanoramaStitching"); VisualGeometryToolkitCore's `check_duplication` `:backend_branch` / `:twin_name` rules (vgtk-hoists d3a02dbf).

- [x] P1: the Cholesky arm's host product `_apply_host` (BLAS/sparse) vs the CG arm's `apply!` closure (stencil kernel, CSR `mul!`, tall transposed product) → ONE operator, `_NormalOperator` with its `mul!`, built on the fit's backend; Core's `normal_cg` applies it and the Cholesky arm reads its relative residual through it; `_apply_host` deleted. Agreement: the operator against the old host product on a random system (nL = 1600, m = 9) 3.8e-16 (CPU) / 2.8e-16 (CUDA) relative; whole fits old vs new (radial V, intensity gains) control points within 1.4e-12 (Cholesky) and 3.2e-10 / 1.2e-8 (CG, CPU / CUDA), the old code's own run-to-run spread (1.3e-12 / 1.2e-8: atomic accumulation order) (closed by 90ac698)
