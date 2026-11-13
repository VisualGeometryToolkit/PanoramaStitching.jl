# PanoramaStitching 0.6.4: the results' serialisation through IO (owner, 2026-11-13: no parallel path)

- [x] the example's hand-made V record (run.jl `write_vignetting`: vignetting.json written field by field; vignetting_profile.csv and vignetting_map.csv, V sampled) → VisualGeometryToolkitIO's `save_panorama` (vgio/panorama/v1: the camera, rotations, V and gains in one record) (36165d4)
- [x] check_duplication over the package and its tests: the new test's V fall-off is not a restated body of vignetting_tests' `radial` (657c35b)
- [x] check_duplication against Core 0.9.5 flags `_AtPixelCentre`'s one-field-wrapper idiom beside Core's `Eliminable` and `Triggs`: different concepts, recorded in DUPLICATION.toml with its reason, as the 0.7 line does (0072826). The gate on 4728e7d: 0 findings

---

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

## Robust schema

### The Student-t keyword (owner ruling 2026-11-11: "there is no threshold for the Student-t, only the outlier half-width")

Every scorer takes its own keyword: `StudentTUniformScoring` its `outlier_halfwidth` (and its priors), never `threshold`; the
threshold scorers (RANSAC, MSAC, the loss-width families) `threshold`, never `outlier_halfwidth`. No number may cross between
the two meanings. Audit: studentt-keyword-overload.md.

A pair's rotation is scored by the Student-t + uniform score at `[registration] outlier_halfwidth_px` (examples, 0.6.0;
the MSAC threshold key retired) and, with `scoring = nothing`, by Estimation's default scorer.

- [x] examples/panorama_vignetting/README.md:73 "MSAC scoring with a 5 px half-width" (a threshold called a half-width) → "MSAC scoring at a 5 px threshold" (this commit)
- [x] test/registration_tests.jl "the rotation's scorer: Student-t + uniform at its outlier half-width, no threshold": the configs' `outlier_halfwidth_px` reaches the scorer exactly (`===`), MSAC refuses `outlier_halfwidth`, the Student-t scorer refuses `threshold` (5262be4)

## The 0.9 concept-level duplication audit (2026-11-12; ~/logs/dgx-agent/dup-audit-ps.md)

- [x] the keypoint store split (`FrameFeatures`' pixels and covariances beside Features' described features) → Features' `StructArray{DescribedFeature}` (35fa078…be86989)
- [x] the frame map's hand-written J Σ Jᵀ, `keypoint_covariance_scale`, `keypoint_uncertain`, `_valid_covariance` and `bundle_whitening` → deleted (covariances off in 0.6.0; Features' `propagate(map, detection)` when they return)
- [x] the pixel → bearing loop of `pair_bearings` → Core's `bearing_csponds` (5cf699f)
- [x] the bundle's hand-built stall check → Core's `is_stationary(blocks, variables, free)` (5cf699f)
- [x] `_rim` and the example's `domain_radius` margin → Core's `lens_coverage(camera, rim)` (5cf699f)
- [x] `fit_vignetting`'s P₀ + Z y increment → Core's `restrict` of V's `ParameterManifold` (bit-identical; e4959a2)
- [x] `_autocalibration`'s Schur reduction and s²H⁻¹ → Core's `implicit_covariance` (f7682cf); its T Σ Tᵀ → Core's `propagate`
- [x] SIMFIT++ (`_pair_scale`, `_chisq2_quantile`) and `_pooled_scale` → deleted; the scorer's ŝ pooled by Estimation's `pooled_noise_scale` (77efcd1)
- [x] the bundles' Irls refusal → Estimation's `bundle_loss` (b58c29d)
- [x] the example's run directory, cache path rule and private arms key → Eval's `keyed_cell_dir`, `cached_result`'s file and `run_key(cfg; files)` (e4959a2)
- [x] deferred by the owner's rulings (not duplicates of a stack implementation today): `_check_tolerances` (Estimation has no public validator), `level_up` vs `up_frame` (no bit-identical rewire), the tracks as Features' `CspondSet` (a public type change), `_in_frame` (one copy, no stack home)

## 0.6.1 (Features 0.8, Core 0.9.3 / 0.9.4)

- [x] the hard-coded ×2 from frame to camera pixels (run.jl's vignetting map image and summary points, its and
  compare_arms.jl's camera extent `2Wf, 2Hf`, precompile.jl's frame pixel centres, registration_tests.jl's `2 .* p`,
  vignetting_tests.jl's three `(j - 0.5) * 2` grids) → the frames' map (`frame_map`; the extent through the package's own
  `_covered_extent`) (d210ffc)
- [x] the precompile workload's one-cell synthetic mosaic, read only for its frame map, and the README example's
  `SimilarityMap(LinearMap(2.0I))` → Core 0.9.4's `raw_frame_map(LinearRaw{:bin2})` (15b8e50)
- [x] Core 0.9.3's public `require_extension`: PanoramaStitching has no extension check of its own (no
  `_require_extension`, `_extension_loaded`, `Base.get_extension`); nothing to switch
- [x] Core 0.9.4's `is_inside_texel_centres`: PanoramaStitching restates no texel chart (it samples through Core's
  warping); nothing to switch
- [x] `check_duplication(PanoramaStitching)` against vgtk 9762b9b4 (Core 0.9.4), BlobBoards bee3b43, lodr-09, rsdr-09,
  CaliBlobs cb-release: 0 findings (Slurm 10548)

## 0.6.2 (on VisualGeometryToolkitEstimation 0.10)

Nothing to delete: the release widens one compat entry; the stack duplication audit lists no item for PanoramaStitching at
this release (its open items: none). `check_duplication` runs in the suite.

## 0.6.3 (Julia 1.13)

A compat-only patch: no source, test or paper code is added, so nothing new can duplicate the stack.

- [x] 0.6.3: no new duplicate (the diff since the registered release adds no code; `check_duplication(PanoramaStitching)` re-runs in the fresh-clone gate of this commit)
