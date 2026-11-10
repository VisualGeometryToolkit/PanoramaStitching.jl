# PanoramaStitching on the 0.9 camera (plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PanoramaStitching reads its camera's limits from the camera itself (`imaged_region(camera)`) and projects grid lenses through the camera's own `SplineLens`. That fixes Ties's issues 1–3 at the root and deletes the per-consumer `circle` and `projection` plumbing.

**Architecture:**
- Every entry point takes a `BoundedCamera`. Its `imaged_region` (sensor ∩ lens coverage ∩ valid domain) replaces the `circle` keyword and the config's `[inputs] circle`. Its `project` replaces the `projection = SplineProjection` keyword: a SplineLens with a forward spline projects through it, and gives NaN outside the fitted chart.
- The bundle drops (and reports) observations whose start prediction is not imaged, instead of refusing to start.
- Rendering and sampling accept a pixel only inside `imaged_region`.

**Tech Stack:** Julia; PanoramaStitching.jl (branch `lens-coverage` from origin/main 0.5.1); VisualGeometryToolkit Core/IO/Estimation 0.9 from the monorepo worktree `~/src/vgtk-bounded-camera-region` through a TEMPORARY `[sources]` (cau-release's downstream rule: never pushed or registered while it is there).

**Spec:**
- Ties's report `panorama_issues_for_james.md`, 2026-11-08: issues 1–3. Issue 4 is calibration, not this repo.
- The owner-agreed camera design, plan 1 (`lens_coverage`, `imaged_region`) and plan 1b (`SplineLens`, the forward spline, ruling (A)) in the monorepo's `docs/superpowers/plans/`.

## Global Constraints

- **No duplication (owner rule, extreme vigilance).** Delete the `circle`/`projection` keywords, `_forward_map(::SplineProjection)`, `_images(::CentralGenericModel)` and every local workaround that the camera now covers. No aliases. Every task has a duplication gate before its commit, against PanoramaStitching and the monorepo: any helper that re-derives what `imaged_region`, `project` or `fit_camera` already give is deleted.
- **Latest only:** compat `VisualGeometryToolkitCore = "0.9"`, `VisualGeometryToolkitIO = "0.9"`, `VisualGeometryToolkitEstimation = "0.9"`. The version becomes 0.6.0 at release (breaking), not in this plan.
- Commits end with the session's Co-Authored-By line (this repo takes it).
- Config files: `[inputs] circle` is REFUSED with a message that names the camera's lens coverage. There is no silent ignore.
- The naming check bans "pass" and its forms.

## Review Focus

1. **A grid camera whose coverage is the whole sensor** (Tamron, no domain stored). Renders stay bounded by the lens's valid domain: the cylindrical panorama height is within 10% of the parametric run's 1056 px, not 2481.
2. **A start prediction past the imaged region.** It is dropped and counted in the result (`n_dropped_start`), and the bundle runs on the rest. If every observation is dropped, it is an error, not a silent empty solve.
3. **A feature detected outside `imaged_region`** (detector margin). It is never kept: features are cropped to `imaged_region` eroded by `detector_margin_px`.
4. **A parametric camera** (`lens_kind = "cameras"`) with no stored coverage. It behaves exactly as 0.5.1 did with a whole-sensor circle (regression: same rotations to 1e-9).
5. **GPU render with a SplineLens forward spline.** It equals the CPU render to 1e-4 relative.

---

### Task 1: The 0.9 stack, and `imaged_region` in place of `circle`

**Files:**
- `Project.toml` (and its test/docs members): compat, TEMPORARY `[sources]`.
- `src/features.jl`, `src/registration.jl`, `src/panorama.jl`, `src/seams.jl`, `src/vignetting.jl`, `src/sampling.jl`, `src/precompile.jl`: delete the `circle` keyword and its threading, and use `imaged_region(camera)`.
- `examples/panorama_vignetting/run.jl` and the configs: `[inputs] circle` is refused; `load_lens` returns a `BoundedCamera`.
- `README.md`, `docs/src/guide.md`, CHANGELOG (Unreleased, breaking).
- Tests: `test/registration_tests.jl` and the vignetting/panorama tests that pass a circle.

**Interfaces:** the entry points become
- `register_rotations(frames, camera::BoundedCamera; raw_scale, tone_percentile, detector_margin_px, …)`
- `render_panorama(camera::BoundedCamera, frames, rotations, view; vignetting, …)`
- `score_seams(camera::BoundedCamera, …)`
- `fit_vignetting(vignetting, camera::BoundedCamera, …)`

There is no `circle` keyword. Every mask reads `imaged_region(camera)`.

- [ ] **Step 1: Write the failing tests.** Each test builds `BoundedCamera(kb, Size2(W, H); lens_coverage = Circle(...))`. Then:
  - `register_rotations` keeps no feature outside `imaged_region` eroded by the margin;
  - `render_panorama` samples no pixel outside it;
  - a config with `[inputs] circle` throws an ArgumentError whose message contains "lens_coverage".
- [ ] **Step 2: Run them.** Expected: FAIL. The `circle` keyword is still required.
- [ ] **Step 3: Implement.** Delete the keyword everywhere; there is one masking helper, at the call sites.
- [ ] **Step 4: Run the full suite.** Expected: green.
- [ ] **Step 5: Duplication gate, then commit.** Message: `"PanoramaStitching: the camera's imaged_region replaces every circle argument (VisualGeometryToolkit 0.9)"`.

### Task 2: Grid lenses as `SplineLens` cameras; the `projection` keyword is gone

**Files:**
- `src/sampling.jl:25-32`: delete `_forward_map(::SplineProjection)`. The device form is the camera's `imaging_transform`, adapted.
- `src/registration.jl`, `src/rotation_bundle.jl:48-61, 141-142, 176`: delete the `projection` keyword and `_images(::CentralGenericModel)`. Keep one `_images(camera::BoundedCamera, u) = Point2(u) ∈ imaged_region(camera)`. Superseded by ruling (Task 1, fix round 1): the bundle masks with `valid_domain(model(camera))`, not `imaged_region` (done in Task 1).
- `examples/panorama_vignetting/run.jl:156-172`:
  - `lens_kind = "grid"` loads with `VgtkIO.load(BoundedCamera, path)`. A v2/v3 grid migrates; its domain becomes the coverage.
  - The forward spline is fitted once with `fit_camera(SplineLens, camera, imaged_region(camera); tolerance = spline_tolerance_px)` and kept in the camera.
  - `lens.forward`/`lens.camera` collapse into one camera.
- Docstrings: registration.jl:58,96; panorama.jl:119-120,199; seams.jl:81; vignetting.jl:607. Also `docs/src/guide.md:32-42`.
- Tests: `registration_tests.jl:9-12, 37, 54, 59, 265-275`, `rotation_bundle_tests.jl:103-104, 127`.

- [ ] **Step 1: Write the failing tests.**
  - A `CameraModel(SplineLens, kb, (W, H))` camera with a fitted forward spline registers and renders. The results equal the old SplineProjection path's within the fit tolerance; reference them to the 0.5.1 numbers in `registration_tests`.
  - Review Focus 1: a whole-sensor SplineLens camera renders a cylindrical panorama no taller than the parametric one + 10%.
  - Review Focus 5: GPU vs CPU render (skipped without CUDA).
- [ ] **Step 2: Run them.** Expected: FAIL (`SplineProjection` is gone).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full suite.** Expected: green.
- [ ] **Step 5: Duplication gate, then commit.** Message: `"PanoramaStitching: a grid lens is a SplineLens camera; the projection keyword and SplineProjection are gone"`.

### Task 3: The bundle drops non-imaged start predictions (issues 1 and 2)

**Files:** `src/rotation_bundle.jl` (≈ lines 200-240: the start's non-finite count and the `:nonfinite_start` outcome); the result struct gains `n_dropped_start::Int`. Tests: `rotation_bundle_tests.jl`.

Supersedes the WIP branch `spline-lens-guards` (86e1707). Read its diff for intent, re-do it on this branch, and delete the WIP branch after the owner's go-ahead.

- [ ] **Step 1: Write the failing test.** A synthetic bundle in which 5 tracks' start predictions fall outside `imaged_region`:
  - `outcome == :converged`;
  - `n_dropped_start == 5`;
  - the rotations equal a solve without those tracks to 1e-9.
  - When all tracks are dropped: an ArgumentError.
- [ ] **Step 2: Run it.** Expected: FAIL with `:nonfinite_start`.
- [ ] **Step 3: Implement.** Drop the observation (not the whole track) where the start prediction is not finite. Report the count in the result and in the example's `summary.md`. `:nonfinite_start` is retired.
- [ ] **Step 4: Run the full suite.** Expected: green.
- [ ] **Step 5: Duplication gate, then commit.** Message: `"PanoramaStitching: the rotation bundle drops (and counts) observations whose start prediction is not imaged"`.

### Task 4: Reproduce Ties's runs

- [ ] Run the example on Ties's three spline configs (`configs/{14mm,fisheye,tamron}_spline.toml`, with `circle` removed) with his `data_root`. The configs are in the session scratchpad `ties-panorama-issues/x/panorama_issues_2026-11-08`. This needs the RAW frames from the Hugging Face dataset `panorama_good/`.
  - Expected: the 14 mm and fisheye bundles converge.
  - Expected: the Tamron cylindrical panorama is about 1056 px tall, not 2481.
  - Record the bundle rms, the drop counts and the panorama sizes in the CHANGELOG entry and in a short note for Ties.
- [ ] The 14 mm config of record: a small example driver derives a vgio/cameras/v5 store from the v4 store (migrated, then
  `with_lens_coverage` of the measured circle, Circle((2131.1, 1407.9), 1680.1)) into the example's results directory, and
  `config_fx3_14mm.toml` points `lens` at it. Until then that config runs on the whole sensor.
- [ ] Commit the CHANGELOG. Do not open a PR to Ties: the owner holds it until RC works.

After Task 4: `/simplify`, then `/code-review` on `origin/main...HEAD`.
