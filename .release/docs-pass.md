docs-pass: 635440d
landing: none (the docs site is the Docs workflow's build of docs/src; no separate site repository)

# 0.10.0 docs pass (julia-docs-pass), at 635440d

On 0.9.0 (8abc4b7): the Core 0.11.0 cascade (VisualGeometryToolkitCore 0.11: an LM tolerance stop is converged).
[x] API diff since 0.9.0: no name added, removed or re-signatured (the diff touches src/rotation_bundle.jl's prose only).
[x] Docstrings: `refine_rotations` (and `refine_camera_and_rotations`, which refers to it): the termination record is Core's
    `lmsolve_chart!` verdict, returned as it is (an ftol/xtol stop is Success with no stationarity test; a NaN wall
    (Stalled, NonFinite); the cap MaxIters; max_iter = 0 (MaxIters, IterationCap)); "a refine descends towards the local
    minimum ... until a stop" (it no longer claims to end in it). `Registration`'s bundle field names Core's record: unchanged.
[x] Guide (docs/src/guide.md:106, `VgtkCore.retcode(reg.bundle)`: "how the bundle ended"), landing page, README and the
    example's README: no claim about stationarity or Stalled (grep): unchanged.
[x] CHANGELOG 0.10.0: Breaking (the bundle's tolerance stop reads Success; its migration) and why a minor; the floors.
[x] Floors: VisualGeometryToolkitCore 0.11, VisualGeometryToolkitEstimation 0.12, VisualGeometryToolkitFeatures 0.9.7
    (test and example: VisualGeometryToolkitIO 0.10.6, VisualGeometryToolkitEval 0.9.2), the registry latest.
[x] The suite's docs item (undocumented public, signature drift, detached docstrings) and the README item, and the strict
    docs build (docs/make.jl): in the release gate.

---

docs-pass: f7eb270
landing: none (the docs site is the Docs workflow's build of docs/src; no separate site repository)

# 0.9.0 docs pass (julia-docs-pass), at f7eb270

On 0.8.4 (befe6ca): the symmetric saturation mask and the pair-equation vignetting (vignetting-saturation-symmetric,
0777a1a..f2a14c3, cherry-picked), the review's fixes and the coordinator's rulings Q1–Q3.
[x] API diff since 0.8.4. Added (exported): `PairCorrespondences`, `ring_correspondences`, `ring_pair_vignetting`,
    `fit_additive_level`, `additive_level_slope`, `PairVignettingMetadata`, each documented and listed in docs/src/api.md
    ("The model-free vignetting: the pair equation") and on the landing page's table. Re-signatured: `fit_vignetting` and
    `score_seams` require `saturation_margin_px` (both signature lines name it; bb921b9, fa07d6b). `ring_correspondences`
    takes no `symmetric` keyword (0e142c1; the per-frame rule is internal). No name removed.
[x] Docstrings: `fit_vignetting` (the row rule: the symmetric, dilated masks), `score_seams` (it reads the fit's observations),
    `ring_correspondences` (one mask and one rotation per frame; the own masks by `sample_frames`' rule and the symmetric test),
    `PairCorrespondences.n_dropped`, `ring_pair_vignetting` ((InitialFailure, NoRule) when no ring pair connects a ring to the
    first). The suite's docs item (undocumented public, signature drift, detached docstrings) ran green in the full suite,
    job 20310 (731/731 at f7eb270).
[x] Guide (docs/src/guide.md): `fit_vignetting` and `score_seams` examples state `saturation_margin_px = 4.0`; the seam score's
    observations are the fit's. Landing page: the pair equation's row. README: names neither function's keywords: unchanged.
    The example's README: the fit's domain and the seam score's masks (symmetric, dilated by `[domain] saturation_margin_px`;
    the seam numbers differ from 0.8's). The configs of record: the margin's rule, √χ²₂(99 %)·σ ≈ 3 × the bundle RMS.
[x] CHANGELOG 0.9.0: Breaking (`saturation_margin_px` required by `fit_vignetting` and by `score_seams`, the seam numbers move;
    each with its migration), Fixed (the symmetric masks, one test for the fit, the seam score and the pair equation), New
    (the pair equation); the floors.
[x] Floors: VisualGeometryToolkitCore 0.10.10, VisualGeometryToolkitEstimation 0.11.7 (0.8.4's, General's latest; the new code
    reads only Core names public in 0.10.10 and nothing of Estimation).
[x] Naming: `ring_correspondences` is a noun for a computation that samples the frames (a pipeline, by the naming rule);
    check_naming reports nothing and the name stays the author's (`pair_correspondences` is Core's Pools accessor). Recorded,
    not changed.
[x] Red-first evidence: (a)–(c) red at c0ff5a2 (job 20254: 6 failures, the expected ones), green at 3d507f7 (job 20255,
    33/33). Q2: red = the missing keyword; behaviour green on fa07d6b (job 20308 at ada5bb1: one error, a MethodError,
    score_seams had no saturation_margin_px keyword; job 20309 at fa07d6b: 84/84).
[x] Docs build: docs/make.jl (strict) in the release gate.

---

docs-pass: 1d35556
landing: none (the docs site is the Docs workflow's build of docs/src; no separate site repository)

# 0.8.4 docs review (the docs skill), at 9a82644, re-recorded at 552a00e and 1d35556

On 0.8.3 (efe6ffc, released): Core 0.10.10's valid domain of a lens that is not radially symmetric is a `PolygonRegion`.
[x] API diff since 0.8.3: no exported or public name added, removed or re-signatured (src/PanoramaStitching.jl unchanged).
    Internal only: `_CameraVariable.domain` holds a `PolygonRegion` too (6373160, comment 2ffc262); `_in_frame` deleted,
    `detect_features` masks through Core's `pixel_mask(region, frame_map, width, height)` (1243042).
[x] Docstrings: `detect_features`, `sample_frames`, `frame_region`, `render_panorama`, `fit_vignetting` and the bundles
    name the imaged region and the valid domain without their types: unchanged. The suite's docs item (undocumented public,
    signature drift, detached docstrings) runs in the release gate.
[x] Guide (docs/src/guide.md:52), landing page, README and the example's README: they name the valid domain, never its
    type, and no lens they show is decentred: unchanged.
[x] CHANGELOG: 0.8.4 (Fixed: the camera variable's domain, detect_features' mask; the floors); 0.8.3's heading as released.
[x] Floors: Core 0.10.10, Estimation 0.11.7 (General's latest) in every Project.toml that names them; Features 0.9.6, IO
    0.10.3 already General's latest; the example's Eval 0.9.1 (General's latest).
[x] Docs build: docs/make.jl (strict) in the release gate.
[x] Re-recorded at 552a00e (the coordinator's rulings): an LM step whose domain Core refuses is a rejected step (f498309,
    `_NoDomain`, internal), the decentred-lens test items (01049d3), the duplication checklist's `_in_frame` note (d46eb41),
    the CHANGELOG's Fixed and Tests lines (cf660a3), the LM step's wording (552a00e: check_naming R7 reserves "trial" for a
    random hypothesis draw). No public name, signature or docstring changed; the guide, README and landing page unchanged.
[x] Re-recorded at 1d35556 (Core a7cdde09, 0.10.10: a PolygonRegion{V} with Adapt rules): the frame masks' `_Sees` adapted
    with its region (f98f057, internal; red on Core a7cdde09 without it, job 20111; green, job 20113), the device test item
    (test/decentred_lens_gpu_tests.jl), the camera variable's polygon field concrete (064ad7c), the CHANGELOG's Fixed and
    Tests lines. No public name, signature or docstring changed: the docstrings of sample_frames, render_panorama and
    fit_vignetting already state that the arrays live on the frames' backend; the guide, README and landing page unchanged.

---

docs-pass: d508b31
landing: none (the docs site is the Docs workflow's build of docs/src; no separate site repository)

# 0.8.3 docs pass (julia-docs-pass), at 07c1ae9, re-recorded at d508b31 (its only change since: the stage check's config; supersedes the 0.9.0 pass at 1b3f6ac: the version is 0.8.3)

On 0.8.2 (0d6dfcb, released): ps-sensor-model's 4f96116 (the example's frames decoded under their sensor record), the example's
port to VisualGeometryToolkitEval 0.9 (keyed products) and its required `[registration] sensor_characterisation`.
[x] API diff since 0.8.2: no file under src/ changed; no name added, removed or re-signatured (hence 0.8.3). The docstring
    scans have nothing new to cover.
[x] Guide: the verified pairs the example keeps are its `verified_pairs/<key>/verified_pairs.jld2` product (was the retired
    `verified_pairs_<key>.jld2` cache file). Landing page (docs/src/index.md) and the root README name the example only by
    its folder: unchanged.
[x] The example's README: the results layout (each step an Eval product, `<product>/<key>/` with its records), what keys each
    product (settings, code trees, the RAWs', the sensor records' and the lens's content, the upstream outputs, the drivers),
    reuse and `--force`, `stages.log`, the required `[decode]` and `[registration] sensor_characterisation` (with "tone" the
    registration's is [decode]'s), the 14 mm lens's missing coverage on the data root, `run/<key>` in the compare_makie lines;
    arms.toml's header; the Maintainers section's stage-cache check. The configs of record state both records.
[x] CHANGELOG 0.8.3 (the example's breaking changes, Known: the unresolved-frame refusal for the next patch and the 14 mm
    lens's coverage, Fixed); 0.8.2's entry dated.
[x] Docs build: docs/make.jl (strict) in the release gate.

---

# 0.8.2 docs pass (julia-docs-pass), at 3d4629d

On 0.8.1 (df36d37, released): `eliminate = true` refused without NLLSsolver's Schur system, the footprint preflight, the
derived bundle references.
[x] API diff since 0.8.1: no name added or removed. `refine_rotations` / `refine_camera_and_rotations`: `eliminate` defaults
    to NLLSsolver's Schur system (false today) and `eliminate = true` without it is refused by name; both docstrings and
    signature lines say so. `register_rotations` / `verify_pairs`: a camera view_overlap refuses is refused before any
    feature is detected, where view_overlap may be read; their docstrings' refusal lists say so.
[x] Guide, README and landing page: they name neither; unchanged.
[x] CHANGELOG 0.8.2 (Changes, Tests); 0.8.1's entry as released, dated, its vacuous-reference sentence corrected; the
    keyword's removal in .release/wave11-removals.md (the monorepo's format).
[x] Docs build: docs/make.jl (strict) in the release gate.

---

# 0.8.1 docs pass (julia-docs-pass), at 86a6296 and a7f4ed6: a test-only patch (released from df36d37)

PanoramaStitching 0.8.1 changes no source file under src/ and no public name, signature or docstring.
[x] API diff since 0.8.0: none (src/ unchanged: `git diff e041d29 -- src` is empty).
[x] Docstrings, guide, README and landing page: unchanged; no claim they make depends on the removed test literals.
[x] The example: write_results_check.jl compares the record read back with the fixture's own (no restated numbers); its
    README unchanged.
[x] CHANGELOG 0.8.1 (the test-only change and why: GitHub's runner ran another count of LM iterations than dgx01); 0.8.0
    dated 2026-11-14.
[x] Docs build: docs/make.jl (strict) in the release gate.
[x] Re-recorded at a7f4ed6: test/runtests.jl releases each finished item's memory (GitHub's 16 GB runner died on af70ea0);
    the CHANGELOG 0.8.1 entry names it; src/, docstrings, guide and README unchanged.

---

docs-pass: 5623193
landing: none (the docs site is the Docs workflow's build of docs/src; no separate site repository)

# 0.8.0 docs pass (julia-docs-pass), at aa88790, 6279499, 58d8157, be09d88, 848150c and 5623193: one overlap estimate, Core's view_overlap (owner ruling 2026-11-14)

The release tree: rel-ps080 from faabf45 (the registered 0.7.0, tree 826ecf66 = GitHub main), resolved from General alone
(Core 0.10.1, Features 0.9.1, Estimation 0.11.1, IO 0.10.0, Eval 0.8.0; the example's environment too).
[x] API diff since 0.7.0: no exported or public name added, removed or renamed (src/PanoramaStitching.jl unchanged).
    Keywords: `verify_pairs` loses `overlap_deg`; `register_rotations` loses `overlap_deg` and `near_duplicate_deg` and gains
    `near_duplicate_overlap = 0.90`. The two retired keywords are refused by name (ArgumentError naming the migration), as
    0.7 refuses its other retired keywords. `Registration.near_duplicates` keeps its name and type; its meaning is the
    view_overlap rule.
[x] Docstrings: `Registration` (near_duplicates), `verify_pairs` (signature and the retired keyword), `register_rotations`
    (both signature lines gain near_duplicate_overlap and check_tracks; the pairs and near-duplicates items; the
    refusals). The suite's docs item (undocumented public, signature drift, detached docstrings) runs in the release gate.
[x] Guide: registration steps 2 (`:lag_overlap`'s pairs past the lag: any shared view, min_inliers deciding) and 5 (the
    near-duplicate rule, 0.90 by default, 0 off).
[x] README and landing page: their example uses `pairs = :all` and neither retired keyword; unchanged (the readme item runs
    them verbatim, near-duplicates now on by default, in the release gate).
[x] The example: README step 2 names the near-duplicate rule and the refused keys; both configs of record state
    `near_duplicate_overlap = 0.90` with the owner's ruling in the comment (were `near_duplicate_deg = 4.0` / `9.0`); no
    `overlap_deg` or `near_duplicate_deg` in src/, docs/src, the README or the example but the refusals (grep; the tests
    and the CHANGELOG name them to refuse them).
[x] CHANGELOG 0.8.0 with "## Breaking changes": the retired keywords and how to migrate, the new criterion, what
    registrations change.
[x] Docs build: docs/make.jl (strict) in the release gate.
[x] Re-recorded at 6279499: the CHANGELOG's one banned word (check_naming R7, the gate at c816935) reworded; no other change.
[x] Re-recorded at 58d8157: two test items (bundle width, view-graph filters) state near_duplicate_overlap = 0 (the gate at
    c816935: view_overlap refuses their bare sensor past the KB lens's fold); no docs change.
[x] Re-recorded at be09d88 (the desktop's ruling on the owner's BoundedCamera rule): 58d8157 reverted (8083f3b), the items'
    camera given its valid region instead (SweepScene's BCAM, an image circle of 500 px; the near-duplicate check runs and
    finds none, asserted); the CHANGELOG names the refusal of a camera footprint the lens cannot back-project, and what to
    set (the lens coverage). The 8 mm config of record keeps near_duplicate_overlap = 0.90 (the owner's value) pending the
    owner's answer on its 24 of 45 near-duplicate pairs.
[x] Re-recorded at 848150c (the desktop: one list of the retired registration settings): the public name
    `PanoramaStitching.refuse_retired_registration(table, where)` added (public, not exported, as detect_features and
    pair_bearings), its docstring with a jldoctest, in docs/src/api.md's Registration section (checkdocs = :public); the
    example's RETIRED_REGISTRATION_KEYS and refuse_retired_registration deleted (load_config calls the package's); the
    CHANGELOG's Added line and the retired message's new form. The API diff since 0.7.0 is now this one added public name
    and the keywords above.
[x] Re-recorded at 5623193: the new test item's one banned word (check_naming R7, the gate at eb09883) reworded; no docs
    change.

---

docs-pass: 51933f0
landing: none (the docs site is the Docs workflow's build of docs/src; no separate site repository)

# 0.7.0 docs pass (julia-docs-pass), at the release tree resolved from General alone

The release tree: w10-ps 6ab1330 (the wave work and the track check; gated green as w10-ps-tmp ad2f3059 on up-v13) with
GitHub main (0.6.5) merged (91c4a7d) and no temporary [sources] (only the members' own path to the repository). Resolved from
General (refreshed from GitHub) alone: Core 0.10.0, IO 0.10.0, Features 0.9.0, Estimation 0.11.0 (job 15032); floor_compat
--raise: 0 entries raised (the stack entries "0.10", "0.9", "0.11" are the registered wave versions' ranges); --latest: 0.
[x] API diff since 0.6.5: added sweep_up, up_frame(up), coverage_extent, vignetting_summary, PositionNoise, OBSERVATION_NOISES,
    the sensor-model keywords, register_rotations' near_duplicate_deg / check_tracks, Registration.near_duplicates /
    track_check, RotationBundleMetadata.inlier_rms; removed IntensityNoise, IntensityAndPositionNoise, MEASUREMENT_NOISES,
    measurement_noise, detection_noise (refused by name). Every added name is in docs/src/api.md (checkdocs = :public);
    no removed name appears in README.md, docs/src or the example's README and configs (grep).
[x] Guide: the registration steps now name the near-duplicate pairs (step 5) and the track check (step 7), on by default.
    The vignetting section already states observation_noise = nothing | PositionNoise and the frames' sensor model.
[x] Docstrings: register_rotations' Registration field list lost a doubled "its". The suite's docs item (undocumented
    public, signature drift, detached docstrings) runs in the release gate.
[x] CHANGELOG 0.7.0: the track check / near-duplicate item (behaviour change: the rotations differ from 0.6's with
    check_tracks on) added; 0.6.5 kept below 0.7.0; dated.
[x] Install lines: README's pkg> add names only General packages and no other registry; the readme item resolves it from
    the registry in the release gate (the known pre-registration failure on every -tmp gate).
[x] The 0.6.5 fix is present: examples/panorama_vignetting/run.jl:538 builds each pose with a static zero translation
    (EuclideanMap(R, zero(SVector{3, Float64}) .* m)), the guide's section 5 the same (docs/src/guide.md:262), and
    write_results_check.jl runs from run_checks.jl:10.
[x] Docs build: docs/make.jl (strict) in the release gate.
[x] Re-recorded at 51933f0: w10-ps bc81352 merged (test/track_check_tests.jl only: the track check's test fixes the up-v13
    gate ad2f3059 carried; the release gate c301a9e failed that item without them); no docs change.

---

docs-pass: 90db87d

# 0.6.5 docs pass (julia-docs-pass) — the example writes its results of record again

PanoramaStitching 0.6.5 is a patch: no source file under src/ changed and no public name was added, removed or re-signatured.
- API diff since 0.6.4: none.
- The bug: examples/panorama_vignetting/run.jl `write_results` built each pose as `EuclideanMap(R, zeros(3) .* m)` with `R` an
  `SMatrix{3, 3}`, a call no Core 0.9 method takes (MethodError after the vignetting fit, every run). Now
  `zero(SVector{3, Float64}) .* m`, through Core's existing boundary constructor `EuclideanMap(::SMatrix{3, 3}, ::StaticVector{3})`.
- Guide (docs/src/guide.md), section 5 "Saving the results": its illustrative block made the same call on the registration's
  SMatrix rotations; corrected the same way (and `using StaticArrays: SVector`). The block stays illustrative (IO is one more
  package), so readme_tests.jl's count of illustrative blocks is unchanged.
- The example's README: unchanged (its outputs table already names panorama.json and panorama_map.png; the "Maintainers"
  paragraph names run_checks.jl, which now also runs write_results_check.jl).
- README and landing page: unchanged.
- CHANGELOG: 0.6.5.
- Docs build: strict, green in the fresh-clone gate on Julia 1.13.3 at 90db87d (job 14670), against the registered Core 0.9.6,
  Estimation 0.10.2, Features 0.8.4, IO 0.9.5 and (the example) Eval 0.7.3; the example's checks green (job 14669: write_results_check 7/7).

---

docs-pass: 4728e7d

# 0.6.4 docs pass (julia-docs-pass) — the results through VisualGeometryToolkitIO's vgio/panorama/v1

PanoramaStitching 0.6.4 is a patch: no source file under src/ changed and no public name was added, removed or re-signatured.
The example saves its results through `VgtkIO.save_panorama`, and its hand-made V record is deleted.
- API diff since 0.6.3: none.
- Guide (docs/src/guide.md): a new section, "5. Saving the results", an `# Illustrative` block (VisualGeometryToolkitIO is
  one more package, so the block is counted and not run; readme_tests.jl pins three illustrative blocks). The intro links it.
  The block's calls are the ones results_io_tests.jl runs: the Trajectory camera → world with zero translations,
  `save_panorama(path, camera, rotations, V, g)`, `load_panorama`, and the loaded QuatRotations passed to render_panorama.
- The example's README: the outputs table names panorama.json, panorama_map.png and the compare files, which replace
  vignetting.json and the two CSVs. The stale "needs the 0.5 stack" paragraph is replaced by the IO 0.9.4 requirement.
- README and landing page: unchanged. The install line and the example do not load IO, and IO is not a package dependency
  (the owner's ruling, option A).
- CHANGELOG: 0.6.4.
- Docs build: strict, green in the fresh-clone gate on Julia 1.13.3 at 4728e7d (job 12563).

---

docs-pass: 346e52e

# 0.6.3 docs pass (julia-docs-pass) — Julia 1.13

PanoramaStitching 0.6.3 is a patch whose only change is the Julia requirement: `julia = "1.13"` in the `[compat]` of every
Project.toml in the repository, the CI workflows on Julia 1.13 only, and the CHANGELOG section.
- API diff since the last release: none (no source file under src/ or ext/ changed).
- Docstrings: unchanged; the suite's docs item runs in the fresh-clone gate.
- README / landing / guides: every statement of the required Julia version says 1.13 now: README.md and docs/src/index.md.
  Mentions of Julia 1.12 left in test comments record why a bound or a workaround was set (LLVM 18 vs 20 rounding,
  Julia 1.12's docstring rules, which 1.13 keeps); they are history, not instructions.
- Docs build: run by the fresh-clone gate on Julia 1.13.3 against the registered stack (General from GitHub).

---

docs-pass: 653ae31

# 0.6.2 docs pass (julia-docs-pass) — on VisualGeometryToolkitEstimation 0.10

PanoramaStitching 0.6.2 is a patch: `[compat]` VisualGeometryToolkitEstimation "0.10" in every member that declares it, and the
CHANGELOG section. No public name added, removed or re-signatured.
- API diff since the last release: none.
- docstrings: `refine_rotations`' docstring and the forward map's comment in sampling.jl name VisualGeometryToolkitEstimation's `fit_forward_spline` for a held SplineLens camera's forward (89dd421); the autocalibration's SplineLens refusal comment says the stack refines such a camera only by back-projection, or holds it (b0815f9); the autocalibration reads its camera variable through `_camera_of_chart` (a JET fix on Estimation 0.10, no docs change).
- stale mentions: nothing here names the SplineLens image-space refine Estimation 0.10.0 retired (its free route,
  `penalty`, the GCV and window metadata): checked by grep over src, docs, README and the paper projects.
- README / install lines: unchanged (no version is named); the README rows resolve the registry and pass once
  Estimation 0.10.0 is registered.

---

PanoramaStitching 0.6.1 (2026-11-12), on d4376f7 (0.6.0, the registered tree = GitHub main). PanoramaStitching has no landing site outside the repository.

- The sensor image's names are VisualGeometryToolkitCore's since Features 0.8: every docstring, guide line, README comment
  and example comment that named Features as the owner of `raw_frame_map`, `bayer_plane_map` or `LinearRaw` names Core
  (src/features.jl, registration.jl, panorama.jl, sampling.jl, vignetting.jl, docs/src/guide.md, the example's README and
  run.jl). The example's version-history comments (v5) and the configs of record are left as they were (the configs'
  bytes key the comparison runs).
- README and docs/src/index.md (one example, shown twice): the frames' map is Core 0.9.4's mode map
  `raw_frame_map(LinearRaw{:bin2})` and places the frame pixels; the install line drops CoordinateTransformations (the
  README item checks the install line against the examples' imports); "registered in CauRegistry" corrected to General.
- CHANGELOG 0.6.1 written. No public name added or removed.
- Gates (test branch ps-0.6.1-tmp094 = 15b8e50 + temporary [sources] to vgtk 9762b9b4, Slurm 10548, Julia 1.13.3, one
  B200): the docs build (checkdocs = :public, doctest, linkcheck, no warnonly) exit 0; Pkg.test 476/476 including the
  README item (fresh environment, the sourced stack developed), the docs, naming, quality and duplication items. The
  release tree's README item runs from General in the fresh-clone gate once Core 0.9.4 / Features 0.8.1 are registered.
- Re-checked at ab6d9f8c7a158b4b48539ab14a9e35076bc56ec1: after the bump only tests, the example's stage-cache check and one CHANGELOG line changed
  (Julia 1.12: the bundle pins per LLVM, Test.anynonpass); no docstring, guide or README text. The README and guide examples
  ran from General on Julia 1.13.3 and 1.12.7 in the fresh-clone gate (Slurm 10660), on the registered Core 0.9.4.
- Re-checked at 7382002d6a13378e0d1a99831440ffa98e8750c4: tests, the example's stage-cache check and the CHANGELOG line only; no docs change.
