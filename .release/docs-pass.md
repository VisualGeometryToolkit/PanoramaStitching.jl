docs-pass: 1e0160b
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
