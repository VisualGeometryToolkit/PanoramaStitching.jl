docs-pass: 4decd45

# 0.6.2 docs pass (julia-docs-pass) — on VisualGeometryToolkitEstimation 0.10

PanoramaStitching 0.6.2 is a patch: `[compat]` VisualGeometryToolkitEstimation "0.10" in every member that declares it, and the
CHANGELOG section. No public name added, removed or re-signatured.
- API diff since the last release: none.
- docstrings: `refine_rotations`' docstring and the forward map's comment in sampling.jl name VisualGeometryToolkitEstimation's `fit_forward_spline` for a held SplineLens camera's forward (89dd421); the autocalibration's SplineLens refusal comment says the stack refines such a camera only by back-projection, or holds it (b0815f9); the autocalibration reads its camera variable through `_camera_model` (a JET fix on Estimation 0.10, no docs change).
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
