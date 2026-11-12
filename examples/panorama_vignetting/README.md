# Panorama vignetting: a camera's vignetting from its own panorama frames

This example takes the RAW frames of a camera that only rotated (a sweep of a distant scene), registers them, fits the camera's
vignetting V from the frames alone — no flat field, no interpolation of the frames — and renders the panorama with and
without the correction. Two configs of record come with it:

| config | camera and lens | frames | V model |
|---|---|---|---|
| `config_fx3_14mm.toml` | Sony FX3, Sigma 14 mm (a Kannala–Brandt fit) | 11 indoor frames, two rows (pitch ≈ 0° and ≈ +14°), yaw over 360° | radial (see `summary.md` for the tensor comparison) |
| `config_fx3_8mm_rooftop.toml` | Sony FX3, 8 mm fisheye (a generic grid lens) | 10 rooftop frames, mostly yaw (−54° … +47°), pitch −5° … +27° | radial |

## Running it

```bash
export VGTK_DATA_ROOT=~/data/rc-cvpr2027            # the data root that holds the inputs (below)
julia --project=<env> examples/panorama_vignetting/run.jl examples/panorama_vignetting/config_fx3_14mm.toml
```

`<env>` is a Julia environment with PanoramaStitching, VisualGeometryToolkitCore, VisualGeometryToolkitEstimation,
VisualGeometryToolkitFeatures, VisualGeometryToolkitIO, CUDA, CUDA_Runtime_jll, CUDNN_jll, LibRaw_jll, PNGFiles, JSON, SHA
and TOML; `Project.toml` in this folder is that environment (`julia --project=examples/panorama_vignetting -e 'using Pkg; Pkg.instantiate()'`).
**It needs the VisualGeometryToolkit 0.5 stack** (VisualGeometryToolkitCore 0.5, VisualGeometryToolkitEstimation 0.6, and
the Features release that ships `load_raw`). Until those are registered, use an environment whose `[sources]` point at a
checkout of the monorepo (the author's scratch environment does exactly that).

Every setting is in the config — nothing in `run.jl` has a default — and every input path is relative to `$VGTK_DATA_ROOT`.
Everything the run writes goes under VisualGeometryToolkitEval's results root (`$VGTK_RESULTS_ROOT`):
`<results>/panorama_vignetting/<name>/` holds the stages' keyed records and the decode cache, and the run's outputs go to
`run-<run key>/` in it, keyed by the whole config (Eval's `run_key`), with its run records. From a REPL,
`include("run.jl"); run_example("config_fx3_14mm.toml")` runs everything; each stage is a function and can be rerun alone.

The compute backend is the config's `[compute] device` (`"cuda"`). On a shared GPU without free memory, run once with
`PANORAMA_VIGNETTING_DEVICE=cpu`: the same kernels run on the host and give the same numbers to the solve's precision (the
atomic sums run in another order; the package's tests check log V device against host to 1e-6, the CG iteration count varies
by a few percent from run to run for the same reason); `run.log` and `summary.md` record which backend ran. A grid lens (the 8 mm rooftop) renders
on the device too (Core's Adapt rules carry its `SplineLens` into `image_warp`).

LoMa (the registration's descriptor and matcher) runs on ONNX Runtime's CUDA provider: the configs of record state
`[registration] execution_provider = "cuda"` (an error naming why where it cannot run; never the CPU). The CUDA 12 runtime
and cuDNN come from this environment's `CUDA_Runtime_jll` and `CUDNN_jll`, downloaded on first use, whatever runtime CUDA.jl
uses; `"auto"` runs there where a device is visible and on the CPU otherwise, with a warning naming why. ONNX Runtime uses
Julia's threads (`--threads`; in a Slurm job `--threads=$SLURM_CPUS_PER_TASK`). The provider is part of the cache keys of
the verified pairs and of the registration, both records hold LoMa's provenance, and `summary.md` states it.

## Inputs

Under `$VGTK_DATA_ROOT`:

- `panorama_good/14mm/*.ARW` and `vignetting/data/sony_fisheye/*.ARW` — the RAW frames, in capture order by name.
- `vignetting/data/ref_grid.json` — the 8 mm fisheye as a generic grid lens (`vgio/generic_camera/v2`).
- `panorama_good/derived/fx3-14mm-kb_fit.cameras-v4.json` — the 14 mm lens, a `vgio/cameras/v4` store. It is DERIVED: the
  original, `geometric_accuracy/results/logs/night_stars/fx3-14mm/kb_fit/lens.json`, is a `vgio/cameras/v3` file and is
  never modified. Produce the v4 copy once with VisualGeometryToolkitIO's `vgio_migrate`, which writes a new file, reads it
  back and compares it with the original number for number before it renames it into place:

  ```julia
  import VisualGeometryToolkitIO as VgtkIO
  root = ENV["VGTK_DATA_ROOT"]; mkpath(joinpath(root, "panorama_good/derived"))
  VgtkIO.vgio_migrate(joinpath(root, "geometric_accuracy/results/logs/night_stars/fx3-14mm/kb_fit/lens.json");
                      out = joinpath(root, "panorama_good/derived/fx3-14mm-kb_fit.cameras-v4.json"))
  ```

  (or the command line: `julia --project=<VisualGeometryToolkitIO> bin/vgio_migrate.jl file --out <new v4 path> --apply
  <original>`; without `--apply` it is a dry run).

## What each step does

1. **Decode** (`decoded_frames`). Each RAW file is read by VisualGeometryToolkitFeatures' `load_raw` (LibRaw through
   LibRaw_jll; the two steps `load_raw_mosaic` then `LinearRaw(mosaic; mode = :bin2, gain)`): 2 × 2 Bayer quads become one
   pixel of linear R, mean G and B (`1.0` = the white level), at half the sensor's resolution (the frames' map onto the
   camera's pixels is VisualGeometryToolkitCore's `raw_frame_map`, x ↦ 2x), each plane with its own map (`bayer_plane_map`: G at the quad
   centre, R and B ¼ px off) and a per-quad saturation mask (any of the four photosites at the white level). Decoding is done once and cached (below).
2. **Register** (`verified_pairs`, `registration`). `verify_pairs`: ASIFT keypoints (none within three scales of a saturated
   pixel: Features' detection mask) described by DeDoDe and matched by LoMa,
   the rotation of every frame pair by RANSAC on bearings (`find_rotation`, the
   Student-t + uniform score at the outlier half-width `outlier_halfwidth_px`, Jeffreys and flat priors), cached in
   `verified_pairs_<key>.jld2` before anything else runs on them, keyed by the matching and verification settings (every
   `[registration]` key but the averaging's and the bundle's), the inputs and the RAW files' hashes: changing only the
   averaging or the bundle reuses the pairs, and a failure past them loses none of the hour they take. Then
   `register_rotations` on the pairs: robust rotation averaging over VisualGeometryToolkitEstimation's view graph, tracks,
   and the bundle of the rotations and track directions. A grid lens is read
   as a `SplineLens` camera and projects through its forward spline, fitted over its imaged region. The rotations are
   cached in the registration record (`registration_<key>.jld2`), keyed by the registration settings, the inputs and the RAW files' hashes, with the bundle's
   outcome (`converged`, `not_run` or another retcode) and its count of observations dropped at its start (a start prediction
   off the camera model's valid domain: such an observation is left out, the rest solved); `summary.md` states it.
3. **Fit the vignetting** (`fit_vignetting`, PanoramaStitching). Every pixel of every frame inside the domain (the camera's
   imaged region — the sensor ∩ its lens coverage ∩ its model's valid domain — less a rim margin at the coverage's edge; unsaturated; positive) is
   one observation of
   `I_k(p) = L(R_k·f(p)) · V(u(p))`: the scene's radiance L in the pixel's direction times the vignetting V at its sensor
   point. The photo is read at its own pixel centres — no interpolation anywhere: the pixel's ray `f(p)` is the lens's
   back-projection, and the scene is a B-spline on the sphere (an azimuthal-equidistant chart, 1° controls, a Δ³ smoothness
   penalty) fitted together with V. log V is a cubic B-spline, RADIAL in the distance from a centre or a 2-D TENSOR spline
   over the sensor (VisualGeometryToolkitCore's `LogVignetting`). The model is linear in log L, log V and the log gains: one
   linear least squares, each row the constraint of its measurement — sensor point and log I — weighted by its whitening by
   the measurement noise `vignetting.measurement_noise` chooses: `"intensity"` (noise in the value only: weight I²) or
   `"intensity_and_position"` (the value and the sensor position, the sensor point's σ the registration bundle's core scale,
   loss width over its efficiency constant, recorded in the registration record (`registration_<key>.jld2`); `log_intensity_scale` the σ of log I;
   `measurement_noise_iterations` its reweight-and-solve cycles after the I² solve). Its normal equations are accumulated on the GPU, and it is
   solved by conjugate gradients there. V's scale is not observable (L can take any factor), so V is reported with mean
   log V = 0 over the domain — no special point is pinned.
   *The frames' exposure gains* (`vignetting.gain_correction`, required: `"none"`, `"intensity"` or `"channel"`): with
   `"intensity"` each frame has its own gain g_k, shared by every colour plane (`I_k = g_k·L·V`, g₁ = 1), with `"channel"` one per frame and colour plane
   (`g_kc`, g_1c = 1: a white balance that drifts too; V is one for all planes), divided out by the seam score and the
   corrected renders as well (per plane). Both configs fit them: the
   rooftop's daylight changed during the sweep (gains up to +18 %; without them V came out 7 % too dark at the domain's edge),
   and the 14 mm frames drift by up to 6.7 % (with the gains the seams' 1.4826·MAD fell from 0.107 to 0.102).
   *Why radial for the rooftop:* a sweep that turns mostly about the vertical cannot see a 2-D V's vertical tilt (a field
   that varies only with the ray's height is unchanged by a pure yaw, so L absorbs it). On the rooftop, a free 2-D V moved its
   maximum ~460 px off-centre. The tensor model needs frames with pitch and roll diversity; the 14 mm set has two rows, so
   its run also fits the tensor model — and it is not observed there either: its maximum lands 522 px off the principal
   point, it is 32 % asymmetric, and it leaves the seams worse than the radial V (config and `summary.md`).
   *The 14 mm image circle:* the sensor's corners lie outside this lens's image circle, so the frames go dark past
   ρ ≈ 1700 px. The circle was measured from the frames (where log G falls through the midpoint between its plateau and the
   corner floor) and is the camera's lens coverage, stored with the camera (a config's `[inputs] circle` is refused); a
   whole-sensor fit put V at 0.07–0.2 in the corners and amplified their noise. The 14 mm config's v4 store holds no coverage
   yet, so that config runs on the whole sensor until a derived v5 store with the measured circle replaces it.
4. **Score the seams** (`score_seams`, PanoramaStitching). At every pixel of an equirectangular raster seen by two or more
   frames: the median over the other frames of `log(I_k/V_k) − log(I_last/V_last)`, the frame shown last being the one the
   unblended render keeps. With the right V only noise, misregistration and parallax remain. Its rms and robust spread
   (1.4826·MAD) are reported for no correction and for the fitted V. **Saturated pixels are excluded** (a clipped value is
   no measurement). `seams_*.png` map it: red where the other frames are brighter than the shown one, blue darker (±0.3 in
   log), grey where fewer than two frames measure the pixel.
5. **Render** (`render_panorama`). Cylindrical, cylindrical equal-area and a rectilinear virtual camera, each corrected
   (divided by V) and uncorrected, unblended (the last frame that sees a pixel wins, so a wrong V shows as a step at every
   frame boundary), at ONE fixed exposure (1 / the 99.5 % green of the corrected cylindrical), LibRaw's daylight white
   balance (for viewing only) and sRGB-encoded (Features' `srgb_image`). The up frame is `level`: the vertical orthogonal to every frame's image x
   axis (a roll-free sweep), which holds whatever the frames' pitch.
   The renders show the REAL images: saturated pixels keep their clipped values (bright windows stay bright), the masks are
   only used by the fit and the seam score.

## The comparison of the fit's settings

`arms.toml` is the comparison of record of `fit_vignetting`'s settings: twelve arms, V `"free"` (self-calibration) or
`"fixed"` (held at the V of record, the `fixed_from` arm's fit, `selfcal_shared_gain_value`: one gain per frame and I²
weights) × the `gain_correction` (`"none"`, `"intensity"`, `"channel"`) × the `measurement_noise` (`"intensity"`,
`"intensity_and_position"`), each on both sets (`config_fx3_14mm.toml`, `config_fx3_8mm_rooftop.toml`: their decode,
registration, domain, radial model and solver). An arm's name is its own label (`selfcal` or `given_v`; `no_gain`,
`shared_gain` or `plane_gain`; `value` or `value_point`), not its tags.

    julia --project=<env> examples/panorama_vignetting/compare_arms.jl examples/panorama_vignetting/arms.toml

Per set, a run keyed by `arms.toml`, the set's config and its registration record writes
`<results>/panorama_vignetting/<name>/arms/<key>/results.json` and `results.md`: per arm the seams' 1.4826·MAD per colour plane and pooled over the
planes, their rms, the gains (`K × 3`), V's range, asymmetry and ring profile (100 px rings about the circle's centre), and
the warm fit time. A key that exists is reused. The configs of record keep their settings until the table decides.

## Outputs (`<results>/panorama_vignetting/<name>/`)

| file | content |
|---|---|
| `decoded/linear_raws.jld2` | the decoded frames: VisualGeometryToolkitFeatures' decode cache (`load_raw(path; cache)`, a `FileCache`), one `LinearRaw` per RAW content and decode settings |
| `verified_pairs_<key>.jld2` | the verified pairs (the frames' keypoints and covariances; per pair its rotation, counts and inlier matches): Eval's keyed result cache (`cached_result`), keyed by Eval's `run_key` of the matching and verification settings, LoMa's provider, the inputs, the decode settings and the RAW files' hashes |
| `registration_<key>.jld2` | the registration (`d = R_k·f_k`, the first frame's camera frame), keyed the same way by every `[registration]` setting |
| `run-<run key>/` | the run's outputs below, keyed by the whole config, with Eval's run records (`write_run_records`: the config of record, the key, the driver, the inputs read) |
| `vignetting.json`, `vignetting_profile.csv`, `vignetting_map.csv`, `vignetting_map.png` | the fitted V of record: its spline space and log-V coefficients; V(ρ) every 4 px (radial); V on the sensor every 32 px; V as an image (normalized to its maximum) |
| `vignetting_compare_<model>.*` | the comparison fits (`compare_models`), the same record |
| `seams_uncorrected.png`, `seams_corrected.png` | the seam-ratio maps |
| `cylindrical_*.png`, `equal_area_*.png`, `rectilinear_*.png` | the renders, `_corrected` and `_uncorrected` at the same exposure |
| `summary.md`, `run.log` | the numbers: registration, the fits (retcode, residual spread, V's range, maximum and asymmetry), the seam spreads, the wall times |

**Viewing a run.** `compare_makie.jl` shows a run's uncorrected and corrected outputs in GLMakie (Features'
`compare_images`), one above the other with linked axes (pan and zoom move both), and a menu to switch between the projections and the seam maps:

```bash
julia --project=examples/panorama_vignetting examples/panorama_vignetting/compare_makie.jl $VGTK_RESULTS_ROOT/panorama_vignetting/fx3_14mm_indoor/run-<key>
```

From a REPL: `include("compare_makie.jl"); compare_makie("<results>/panorama_vignetting/fx3_14mm_indoor/run-<key>")`. `save_comparison(path, dir; view)` writes
the figure as a PNG.

**The decode cache.** The linear image is stored as the photosite code sums rather than 16-bit rounded linear values, so it
is lossless: the linear value is `(code/2 − black_p)/(white − black_p)·gain`, and reading the cache back gives load_raw's
values exactly (in the Float32 the pipeline uses). The mask is a separate file next to it; the render steps read only the
images, the fit and the seam score read image and mask. A rerun reuses the cache when a frame's RAW hash and the decode
settings (mode, gain, cache format) match `decode.json`; anything else decodes the frame again.

## Maintainers

The example's checks run in this environment, not in PanoramaStitching's test suite (they need VisualGeometryToolkitEval,
which the stack's allow-list keeps out of the package's tests): `run_checks.jl` is their entry point,

```bash
julia --project=examples/panorama_vignetting examples/panorama_vignetting/run_checks.jl
```

and `reproduce.toml`'s `checks` stage runs it. It runs `stage_cache_check.jl` (the verified pairs reused when only the
averaging or the bundle changes, recomputed when a pair setting, LoMa's provider or a RAW file changes) and
`autocalibration_check.jl` (the 14 mm bundle-equivalence check of `refine_camera_and_rotations` against PanoramaStitching
7bbbea03's bundle on the same tracks; not run where its declared inputs are absent from the data root). Each exits 1 on a
failure. PanoramaStitching's own suite checks only that `run.jl` and the checks parse, and the example's retired keys.
