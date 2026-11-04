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
VisualGeometryToolkitFeatures, VisualGeometryToolkitIO, CUDA, LibRaw_jll, PNGFiles, JSON, SHA and TOML; `Project.toml` in
this folder is that environment (`julia --project=examples/panorama_vignetting -e 'using Pkg; Pkg.instantiate()'`).
**It needs the VisualGeometryToolkit 0.5 stack** (VisualGeometryToolkitCore 0.5, VisualGeometryToolkitEstimation 0.6, and
the Features release that ships `load_raw`). Until those are registered, use an environment whose `[sources]` point at a
checkout of the monorepo (the author's scratch environment does exactly that).

Every setting is in the config — nothing in `run.jl` has a default — and every input path is relative to `$VGTK_DATA_ROOT`.
The outputs go to the config's `output` folder (relative to the config file: `out/<name>/`, gitignored). From a REPL,
`include("run.jl"); run_example("config_fx3_14mm.toml")` runs everything; each stage is a function and can be rerun alone.

The compute backend is the config's `[compute] device` (`"cuda"`). On a shared GPU without free memory, run once with
`PANORAMA_VIGNETTING_DEVICE=cpu`: the same kernels run on the host and give the same numbers to the solve's precision (the
atomic sums run in another order; the package's tests check log V device against host to 1e-6, the CG iteration count varies
by a few percent from run to run for the same reason); `run.log` and `summary.md` record which backend ran. A grid lens (the 8 mm rooftop) renders
on the device too (since PanoramaStitching 0.2.1 with Core 0.5.1, whose Adapt rules carry its `SplineProjection` into
`image_warp`).

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
   pixel of linear R, mean G and B (`1.0` = the white level), at half the sensor's resolution (`raw_scale = 2`), with the
   planes' true sample positions (G at the quad centre, R and B ¼ px off) and a per-quad saturation mask (any of the four
   photosites at the white level). Decoding is done once and cached (below).
2. **Register** (`registration`). `register_rotations`: ASIFT keypoints described by DeDoDe and matched by LoMa, the rotation
   of every frame pair by RANSAC on bearings (`find_rotation`, MSAC scoring with a 5 px half-width), robust rotation
   averaging over the view graph, tracks, and the bundle of the rotations and track directions. A grid lens has no
   closed-form projection, so its forward map is a `SplineProjection` fitted to it over the image circle. The rotations are
   cached in `rotations.json`, keyed by the registration settings, the inputs and the RAW files' hashes, with the bundle's
   outcome (`converged`, `nonfinite_start` — its start not finite, a start prediction outside the lens's domain —, `not_run`
   or another retcode) and its count of observations not finite at the start; `summary.md` states it.
3. **Fit the vignetting** (`fit_vignetting`, PanoramaStitching). Every pixel of every frame inside the domain (the image
   circle less a rim margin, or the whole sensor; unsaturated; positive) is one observation of
   `I_k(p) = L(R_k·f(p)) · V(u(p))`: the scene's radiance L in the pixel's direction times the vignetting V at its sensor
   point. The photo is read at its own pixel centres — no interpolation anywhere: the pixel's ray `f(p)` is the lens's
   back-projection, and the scene is a B-spline on the sphere (an azimuthal-equidistant chart, 1° controls, a Δ³ smoothness
   penalty) fitted together with V. log V is a cubic B-spline, RADIAL in the distance from a centre or a 2-D TENSOR spline
   over the sensor. The least squares is weighted by I² (the log of a photon-limited value), its normal equations are
   accumulated on the GPU, and it is solved by conjugate gradients there. V's scale is not observable (L can take any
   factor), so V is reported with mean log V = 0 over the domain — no special point is pinned.
   *The frames' exposure gains* (`vignetting.frame_gains`, required): with `true` each frame has its own gain g_k
   (`I_k = g_k·L·V`, g₁ = 1), divided out by the seam score and the corrected renders as well. Both configs fit them: the
   rooftop's daylight changed during the sweep (gains up to +18 %; without them V came out 7 % too dark at the domain's edge),
   and the 14 mm frames drift by up to 6.7 % (with the gains the seams' 1.4826·MAD fell from 0.107 to 0.102).
   *Why radial for the rooftop:* a sweep that turns mostly about the vertical cannot see a 2-D V's vertical tilt (a field
   that varies only with the ray's height is unchanged by a pure yaw, so L absorbs it). On the rooftop, a free 2-D V moved its
   maximum ~460 px off-centre. The tensor model needs frames with pitch and roll diversity; the 14 mm set has two rows, so
   its run also fits the tensor model — and it is not observed there either: its maximum lands 522 px off the principal
   point, it is 32 % asymmetric, and it leaves the seams worse than the radial V (config and `summary.md`).
   *The 14 mm image circle:* the sensor's corners lie outside this lens's image circle, so the frames go dark past
   ρ ≈ 1700 px. The config's circle was measured from the frames (where log G falls through the midpoint between its plateau
   and the corner floor); a whole-sensor fit put V at 0.07–0.2 in the corners and amplified their noise.
4. **Score the seams** (`score_seams`, PanoramaStitching). At every pixel of an equirectangular raster seen by two or more
   frames: the median over the other frames of `log(I_k/V_k) − log(I_last/V_last)`, the frame shown last being the one the
   unblended render keeps. With the right V only noise, misregistration and parallax remain. Its rms and robust spread
   (1.4826·MAD) are reported for no correction and for the fitted V. **Saturated pixels are excluded** (a clipped value is
   no measurement). `seams_*.png` map it: red where the other frames are brighter than the shown one, blue darker (±0.3 in
   log), grey where fewer than two frames measure the pixel.
5. **Render** (`render_panorama`). Cylindrical, cylindrical equal-area and a rectilinear virtual camera, each corrected
   (divided by V) and uncorrected, unblended (the last frame that sees a pixel wins, so a wrong V shows as a step at every
   frame boundary), at ONE fixed exposure (1 / the 99.5 % green of the corrected cylindrical), LibRaw's daylight white
   balance (for viewing only) and sRGB-encoded. The up frame is `level`: the vertical orthogonal to every frame's image x
   axis (a roll-free sweep), which holds whatever the frames' pitch.
   The renders show the REAL images: saturated pixels keep their clipped values (bright windows stay bright), the masks are
   only used by the fit and the seam score.

## Outputs (`out/<name>/`)

| file | content |
|---|---|
| `decoded/<stem>.png` | the cached frame: 16-bit RGB of per-plane photosite code sums (R and B `2·raw`, G `raw₁ + raw₂`) |
| `decoded/<stem>_saturated.png` | its saturation mask, 8-bit 0/1, the same size |
| `decoded/decode.json` | once per set: the decode settings (mode, gain), black and white levels, the planes' sample offsets, and per frame the RAW's SHA-256 |
| `rotations.json` | the registration (`d = R_k·f_k`, the first frame's camera frame), with its cache key |
| `vignetting.json`, `vignetting_profile.csv`, `vignetting_map.csv`, `vignetting_map.png` | the fitted V of record: its spline space and log-V coefficients; V(ρ) every 4 px (radial); V on the sensor every 32 px; V as an image (normalized to its maximum) |
| `vignetting_compare_<model>.*` | the comparison fits (`compare_models`), the same record |
| `seams_uncorrected.png`, `seams_corrected.png` | the seam-ratio maps |
| `cylindrical_*.png`, `equal_area_*.png`, `rectilinear_*.png` | the renders, `_corrected` and `_uncorrected` at the same exposure |
| `summary.md`, `run.log` | the numbers: registration, the fits (retcode, residual spread, V's range, maximum and asymmetry), the seam spreads, the wall times |

**Viewing a run.** `compare_makie.jl` shows a run's uncorrected and corrected outputs in GLMakie, one above the other with
linked axes (pan and zoom move both), and a menu to switch between the projections and the seam maps:

```bash
julia --project=examples/panorama_vignetting examples/panorama_vignetting/compare_makie.jl examples/panorama_vignetting/out/fx3_14mm_indoor
```

From a REPL: `include("compare_makie.jl"); compare_makie("out/fx3_14mm_indoor")`. `save_comparison(path, dir; view)` writes
the figure as a PNG.

**The decode cache.** The linear image is stored as the photosite code sums rather than 16-bit rounded linear values, so it
is lossless: the linear value is `(code/2 − black_p)/(white − black_p)·gain`, and reading the cache back gives load_raw's
values exactly (in the Float32 the pipeline uses). The mask is a separate file next to it; the render steps read only the
images, the fit and the seam score read image and mask. A rerun reuses the cache when a frame's RAW hash and the decode
settings (mode, gain, cache format) match `decode.json`; anything else decodes the frame again.
