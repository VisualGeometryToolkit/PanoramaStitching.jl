# PanoramaStitching.jl

[![CI](https://github.com/VisualGeometryToolkit/PanoramaStitching.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/VisualGeometryToolkit/PanoramaStitching.jl/actions/workflows/CI.yml)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://visualgeometrytoolkit.github.io/PanoramaStitching.jl/dev/)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/PanoramaStitching.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/VisualGeometryToolkit/PanoramaStitching.jl)

PanoramaStitching estimates the rotations of a camera that only rotated from its frames, and renders the panorama
in any view, through a calibrated lens of any field of view (fisheyes past 90° included). With no calibration target, the
same frames also refine the camera itself (autocalibration), and they fit its vignetting with no flat field. It is part
of the
[VisualGeometryToolkit](https://visualgeometrytoolkit.github.io) stack.

## Installation

PanoramaStitching is registered in [CauRegistry](https://github.com/prittjam/CauRegistry.jl); the packages it is built
on are in General. Julia 1.12 or later. With CauRegistry added to your depot, add the package and the packages the
example also loads:

```julia-repl
pkg> add PanoramaStitching VisualGeometryToolkitCore StaticArrays CoordinateTransformations
```

## Example

A fisheye sweeps a synthetic scene in four frames 35° apart. The example estimates the frames' rotations from the
frames alone and renders their cylindrical panorama.

```julia
using PanoramaStitching
using VisualGeometryToolkitCore: CameraModel, BoundedCamera, Size2, KannalaBrandt, Circle, Point2, Equirectangular,
    Cylindrical, CauchyShape, Triggs, SimilarityMap, backproject, project, bilinear_sample, exp_so3, log_so3, px
using StaticArrays: SA
using CoordinateTransformations: LinearMap
using LinearAlgebra: I, norm
using Random: Xoshiro

# The scene: random Gaussian blobs on an equirectangular texture (columns the longitude, rows the latitude).
function blob_scene(rng; width = 2048, height = 1024, n = 20_000)
    scene = fill(0.15f0, height, width, 3)
    for _ in 1:n
        x, y, s, a, c = width * rand(rng), height * rand(rng), 1 + 3rand(rng), 0.6rand(rng), 0.3 .+ 0.7rand(rng, 3)
        sx = s / max(sinpi(y / height), 0.05)                    # round on the sphere, not on the texture
        for j in floor(Int, x - 3sx):ceil(Int, x + 3sx), i in max(1, floor(Int, y - 3s)):min(height, ceil(Int, y + 3s))
            scene[i, mod1(j, width), :] .+= a * exp(-((j - x)^2 / sx^2 + (i - y)^2 / s^2) / 2) .* c
        end
    end
    return d -> (p = project(Equirectangular(), d); bilinear_sample(scene, Val(3), SA[(p[1] + π) / 2π, 0.5 - p[2] / π]))
end
radiance = blob_scene(Xoshiro(7))

# A Kannala–Brandt fisheye (300 px focal length, 1000 × 1000 sensor, imaging to 100° off axis, its image circle of
# radius 480 px the lens coverage: every mask reads the camera's imaged region) turned 35° between frames, each frame
# 2 × 2 binned: frame pixel (i, j) is centred at camera pixel (2j − 1, 2i − 1), the frames' map onto the camera's pixels
# x ↦ 2x (a RAW's :bin2 frames: VisualGeometryToolkitFeatures' raw_frame_map).
camera = BoundedCamera(CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02)), Size2(1000, 1000);
                       lens_coverage = Circle(Point2(500.0, 500.0), 480.0))
truth = [exp_so3(SA[0.0, deg2rad(35.0 * k), 0.0]) for k in 0:3]  # d = R_k·f, the first frame the common frame
frames = map(truth) do R
    rgb = [hypot(2j - 501, 2i - 501) < 480 ? radiance(R * backproject(camera, SA[2j - 1.0, 2i - 1.0])) : SA[0f0, 0f0, 0f0]
           for i in 1:500, j in 1:500]
    Float32[rgb[i, j][c] for i in 1:500, j in 1:500, c in 1:3]   # H × W × 3, linear
end
frame_map = SimilarityMap(LinearMap(2.0I))

# The rotations from the frames alone, then the cylindrical panorama at 0.25° per pixel.
reg = register_rotations(frames, camera; frame_map, tone_percentile = 99.5, detector_margin_px = 20.0,
                         asift_tilts = [(1.0, 0.0)], max_detections = 1024, descriptor = :b, matcher = :b128,
                         min_score = 0.0, pairs = :all, max_lag = 1, overlap_deg = 120.0, min_inliers = 40, seed = 1,
                         max_trials = 10_000, min_trials = 100,
                         bundle_optimization_strategy = Triggs{CauchyShape}(efficiency_constant = 2.5), bundle_max_iter = 1000,
                         bundle_ftol = 1e-10, bundle_xtol = 1e-10)
errors = [rad2deg(norm(log_so3(reg.rotations[k]' * truth[k]))) for k in 1:4]   # degrees, each below 0.05
reg.bundle.rms                                                   # the bundle's rms reprojection error, below 1 px
cyl = panorama_view(Cylindrical(), (-100.0, 205.0, -45.0, 45.0), 0.25)
pano = render_panorama(camera, frames, reg.rotations, cyl; vignetting = (u, f) -> 1.0, frame_map, rim_margin_px = 12.0,
                       fov_deg = 89.0)
size(pano.image)                                                 # (458, 1220, 3); NaN where no frame sees the pixel
```

[Getting started](https://visualgeometrytoolkit.github.io/PanoramaStitching.jl/dev/guide/) continues from here.

## Documentation

- [Getting started](https://visualgeometrytoolkit.github.io/PanoramaStitching.jl/dev/guide/): the registration,
  sampling the frames at directions, the panorama views and the vignetting fit, step by step.
- [API reference](https://visualgeometrytoolkit.github.io/PanoramaStitching.jl/dev/api/).

The features come from VisualGeometryToolkitFeatures, the pairwise rotations and the rotation averaging from
VisualGeometryToolkitEstimation, and the cameras, panorama projections and warping from VisualGeometryToolkitCore. The
[portal](https://visualgeometrytoolkit.github.io) links every package of the stack. The repository's
`examples/panorama_vignetting/` runs the whole chain on RAW sweeps from configs of record.

## Citing

PanoramaStitching is written by James Pritts. Its
[CITATION.cff](https://github.com/VisualGeometryToolkit/PanoramaStitching.jl/blob/main/CITATION.cff) is the source of
this entry, and GitHub's "Cite this repository" button exports it:

```bibtex
@software{pritts2026panoramastitching,
  author = {Pritts, James},
  title  = {{PanoramaStitching.jl}: Panoramas and Vignetting of a Rotating Camera},
  url    = {https://github.com/VisualGeometryToolkit/PanoramaStitching.jl},
  year   = {2026}
}
```

## License

MIT
