# PanoramaStitching.jl

```@docs
PanoramaStitching
```

PanoramaStitching estimates the rotations of a camera that only rotated from its frames, and renders the panorama in any
view, through a calibrated lens of any field of view (fisheyes past 90° included). It also fits the camera's vignetting
from the same frames, with no flat field. All geometry is on unit bearings, and every residual is in the camera's pixels.

| Step | Function | Result |
|---|---|---|
| the rotation of every frame | [`register_rotations`](@ref) | a [`Registration`](@ref): rotations, view graph, tracks, bundle |
| the frames sampled at directions | [`sample_frames`](@ref) | [`FrameSamples`](@ref) |
| the panorama in a view | [`panorama_view`](@ref), [`render_panorama`](@ref), [`render_cube_map`](@ref) | `(; image, owner, coverage, extent)` |
| the vignetting from the frames alone | [`fit_vignetting`](@ref) | a `FitResult` of a [`SplineVignetting`](@ref) |
| the seams a vignetting leaves | [`score_seams`](@ref) | `(; ratio, rms, mad, n)` |

The sampling, the render and the vignetting fit run as KernelAbstractions kernels (the first two through
VisualGeometryToolkitCore's warping machinery): GPU arrays run on the device, host arrays run the same code on the CPU.

## Installation

PanoramaStitching is registered in [CauRegistry](https://github.com/prittjam/CauRegistry.jl); the packages it is built
on are in General. Julia 1.12 or later. Add both registries once (naming General keeps it in a new depot), then the
package and the two packages the example also loads:

```julia-repl
pkg> registry add General https://github.com/prittjam/CauRegistry.jl
pkg> add PanoramaStitching VisualGeometryToolkitCore StaticArrays
```

## A first example

A fisheye sweeps a synthetic scene in four frames 35° apart. The example estimates the frames' rotations from the
frames alone and renders their cylindrical panorama.

```julia
using PanoramaStitching
using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, Circle, Point2, Equirectangular, Cylindrical, HuberShape,
    CauchyShape, backproject, project, bilinear_sample, exp_so3, log_so3, px
using StaticArrays: SA
using LinearAlgebra: norm
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

# A Kannala–Brandt fisheye (300 px focal length, 1000 × 1000 sensor, imaging to 100° off axis) turned 35° between
# frames, each frame 2 × 2 binned: frame pixel (i, j) is centred at camera pixel (2j − 1, 2i − 1).
camera = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
circle = Circle(Point2(500.0, 500.0), 480.0)                     # the image circle, camera pixels
truth = [exp_so3(SA[0.0, deg2rad(35.0 * k), 0.0]) for k in 0:3]  # d = R_k·f, the first frame the common frame
frames = map(truth) do R
    rgb = [hypot(2j - 501, 2i - 501) < 480 ? radiance(R * backproject(camera, SA[2j - 1.0, 2i - 1.0])) : SA[0f0, 0f0, 0f0]
           for i in 1:500, j in 1:500]
    Float32[rgb[i, j][c] for i in 1:500, j in 1:500, c in 1:3]   # H × W × 3, linear
end

# The rotations from the frames alone, then the cylindrical panorama at 0.25° per pixel.
reg = register_rotations(frames, camera; raw_scale = 2, circle, tone_percentile = 99.5, detector_margin_px = 20.0,
                         asift_tilts = [(1.0, 0.0)], max_detections = 1024, descriptor = :b, matcher = :b128,
                         min_score = 0.0, pairs = :all, max_lag = 1, overlap_deg = 120.0, min_inliers = 40, seed = 1,
                         max_trials = 10_000, min_trials = 100, averaging_loss = HuberShape(), bundle_loss = CauchyShape(),
                         bundle_scale_multiple = 3.0, bundle_iterations = 3, bundle_max_iter = 100, bundle_ftol = 1e-10,
                         bundle_xtol = 1e-10)
errors = [rad2deg(norm(log_so3(reg.rotations[k]' * truth[k]))) for k in 1:4]   # degrees, each below 0.05
reg.bundle.rms                                                   # the bundle's rms reprojection error, below 1 px
cyl = panorama_view(Cylindrical(), (-100.0, 205.0, -45.0, 45.0), 0.25)
pano = render_panorama(camera, frames, reg.rotations, cyl; vignetting = (u, f) -> 1.0, circle, raw_scale = 2,
                       rim_margin_px = 12.0, fov_deg = 89.0)
size(pano.image)                                                 # (458, 1220, 3); NaN where no frame sees the pixel
```

## Documentation

- **Start here**: [Getting started](guide.md) continues from the example, step by step: the inputs, the registration,
  sampling the frames at directions, the panorama views and the vignetting fit.
- **Reference**: the [API](api.md), every public function and type.
- **Worked example**: the repository's `examples/panorama_vignetting/` runs the whole chain, from RAW decode to corrected
  panoramas, on two RAW sweeps from configs of record.

## Role in the stack

PanoramaStitching is part of the [VisualGeometryToolkit](https://visualgeometrytoolkit.github.io) stack. It composes
VisualGeometryToolkitFeatures' ASIFT detector and LoMa descriptor and matcher, VisualGeometryToolkitEstimation's
`find_rotation` and `fit_rotation_averaging`, and VisualGeometryToolkitCore's cameras, panorama projections and warping.
Its one radiometric estimate is the vignetting a rotating camera's own frames determine; exposure, response and colour are
left to the caller.

## Citing

PanoramaStitching is written by James Pritts. Its
[CITATION.cff](https://github.com/VisualGeometryToolkit/PanoramaStitching.jl/blob/main/CITATION.cff) is the source of
this entry:

```bibtex
@software{pritts2026panoramastitching,
  author = {Pritts, James},
  title  = {{PanoramaStitching.jl}: Panoramas and Vignetting of a Rotating Camera},
  url    = {https://github.com/VisualGeometryToolkit/PanoramaStitching.jl},
  year   = {2026}
}
```

MIT licence.
