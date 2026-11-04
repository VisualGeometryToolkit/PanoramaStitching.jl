# PanoramaStitching.jl

```@docs
PanoramaStitching
```

The package registers a pure-rotation sequence and renders its panorama:

| Step | Function | Result |
|---|---|---|
| the rotation of every frame | [`register_rotations`](@ref) | a [`Registration`](@ref): rotations, view graph, tracks, bundle |
| the frames sampled at directions | [`sample_frames`](@ref) | [`FrameSamples`](@ref) |
| the panorama in a view | [`panorama_view`](@ref), [`render_panorama`](@ref), [`render_cube_map`](@ref) | `(; image, owner, coverage, extent)` |
| the vignetting from the frames alone | [`fit_vignetting`](@ref) | a `FitResult` of a [`SplineVignetting`](@ref) |
| the seams a vignetting leaves | [`score_seams`](@ref) | `(; ratio, rms, mad, n)` |

[Getting started](guide.md) walks through them on a fisheye sweep; the [API](api.md) page documents every function. The
repository's `examples/panorama_vignetting/` runs the whole chain on two RAW sweeps from configs of record.

PanoramaStitching is part of the [VisualGeometryToolkit](https://visualgeometrytoolkit.github.io) stack. Its one radiometric
estimate is the vignetting a rotating camera's own frames determine (`fit_vignetting`, the scene and V fitted together,
interpolation free); exposure, response and colour are left to the caller. MIT licence.
