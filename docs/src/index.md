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

[Getting started](guide.md) walks through all three on a fisheye sweep; the [API](api.md) page documents every function.

PanoramaStitching is part of the [VisualGeometryToolkit](https://visualgeometrytoolkit.github.io) stack. It does no
radiometry: a vignetting or exposure fit consumes its rotations, direction samples and panoramas downstream. MIT licence.
