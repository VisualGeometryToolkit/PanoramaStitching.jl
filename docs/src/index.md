# PanoramaStitching.jl

```@docs
PanoramaStitching
```

The package registers a pure-rotation sequence and renders its panorama in three calls:

| Step | Function | Result |
|---|---|---|
| the rotation of every frame | [`register_rotations`](@ref) | a [`Registration`](@ref): rotations, view graph, tracks, bundle |
| world cells sampled in every frame | [`sample_cells`](@ref) | [`CellSamples`](@ref) |
| the equirectangular panorama | [`render_panorama`](@ref) | `(image, extent, composite)` |

[Getting started](guide.md) walks through all three on a fisheye sweep; the [API](api.md) page documents every function.

PanoramaStitching is part of the [VisualGeometryToolkit](https://visualgeometrytoolkit.github.io) stack. It does no
radiometry: a vignetting or exposure fit consumes its rotations, cells and panoramas downstream. MIT licence.
