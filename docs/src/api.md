# API

## Registration

```@docs
register_rotations
Registration
ViewGraph
chain_tracks
refine_rotations
RotationBundleMetadata
```

## Features

```@docs
PanoramaStitching.FrameFeatures
PanoramaStitching.detect_features
PanoramaStitching.pair_bearings
PanoramaStitching.LOMA_SIZE
```

## Sampling the frames at directions

```@docs
world_direction
fibonacci_directions
FrameSamples
sample_frames
```

## The panorama

```@docs
true_up
up_frame
PanoramaView
panorama_view
render_panorama
render_cube_map
PanoramaStitching.CUBE_FACES
```

## The vignetting from the frames

```@docs
fit_vignetting
RadialVignetting
TensorVignetting
SplineVignetting
log_vignetting
AzimuthalChart
VignettingMetadata
ConjugateGradient
SparseCholesky
score_seams
```
