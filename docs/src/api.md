# API

## Registration

```@docs
register_rotations
Registration
verify_pairs
VerifiedPairs
chain_tracks
refine_rotations
refine_camera_and_rotations
RotationBundleMetadata
```

## Features

```@docs
PanoramaStitching.detect_features
PanoramaStitching.pair_bearings
```

## Sampling the frames at directions

```@docs
world_direction
fibonacci_directions
FrameSamples
sample_frames
PanoramaStitching.frame_region
```

## The panorama

```@docs
true_up
up_frame
sweep_up
PanoramaView
panorama_view
render_panorama
render_cube_map
coverage_extent
PanoramaStitching.CUBE_FACES
```

## The vignetting from the frames

```@docs
fit_vignetting
AzimuthalChart
VignettingMetadata
ConjugateGradient
SparseCholesky
PositionNoise
IntensityGains
ChannelGains
PanoramaStitching.GAIN_CORRECTIONS
PanoramaStitching.OBSERVATION_NOISES
score_seams
vignetting_summary
```
