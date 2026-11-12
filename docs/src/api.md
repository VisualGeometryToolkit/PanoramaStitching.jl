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
PanoramaStitching.FrameFeatures
PanoramaStitching.detect_features
PanoramaStitching.pair_bearings
PanoramaStitching.keypoint_covariance_scale
PanoramaStitching.keypoint_uncertain
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
PanoramaView
panorama_view
render_panorama
render_cube_map
PanoramaStitching.CUBE_FACES
```

## The vignetting from the frames

```@docs
fit_vignetting
AzimuthalChart
VignettingMetadata
ConjugateGradient
SparseCholesky
IntensityNoise
IntensityAndPositionNoise
IntensityGains
ChannelGains
PanoramaStitching.GAIN_CORRECTIONS
PanoramaStitching.MEASUREMENT_NOISES
score_seams
```
