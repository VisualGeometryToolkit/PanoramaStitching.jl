# API

## Registration

```@docs
register_rotations
Registration
verify_pairs
VerifiedPairs
AllPairs
LagOverlap
chain_tracks
RotationBundleProblem
RotationBundleMetadata
TrackCheck
PanoramaStitching.refuse_retired_registration
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
SweepUp
RollFreeUp
PanoramaView
panorama_view
render_panorama
render_cube_map
coverage_extent
PanoramaStitching.CUBE_FACES
```

## The vignetting from the frames

```@docs
VignettingSelfCalibration
AzimuthalChart
VignettingMetadata
SparseCholesky
PositionNoise
IntensityGains
ChannelGains
PanoramaStitching.GAIN_CORRECTIONS
PanoramaStitching.OBSERVATION_NOISES
score_seams
vignetting_summary
```

## The model-free vignetting: the pair equation

```@docs
ring_correspondences
PairCorrespondences
RingVignetting
RingPairMedians
PairVignettingMetadata
AdditiveLevel
LevelBisection
```

## The results of record

```@docs
save_panorama
load_panorama
PanoramaStitching.PANORAMA_KIND
PanoramaStitching.panorama_json_schema
```
