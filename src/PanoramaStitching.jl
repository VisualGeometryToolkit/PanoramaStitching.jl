"""
    PanoramaStitching

The panorama of a camera that only rotated (a sweep of a distant scene), seen
through a calibrated lens of any field of view: the rotations of its frames
(`register_rotations` — features, pairwise rotations, the view graph's rotation
averaging, the tracks and the rotation and direction bundle), the frames sampled
at directions of the sphere (`sample_frames`, the observations of a vignetting
validation), and the panorama by the reverse warp of the frames through the
camera into any view — the cylindrical family, a virtual camera, a cube map —
unblended (`render_panorama`, `render_cube_map`); the camera's vignetting
(VisualRadiometryToolkitCore's model) and the frames' exposure gains from the
frames alone, interpolation free, or the gains alone under a given vignetting
(`VignettingSelfCalibration`), and the seams a vignetting leaves (`score_seams`); the
results saved and loaded as a `vgio/panorama/v2.0` file (`save_panorama`,
`load_panorama`), each part written by the kind that owns it.

Built on the VisualGeometryToolkit stack: VisualGeometryToolkitFeatures' ASIFT
detector and LoMa descriptor and matcher, VisualGeometryToolkitEstimation's
`Ransac` on a `RotationProblem`, `ViewGraph` and `RotationAveraging`, VisualGeometryToolkitCore's
estimation interface (`solve(problem, algorithm)`: the rotation bundle is a `RotationBundleProblem` refined by its
`LevenbergMarquardt`, the vignetting fits are algorithms of this package),
cameras, rotation charts, panorama projections and warping
machinery (`image_warp`, `image_sample!`, `pointwise!`: KernelAbstractions
kernels on GPU arrays, the same code on the host), VisualRadiometryToolkitCore's
vignetting models, VisualGeometryToolkitIO's file envelope, camera, trajectory and
fit records and VisualRadiometryToolkitIO's vignetting body (the panorama
record); KernelAbstractions.jl and
Atomix.jl (the vignetting fit's normal equations), Adapt.jl (a camera's forward
map moved to the frames' backend), SparseArrays (the fit's host Cholesky solve),
NLLSsolver.jl (the rotation bundle) and StableRNGs.jl (the pairs'
RANSAC samplers, seeded by VisualGeometryToolkitCore's `stable_seed`).
"""
module PanoramaStitching

using LinearAlgebra: LinearAlgebra, I, norm, normalize, dot
using Statistics: Statistics, quantile
using StatsBase: StatsBase
using StableRNGs: StableRNG
using Static: static
using StaticArrays: SVector, SMatrix, StaticVector
using Rotations: RotMatrix
using CoordinateTransformations: AffineMap, LinearMap
import NLLSsolver
import KernelAbstractions as KA
import Adapt
using KernelAbstractions: @kernel, @index, @Const
import Atomix
import SparseArrays
import ForwardDiff
import VisualGeometryToolkitCore as VgtkCore
import VisualRadiometryToolkitCore as VrtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
import VisualGeometryToolkitFeatures as VgtkFeatures
import VisualGeometryToolkitIO as VgtkIO
import VisualRadiometryToolkitIO as VrtkIO
using JSON: JSON                                                # src/panorama_record.jl: the bodies lowered and lifted on VgtkIO's style
using PrecompileTools: @setup_workload, @compile_workload       # src/precompile.jl

export register_rotations, Registration, verify_pairs, VerifiedPairs, AllPairs, LagOverlap, TrackCheck
export chain_tracks, RotationBundleProblem, RotationBundleMetadata
export world_direction, FrameSamples, sample_frames, fibonacci_directions
export true_up, up_frame, SweepUp, RollFreeUp, PanoramaView, panorama_view, render_panorama, render_cube_map, coverage_extent
export VignettingSelfCalibration, AzimuthalChart, VignettingMetadata, SparseCholesky, PositionNoise
export IntensityGains, ChannelGains
export score_seams, vignetting_summary
export save_panorama, load_panorama
export PairCorrespondences, ring_correspondences, RingVignetting, RingPairMedians, AdditiveLevel, LevelBisection, additive_level_slope,
       PairVignettingMetadata
public PANORAMA_KIND, panorama_json_schema
public detect_features, pair_bearings, frame_region, CUBE_FACES, GAIN_CORRECTIONS, OBSERVATION_NOISES, refuse_retired_registration

include("tracks.jl")
include("sampling.jl")
include("rotation_bundle.jl")
include("features.jl")
include("registration.jl")
include("panorama.jl")
include("vignetting.jl")
include("seams.jl")
include("pair_vignetting.jl")
include("panorama_record.jl")        # vgio/panorama/v2.0: the results saved and loaded, each part by its owner's kind
include("precompile.jl")             # the precompile workload: the entry points' native code, cached in the package image

end # module PanoramaStitching
