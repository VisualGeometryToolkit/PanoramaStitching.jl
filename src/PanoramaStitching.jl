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
(VisualGeometryToolkitCore's model) and the frames' exposure gains from the
frames alone, interpolation free, or the gains alone under a given vignetting
(`fit_vignetting`), and the seams a vignetting leaves (`score_seams`).

Built on the VisualGeometryToolkit stack: VisualGeometryToolkitFeatures' ASIFT
detector and LoMa descriptor and matcher, VisualGeometryToolkitEstimation's
`find_rotation` and `fit_rotation_averaging`, VisualGeometryToolkitCore's
cameras, vignetting models, rotation charts, panorama projections and warping
machinery (`image_warp`, `image_sample!`, `pointwise!`: KernelAbstractions
kernels on GPU arrays, the same code on the host); KernelAbstractions.jl and
Atomix.jl (the vignetting fit's normal equations), Adapt.jl (a camera's forward
map moved to the frames' backend), SparseArrays (the fit's host Cholesky solve),
Graphs.jl, DataStructures.jl, NLLSsolver.jl and StableRNGs.jl (the pairs'
RANSAC samplers, seeded by VisualGeometryToolkitCore's `stable_seed`).
"""
module PanoramaStitching

using LinearAlgebra: LinearAlgebra, I, norm, normalize, dot, cross
using Statistics: Statistics, quantile
using StableRNGs: StableRNG
using Static: static
using StaticArrays: SVector, SMatrix, StaticVector
using Rotations: RotMatrix
using CoordinateTransformations: AffineMap, LinearMap, Translation
import Graphs
import DataStructures
import NLLSsolver
import KernelAbstractions as KA
import Adapt
using KernelAbstractions: @kernel, @index, @Const
import Atomix
import SparseArrays
import ForwardDiff
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
import VisualGeometryToolkitFeatures as VgtkFeatures
using PrecompileTools: @setup_workload, @compile_workload       # src/precompile.jl

export register_rotations, Registration
export ViewGraph, chain_tracks, refine_rotations, RotationBundleMetadata
export world_direction, FrameSamples, sample_frames, fibonacci_directions
export true_up, up_frame, PanoramaView, panorama_view, render_panorama, render_cube_map
export fit_vignetting, AzimuthalChart, VignettingMetadata, ConjugateGradient, SparseCholesky, IntensityWeights, SampsonWeights
export FrameGains, ChannelGains
export score_seams
public FrameFeatures, detect_features, pair_bearings, LOMA_SIZE, CUBE_FACES, GAIN_CORRECTIONS, WEIGHTS

include("view_graph.jl")
include("sampling.jl")
include("rotation_bundle.jl")
include("features.jl")
include("registration.jl")
include("panorama.jl")
include("vignetting.jl")
include("seams.jl")
include("precompile.jl")             # the precompile workload: the entry points' native code, cached in the package image

end # module PanoramaStitching
