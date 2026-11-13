# The frames' and the planes' maps onto the camera's pixels of the suite's 2 × 2-binned synthetic frames: VisualGeometryToolkitCore's
# `raw_frame_map` and `bayer_plane_map` of a `:bin2` LinearRaw of a synthetic BGGR mosaic (R at (+¼, +¼) cell pixels from the
# cell centre, B at (−¼, −¼)), and the identity map of frames already at the camera's resolution.

@testsetup module BayerMaps
using StaticArrays: SVector
using Rotations: RotMatrix
import VisualGeometryToolkitCore as VgtkCore
export FRAME_MAP, PLANE_MAPS, IDENTITY_MAP, detection_pixels

const BGGR = VgtkCore.LinearRaw(VgtkCore.RawMosaic(zeros(UInt16, 4, 4), UInt8[2 1; 3 0], (0.0, 0.0, 0.0, 0.0), 1.0, (0, 0),
                                                   (4, 4), (; model = "synthetic")); mode = :bin2)
const FRAME_MAP = VgtkCore.raw_frame_map(BGGR)                                     # x ↦ 2x
const PLANE_MAPS = ntuple(c -> VgtkCore.bayer_plane_map(BGGR, c), 3)
const IDENTITY_MAP = VgtkCore.SimilarityMap(1.0, one(RotMatrix{2, Float64}), SVector(0.0, 0.0))            # a frame at the camera's own pixels
# A frame's detections' centres (its described features) in the camera's pixels, through the frame's map.
detection_pixels(features, frame_map = FRAME_MAP) = [frame_map(SVector{2, Float64}(d.center...)) for d in features]
end
