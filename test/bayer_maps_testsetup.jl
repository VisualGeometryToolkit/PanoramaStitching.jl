# The frames' and the planes' maps onto the camera's pixels of the suite's 2 × 2-binned synthetic frames: VisualGeometryToolkitCore's
# `raw_frame_map` and `bayer_plane_map` of a `:bin2` LinearRaw of a synthetic BGGR mosaic (R at (+¼, +¼) cell pixels from the
# cell centre, B at (−¼, −¼)), and the identity map of frames already at the camera's resolution.

@testsetup module BayerMaps
using StaticArrays: SVector
using Rotations: RotMatrix
import VisualGeometryToolkitCore as VgtkCore
export FRAME_MAP, PLANE_MAPS, IDENTITY_MAP, detection_pixels, rggb_mosaic

const BGGR = VgtkCore.LinearRaw(VgtkCore.RawMosaic(zeros(UInt16, 4, 4), UInt8[2 1; 3 0], (0.0, 0.0, 0.0, 0.0), 1.0, (0, 0),
                                                   (4, 4), (; model = "synthetic")); mode = :bin2)
const FRAME_MAP = VgtkCore.raw_frame_map(BGGR)                                     # x ↦ 2x
const PLANE_MAPS = ntuple(c -> VgtkCore.bayer_plane_map(BGGR, c), 3)
const IDENTITY_MAP = VgtkCore.SimilarityMap(1.0, one(RotMatrix{2, Float64}), SVector(0.0, 0.0))            # a frame at the camera's own pixels
# A frame's detections' centres (its described features; Core's `center` of the frame, the FeatureFrame interface) in the
# camera's pixels, through the frame's map.
detection_pixels(features, frame_map = FRAME_MAP) = [frame_map(SVector{2, Float64}(VgtkCore.center(VgtkCore.feature_frame(d)))) for d in features]
# An H × W × 3 linear frame as a 2H × 2W RGGB mosaic (black 512, white 16383): each photosite its plane's value (G twice), so
# the mosaic's :bin2 LinearRaw is the frame, to the 14-bit codes' rounding.
function rggb_mosaic(F; black = 512.0, white = 16383.0)
    code(v) = UInt16(round(black + clamp(v, 0, 1) * (white - black)))
    Hf, Wf = size(F, 1), size(F, 2); raw = Matrix{UInt16}(undef, 2Hf, 2Wf)
    for j in 1:Wf, i in 1:Hf
        raw[2i - 1, 2j - 1] = code(F[i, j, 1]); raw[2i - 1, 2j] = code(F[i, j, 2])
        raw[2i, 2j - 1] = code(F[i, j, 2]); raw[2i, 2j] = code(F[i, j, 3])
    end
    return VgtkCore.RawMosaic(raw, UInt8[0 1; 3 2], (black, black, black, black), white, (0, 0), (2Hf, 2Wf), (; model = "synthetic"))
end
end
