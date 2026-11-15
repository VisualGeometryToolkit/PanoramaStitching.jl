# =============================================================================
# A decentred lens (radial and tangential terms): VisualGeometryToolkitCore 0.10.10's valid domain of a lens that is not
# radially symmetric is the star `PolygonRegion` through each azimuth's own fold. The autocalibration's camera variable holds
# it, its LM update builds it again, a trial step whose domain Core refuses is a rejected step (+Inf, never an error), and
# detect_features masks the imaged region that holds it.
# =============================================================================

@testsetup module DecentredLens
using StaticArrays: @SMatrix
import VisualGeometryToolkitCore as VgtkCore
export DECENTRED, DECENTRED_FREE
# Core's Brown–Conrady test lens (valid_domain_polygon_tests.jl: radial k1, k2 and tangential p1, p2, its fold radius 17 %
# apart across the azimuths) behind f = 300 px, its principal point the centre of a 1000 × 1000 sensor: the polygon lies
# inside the sensor
const DECENTRED = VgtkCore.CameraModel(VgtkCore.CameraCalibrationMat(@SMatrix([300.0 0.0 500.0; 0.0 300.0 500.0; 0.0 0.0 1.0]));
                                       lens = VgtkCore.Lens(VgtkCore.Perspective(),
                                                            VgtkCore.Additive(VgtkCore.Radial((-0.3, 0.04)),
                                                                              VgtkCore.Tangential(1.0e-2, -5.0e-3))))
const DECENTRED_FREE = (:f, :cx, :cy, :k1, :k2, :p1, :p2)              # every parameter but the aspect and the skew
end

@testitem "a decentred lens: the camera variable holds its polygon domain, an update builds it again, a refused trial domain is a rejected step" setup = [DecentredLens] begin
    using Test, LinearAlgebra, StaticArrays
    import NLLSsolver, PanoramaStitching
    import VisualGeometryToolkitCore as VgtkCore
    chart = VgtkCore.restrict(VgtkCore.ParameterManifold(DECENTRED), DECENTRED_FREE)
    cc = PanoramaStitching._CameraVariable(chart)
    @test cc.domain isa VgtkCore.PolygonRegion && cc.domain == VgtkCore.valid_domain(DECENTRED)
    # a small step (p1 larger by 10⁻³): the polygon of the stepped lens, the variable's one type
    cu = NLLSsolver.update(cc, [0.0, 0, 0, 0, 0, 1.0e-3, 0])
    @test typeof(cu) === typeof(cc) && cu.domain isa VgtkCore.PolygonRegion && cu.domain == VgtkCore.valid_domain(cu.model)
    # an observation of a direction the camera images inside its domain
    d = normalize(SVector(0.2, -0.1, 1.0))
    ch = VgtkCore.SO3Manifold(SMatrix{3, 3, Float64, 9}(I)); dv = VgtkCore.Eliminable{false}(VgtkCore.SphereManifold(d))
    x = VgtkCore.project(DECENTRED, d) + SVector(0.3, -0.2)
    obs = PanoramaStitching._CameraObservation{false, NLLSsolver.NoRobust, Nothing}(x, SVector(1, 2, 3), NLLSsolver.NoRobust(), nothing)
    @test VgtkCore.Point2(VgtkCore.project(DECENTRED, d)...) ∈ cu.domain
    @test isfinite(NLLSsolver.computecost(Any[ch, dv, cu], obs))
    # a trial step to p1 = 0.31: the stepped lens's fold curve is not star-shaped about its centre, and Core refuses its domain.
    # The update holds no domain: the residual is NaN, the block's cost +Inf, a step the LM rejects.
    bad = [0.0, 0, 0, 0, 0, 0.3, 0]
    @test_throws ArgumentError VgtkCore.valid_domain(VgtkCore.model(NLLSsolver.update(chart, bad)))
    cb = NLLSsolver.update(cc, bad)
    @test typeof(cb) === typeof(cc)
    @test all(isnan, NLLSsolver.computeresidual(obs, ch, dv, cb))
    @test NLLSsolver.computecost(Any[ch, dv, cb], obs) == Inf
    # the start is no trial step: a start camera whose domain Core refuses is refused
    @test_throws ArgumentError PanoramaStitching._CameraVariable(NLLSsolver.update(chart, bad))
end

@testitem "a decentred lens: detect_features masks its imaged region (the sensor ∩ its polygon domain)" setup = [DecentredLens, SyntheticSequence, BayerMaps] begin
    using Test, StaticArrays
    using PanoramaStitching: detect_features
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitFeatures as VgtkFeatures
    cam = VgtkCore.BoundedCamera(DECENTRED, VgtkCore.Size2(1000, 1000))
    region = VgtkCore.imaged_region(cam)
    @test region isa VgtkCore.RegionIntersection && region.b isa VgtkCore.PolygonRegion
    frame = sequence(; K = 1).frames[1]                                  # the suite's 2 × 2-binned frame: an image to detect on
    N = 200
    desc = VgtkFeatures.LoMaDescriptor(:b; max_detections = N, execution_provider = :cpu)
    fs = detect_features(cam, frame, desc; frame_map = FRAME_MAP, tone_percentile = 99.5, detector_margin_px = 20.0,
                         asift_tilts = [(1.0, 0.0)], max_detections = N)
    px = detection_pixels(fs)
    @test length(px) > N ÷ 2
    @test all(p -> VgtkCore.Point2(p...) ∈ region, px)                 # every detection inside the polygon and the sensor
end
