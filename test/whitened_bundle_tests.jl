# =============================================================================
# The whitened rotation bundle (src/rotation_bundle.jl): observations that carry their
# detections' pixel covariances (`frame => Uncertain(pixel, CovMat(Σ))`) are whitened by L⁻¹ (Σ = LLᵀ). On a synthetic fisheye
# sweep with known anisotropic, heteroscedastic detection noise the whitened bundle recovers the rotations better than the
# unweighted one; its closed-form Jacobian matches ForwardDiff; the refusals; register_rotations' observations plain pixels
# (detection covariances off in this release), with no consumer default.
# =============================================================================

@testitem "the rotation bundle whitened by known anisotropic detection covariances beats the unweighted bundle" setup = [SweepScene] begin
    using Test, Random, LinearAlgebra, StaticArrays, Statistics
    import Rotations
    using PanoramaStitching: RotationBundleProblem
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode

    # per observation a covariance Σ = Q diag(σ₁², σ₂²) Qᵀ, σ₁ ∈ [0.1, 0.3] px, σ₂ ∈ [1, 3] px at a random angle, the noise L·z
    function anisotropic_tracks(rng, R, n)
        clean, dirs, _ = sweep_tracks(rng, R, n; σ = 0.0)
        tracks = map(clean) do t
            map(t) do (k, u)
                φ = 2π * rand(rng); Q = SMatrix{2, 2}(cos(φ), sin(φ), -sin(φ), cos(φ))
                Σ = Q * Diagonal(SVector((0.1 + 0.2rand(rng))^2, (1 + 2rand(rng))^2)) * Q'
                Σ = (Σ + Σ') / 2
                k => VgtkCore.Uncertain(u + cholesky(Symmetric(Σ)).L * randn(rng, SVector{2, Float64}), VgtkCore.CovMat(SMatrix{2, 2, Float64, 4}(Σ)))
            end
        end
        return tracks, dirs
    end
    lm = LevenbergMarquardt(; optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}())
    kw = (; width = 3.0, stops = LmStops(; max_iter = 1000, ftol = 1.0e-12, xtol = 1.0e-10))
    K = 5
    eu = Float64[]; ew = Float64[]
    for seed in 1:4
        rng = Xoshiro(100 + seed)
        R = sweep_rotations(rng, K)
        tracks, _ = anisotropic_tracks(rng, R, 300)
        start = Dict(k => SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec((0.01 .* randn(rng, 3))...))) * R[k] for k in 1:K)
        start[1] = R[1]
        err(Rs) = sqrt(mean(k -> rad2deg(Rotations.rotation_angle(Rotations.RotMatrix{3}(SMatrix{3, 3}(Rs[k])' * R[k])))^2, 2:K))
        plain = solve(RefineProblem(RotationBundleProblem(CAM, [[k => VgtkCore.geom(x) for (k, x) in t] for t in tracks]), start), lm; kw...)
        white = solve(RefineProblem(RotationBundleProblem(CAM, tracks), start), lm; kw...)
        @test successful_retcode(plain) && successful_retcode(white)
        push!(eu, err(plain.u)); push!(ew, err(white.u))
        # the residuals the metadata reports are pixels in both; the whitened loss width is in units of the covariances'
        @test length(white.metadata.residuals) == length(tracks)
        @test 1.0 < white.metadata.loss_width < 6.0                          # 3 × the Rayleigh scale of unit-covariance residuals
    end
    # measured (2026-11-04): the whitened rotations' rms error is a quarter to a half of the unweighted one's on every seed
    @test all(ew .< eu)
    @test mean(ew) < 0.75 * mean(eu)
end

@testitem "the whitened rotation bundle: the closed-form Jacobian; the refusals" setup = [SweepScene] begin
    using Test, Random, LinearAlgebra, StaticArrays
    import ForwardDiff, NLLSsolver
    using PanoramaStitching: RotationBundleProblem
    import PanoramaStitching
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt, LmStops

    rng = Xoshiro(5)
    R = sweep_rotations(rng, 3)
    d = normalize(SMatrix{3, 3}(R[2]) * SVector(0.3, -0.2, 1.0))
    ch = VgtkCore.SO3Manifold(SMatrix{3, 3, Float64, 9}(R[2])); dv = VgtkCore.Eliminable{false}(VgtkCore.SphereManifold(d))
    x = VgtkCore.project(CAM, SMatrix{3, 3}(R[2])' * d) + SVector(0.3, -0.2)
    Σ = SMatrix{2, 2, Float64, 4}(2.0, 0.7, 0.7, 0.5)
    Li = PanoramaStitching._observation_whitening(VgtkCore.Uncertain(x, VgtkCore.CovMat(Σ)))
    @test Li * Σ * Li' ≈ I(2)
    fwd = PanoramaStitching._forward_map(CAM)
    bc = PanoramaStitching._BundleCamera(CAM)
    obs = PanoramaStitching._Observation{false, typeof(bc), typeof(fwd), Nothing, typeof(Li)}(bc, fwd, x, SVector(1, 2), nothing, Li)
    e, J = NLLSsolver.computeresjac(NLLSsolver.static(3), obs, ch, dv)
    @test e ≈ NLLSsolver.computeresidual(obs, ch, dv)
    f = updated_residual(obs, ch, dv)
    @test J ≈ ForwardDiff.jacobian(f, zeros(5)) rtol = 1.0e-7

    lm = LevenbergMarquardt(; optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}())
    stops = LmStops(; max_iter = 1000, ftol = 1.0e-12, xtol = 1.0e-10)
    bundle(tracks) = solve(RefineProblem(RotationBundleProblem(CAM, tracks), R), lm; width = 3.0, stops)
    u1 = VgtkCore.project(CAM, SMatrix{3, 3}(R[1])' * d); u2 = VgtkCore.project(CAM, SMatrix{3, 3}(R[2])' * d)
    unc(u, S) = VgtkCore.Uncertain(u, VgtkCore.CovMat(S))
    @test_throws ArgumentError bundle([[1 => u1, 2 => unc(u2, Σ)]])                     # some, not all
    @test_throws ArgumentError bundle([[1 => unc(u1, Σ), 2 => unc(u2, fill(NaN, SMatrix{2, 2, Float64, 4}))]])
    @test_throws ArgumentError bundle([[1 => unc(u1, Σ), 2 => unc(u2, SMatrix{2, 2, Float64, 4}(1.0, 2.0, 2.0, 1.0))]])
    # the track check refuses whitened observations: its cut compares pixel residuals with the whitened noise scale
    whitened = [[1 => unc(u1, Σ), 2 => unc(u2, Σ)]]
    @test bundle(whitened) isa VgtkCore.FitResult                                         # the bundle itself takes them
    err = try
        solve(RefineProblem(RotationBundleProblem(CAM, whitened), R), PanoramaStitching.TrackCheck(lm); width = 3.0, stops); nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("Uncertain", err.msg) && occursin("unwhitened pixel residual", err.msg)
end

@testitem "detection covariances off: the pairs and the bundle read plain pixels, no consumer default" setup = [SweepScene, BayerMaps] begin
    # This release runs the detector at its default (no detection covariance): the pairs' correspondences carry no covariance
    # columns and the bundle's observations are the detections' pixels; PanoramaStitching normalises and substitutes none.
    using Test, StaticArrays, Statistics
    import PanoramaStitching
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitFeatures as VgtkFeatures
    f = described([SVector(600.0, 600.0), SVector(610.0, 590.0)])
    cs, kept = PanoramaStitching.pair_bearings(VgtkCore.BoundedCamera(CAM, VgtkCore.Size2(1200, 1200)), f, f,
                                               [VgtkCore.Attributed(VgtkCore.Correspondence{Int, Int}(i, i), VgtkFeatures.NNMatchMeta(s, 0.5f0)) for (i, s) in ((1, 1.0f0), (2, 0.5f0))];
                                               frame_map = IDENTITY_MAP)
    @test length(cs) == 2 && kept == [(1, 1), (2, 2)] && !VgtkCore.hasattr(cs, :Σ₁) && cs.score == [1.0, 0.5]
    src0 = join(read.(filter(endswith(".jl"), readdir(joinpath(pkgdir(PanoramaStitching), "src"); join = true)), String))
    @test !occursin("Feature" * "Covariance()", src0)                     # the detector at its default
    # the retired consumer-side policy is gone by name: no normalisation, no identity fallback, no whitening switch
    # (the retired names in fragments: check_naming lists them)
    for retired in Symbol.(("Frame" * "Features", "keypoint_covariance" * "_scale", "keypoint" * "_uncertain", "_valid_covariance",
                            "_uncertain_observations", "_check_whitening"))
        @test !isdefined(PanoramaStitching, retired)
    end
    src = join(read.(filter(endswith(".jl"), readdir(joinpath(pkgdir(PanoramaStitching), "src"); join = true)), String))
    for policy in ("bundle_" * "whitening = :", "covariance_scale", "pixel" * "_covariances", "LinearAlgebra.I)", "_valid_covariance")
        @test !occursin(policy, src)
    end
end
