# =============================================================================
# The whitened rotation bundle (src/rotation_bundle.jl, register_rotations' bundle_whitening): observations that carry their
# keypoints' pixel covariances (`frame => Uncertain(pixel, CovMat(Σ))`) are whitened by L⁻¹ (Σ = LLᵀ). On a synthetic fisheye
# sweep with known anisotropic, heteroscedastic keypoint noise the whitened bundle recovers the rotations better than the
# unweighted one; its closed-form Jacobian matches ForwardDiff; the refusals; the covariances over their median, an unrefined keypoint's identity.
# =============================================================================

@testitem "refine_rotations whitened by known anisotropic keypoint covariances beats the unweighted bundle" setup = [SweepScene] begin
    using Test, Random, LinearAlgebra, StaticArrays, Statistics
    import Rotations
    using PanoramaStitching: refine_rotations
    import VisualGeometryToolkitCore as VgtkCore

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
    kw = (; optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(efficiency_constant = 1.0), width = 3.0, max_iter = 1000, ftol = 1.0e-12, xtol = 1.0e-10)
    K = 5
    eu = Float64[]; ew = Float64[]
    for seed in 1:4
        rng = Xoshiro(100 + seed)
        R = sweep_rotations(rng, K)
        tracks, _ = anisotropic_tracks(rng, R, 300)
        start = Dict(k => SMatrix{3, 3}(Rotations.RotMatrix(Rotations.RotationVec((0.01 .* randn(rng, 3))...))) * R[k] for k in 1:K)
        start[1] = R[1]
        err(Rs) = sqrt(mean(k -> rad2deg(Rotations.rotation_angle(Rotations.RotMatrix{3}(SMatrix{3, 3}(Rs[k])' * R[k])))^2, 2:K))
        plain = refine_rotations(CAM, start, [[k => VgtkCore.geom(x) for (k, x) in t] for t in tracks]; kw...)
        white = refine_rotations(CAM, start, tracks; kw...)
        @test plain.metadata.outcome === :converged && white.metadata.outcome === :converged
        push!(eu, err(VgtkCore.estimate(plain))); push!(ew, err(VgtkCore.estimate(white)))
        # the residuals the metadata reports are pixels in both; the whitened loss width is in units of the covariances'
        @test length(white.metadata.residuals) == length(tracks)
        @test 1.0 < white.metadata.loss_width < 6.0                          # 3 × the Rayleigh scale of unit-covariance residuals
    end
    # measured (2026-11-04): the whitened rotations' rms error is a quarter to a half of the unweighted one's on every seed
    @test all(ew .< eu)
    @test mean(ew) < 0.75 * mean(eu)
end

@testitem "refine_rotations whitened: the closed-form Jacobian; the refusals" setup = [SweepScene] begin
    using Test, Random, LinearAlgebra, StaticArrays
    import ForwardDiff, NLLSsolver
    using PanoramaStitching: refine_rotations
    import PanoramaStitching
    import VisualGeometryToolkitCore as VgtkCore

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
    f(δ) = NLLSsolver.computeresidual(obs, NLLSsolver.update(ch, δ[1:3]), NLLSsolver.update(dv, δ[4:5]))
    @test J ≈ ForwardDiff.jacobian(f, zeros(5)) rtol = 1.0e-7

    kw = (; optimization_strategy = VgtkCore.Triggs{VgtkCore.CauchyShape}(efficiency_constant = 1.0), width = 3.0, max_iter = 1000, ftol = 1.0e-12, xtol = 1.0e-10)
    u1 = VgtkCore.project(CAM, SMatrix{3, 3}(R[1])' * d); u2 = VgtkCore.project(CAM, SMatrix{3, 3}(R[2])' * d)
    unc(u, S) = VgtkCore.Uncertain(u, VgtkCore.CovMat(S))
    @test_throws ArgumentError refine_rotations(CAM, R, [[1 => u1, 2 => unc(u2, Σ)]]; kw...)                     # some, not all
    @test_throws ArgumentError refine_rotations(CAM, R, [[1 => unc(u1, Σ), 2 => unc(u2, fill(NaN, SMatrix{2, 2, Float64, 4}))]]; kw...)
    @test_throws ArgumentError refine_rotations(CAM, R, [[1 => unc(u1, Σ), 2 => unc(u2, SMatrix{2, 2, Float64, 4}(1.0, 2.0, 2.0, 1.0))]]; kw...)
end

@testitem "bundle_whitening = :keypoint_covariance: the observations over their median scale, an unrefined keypoint's identity" begin
    using Test, StaticArrays, Statistics
    import PanoramaStitching
    using PanoramaStitching: FrameFeatures
    import VisualGeometryToolkitCore as VgtkCore

    S(a, b, c) = SMatrix{2, 2, Float64, 4}(a, c, c, b)
    nanS = fill(NaN, SMatrix{2, 2, Float64, 4})
    f1 = FrameFeatures([SVector(1.0, 2.0), SVector(3.0, 4.0)], [S(4.0, 1.0, 0.5), nanS], nothing, 2)
    f2 = FrameFeatures([SVector(5.0, 6.0), SVector(7.0, 8.0)], [S(2.0, 2.0, 0.0), S(9.0, 1.0, 0.0)], nothing, 2)
    obs = PanoramaStitching._uncertain_observations([f1, f2], [[(1, 1), (2, 1)], [(1, 2), (2, 2)]])
    @test VgtkCore.geom(last(obs[1][1])) == SVector(1.0, 2.0)
    # every covariance over m, the median of tr Σ / 2 over the refined ones (2.5, 2, 5) = 2.5: the whitened residual in pixels
    # of the median keypoint
    @test SMatrix{2, 2}(Statistics.cov(last(obs[1][1]))) ≈ S(4.0, 1.0, 0.5) / 2.5
    # frame 1's keypoint 2 is unrefined: the identity (the median keypoint's)
    @test SMatrix{2, 2}(Statistics.cov(last(obs[2][1]))) == S(1.0, 1.0, 0.0)
    # the median is over EVERY frame's refined keypoints (the one rule the pairs use, keypoint_covariance_scale), so only a
    # set with no refined keypoint at all is refused
    @test PanoramaStitching.keypoint_covariance_scale([f1]) == 2.5
    f0 = FrameFeatures([SVector(1.0, 2.0), SVector(3.0, 4.0)], [nanS, nanS], nothing, 2)
    @test_throws ArgumentError PanoramaStitching._uncertain_observations([f0], [[(1, 1), (1, 2)]])
end
