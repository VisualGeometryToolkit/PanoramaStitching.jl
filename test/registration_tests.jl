# register_rotations, sample_cells and render_panorama on a synthetic pure-rotation sequence rendered through a known fisheye
# (a Kannala–Brandt lens as a generic grid model imaging past 90°): the rotations, the cells (every frame seeing a cell sees one
# radiance once its gain and vignetting are divided out) and the panorama (the composite is the scene).

@testsetup module SyntheticSequence
using LinearAlgebra: I
using StaticArrays: SVector, SMatrix, SA
using StableRNGs: StableRNG
using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, px, CentralGenericModel, backproject, bilinear_sample, Equirectangular,
    project, exp_so3, angle_between
export sequence

# the scene: random Gaussian blobs on an equirectangular texture; the camera: a KB fisheye (f = 300 px) imaging to 100° off axis;
# K frames rotating 35° about the vertical, 2×2 binned; a vignetting V(θ) = exp(−θ²/2·1.4²) and per-frame gains
function sequence(; K = 4)
    rng = StableRNG(7)
    TW, TH = 2048, 1024
    tex = fill(0.15f0, TH, TW, 3)
    for _ in 1:20000
        λ = 2π * rand(rng) - π; φ = asin(2rand(rng) - 1)
        x = (λ + π) / 2π * TW; y = (π / 2 - φ) / π * TH
        s = 1.0 + 3rand(rng); a = 0.6 * rand(rng); c = 0.3 .+ 0.7 .* rand(rng, 3)
        sx = s / max(cos(φ), 0.05)
        for j in floor(Int, x - 3sx):ceil(Int, x + 3sx), i in max(1, floor(Int, y - 3s)):min(TH, ceil(Int, y + 3s))
            w = a * exp(-((j - x)^2 / sx^2 + (i - y)^2 / s^2) / 2)
            for ch in 1:3
                tex[i, mod1(j, TW), ch] += w * c[ch]
            end
        end
    end
    texture(d) = (p = project(Equirectangular(), d); bilinear_sample(tex, Val(3), SVector((p[1] + π) / 2π, (π / 2 - p[2]) / π)))
    V(θ) = exp(-0.5 * (θ / 1.4)^2)
    W = H = 1000; s = 2; Wf, Hf = W ÷ s, H ÷ s
    model = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))   # the parametric lens the grid resamples
    cam = CentralGenericModel(model, (W, H); grid_size = (40, 40))
    circle = (500.0, 500.0, 480.0)
    axis = backproject(cam, SVector(500.0, 500.0))
    R = [k == 1 ? SMatrix{3, 3}(1.0I) : SMatrix{3, 3}(exp_so3(SVector(0.05randn(rng), deg2rad(35) * (k - 1), 0.05randn(rng)))) for k in 1:K]
    gains = [1.0; exp.(0.1 .* randn(rng, K - 1))]
    frames = map(1:K) do k
        img = zeros(Float32, Hf, Wf, 3)
        for j in 1:Wf, i in 1:Hf
            u = SVector((j - 0.5) * s, (i - 0.5) * s)
            hypot(u[1] - circle[1], u[2] - circle[2]) < circle[3] || continue
            f = backproject(cam, u)
            f === nothing && continue
            img[i, j, :] .= gains[k] * V(angle_between(f, axis)) .* texture(R[k] * f)
        end
        img
    end
    return (; frames, cam, model, circle, raw_scale = s, R, gains, V, texture)
end
end

@testitem "sample_cells: any camera model, not only a generic grid" setup = [SyntheticSequence] begin
    # the sequence's Kannala–Brandt lens as the parametric CameraModel itself (its project has no `init`) and as the 40 × 40 grid
    # resampled from it: the cells through the true rotations agree
    using Test, StableRNGs
    using PanoramaStitching: sample_cells
    q = sequence(; K = 2)
    Hf, Wf = size(q.frames[1], 1), size(q.frames[1], 2)
    cells(cam) = sample_cells(q.frames, [falses(Hf, Wf) for _ in 1:2], q.R, cam; raw_scale = q.raw_scale, circle = q.circle, cell_deg = 4.0,
                              sub = 2, blur_px = 1.0, rim_margin_px = 12.0, fov_deg = 89.0, field_angle = :lens, rng = StableRNG(1))
    p, g = cells(q.model), cells(q.cam)
    both = p.ok .& g.ok
    @test count(p.ok) > 0 && count(both) >= 0.95 * count(g.ok)
    @test maximum(abs.(p.val[both, 2] .- g.val[both, 2])) < 0.02
    @test maximum(abs.(p.theta_model[both] .- g.theta_model[both])) < 0.05     # degrees
end

@testitem "register_rotations, sample_cells, render_panorama: a synthetic fisheye sweep" setup = [SyntheticSequence] begin
    using Test, LinearAlgebra, Statistics, StaticArrays, StableRNGs
    using PanoramaStitching: register_rotations, sample_cells, render_panorama
    import VisualGeometryToolkitCore as VgtkCore
    q = sequence()
    K = length(q.frames)
    reg = register_rotations(q.frames, q.cam; raw_scale = q.raw_scale, circle = q.circle, tone_percentile = 99.5, detector_margin_px = 20.0,
                             asift_tilts = [(1.0, 0.0)], max_detections = 1024, descriptor = :b, matcher = :b128, min_score = 0.0,
                             pairs = :all, max_lag = 1, overlap_deg = 120.0, min_inliers = 40, seed = 1, max_trials = 10000, min_trials = 100,
                             averaging_loss = VgtkCore.HuberShape(), bundle_loss = VgtkCore.CauchyShape(), bundle_scale_multiple = 3.0,
                             bundle_iterations = 3, bundle_max_iter = 100, bundle_ftol = 1.0e-10, bundle_xtol = 1.0e-10)
    errs = [rad2deg(norm(VgtkCore.log_so3(SMatrix{3, 3}(reg.rotations[k])' * q.R[k]))) for k in 1:K]
    @info "synthetic register: rotation errors (deg) $(round.(errs; digits = 4)), bundle rms $(round(reg.bundle.rms; digits = 3)) px"
    @test maximum(errs) < 0.02                                  # 1 px ≈ 0.19° at f = 300 px; thousands of tracks in the bundle
    @test length(reg.pairs) == K * (K - 1) ÷ 2 && all(p -> p.inliers >= 100, values(reg.pairs))
    @test reg.bundle.rms < 0.5 && reg.dropped == 0
    Hf, Wf = size(q.frames[1], 1), size(q.frames[1], 2)
    c = sample_cells(q.frames, [falses(Hf, Wf) for _ in 1:K], reg.rotations, q.cam; raw_scale = q.raw_scale, circle = q.circle, cell_deg = 2.0,
                     sub = 4, blur_px = 1.0, rim_margin_px = 12.0, fov_deg = 89.0, field_angle = :lens, rng = StableRNG(1))
    # every frame seeing a cell sees one radiance once its gain and V are divided out
    spread = Float64[]
    for i in 1:size(c.cells, 1)
        ks = findall(c.ok[:, i]); length(ks) >= 2 || continue
        L = [c.val[k, i, 2] / (q.gains[k] * q.V(deg2rad(c.theta_model[k, i]))) for k in ks]
        push!(spread, maximum(L) / minimum(L) - 1)
    end
    @info "synthetic cells: $(count(c.ok)) samples, median radiance spread across frames $(round(median(spread); digits = 4))"
    @test median(spread) < 0.03
    # the panorama of the corrected values: each seen cell is its last frame's value over that frame's gain
    vals = [c.ok[k, i] ? c.val[k, i, ch] / q.V(deg2rad(c.theta_model[k, i])) : NaN for k in 1:K, i in 1:size(c.cells, 1), ch in 1:3]
    img, ext, comp = render_panorama(c.cells, vals, c.ok, log.(repeat(q.gains, 1, 3)))
    seen = findall(vec(any(c.ok; dims = 1)))
    @test all(i -> comp[i, 2] ≈ vals[findlast(c.ok[:, i]), i, 2] / q.gains[findlast(c.ok[:, i])], seen)
    @test all(i -> isnan(comp[i, 2]), findall(.!vec(any(c.ok; dims = 1))))
    @test size(img, 3) == 3 && count(isfinite, img[:, :, 2]) > 0.5 * length(seen)
    @test ext[2] > ext[1] && ext[4] > ext[3]
end
