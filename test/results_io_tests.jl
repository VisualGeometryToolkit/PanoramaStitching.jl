# PanoramaStitching's results through VisualGeometryToolkitIO's vgio/panorama/v2.0: the camera, the frames' rotations, the fitted
# V and the frames' gains saved and loaded, and the loaded values render and score the panorama as the fit's own do.

@testitem "results: saved and loaded through vgio/panorama/v2.0, they render and score as the fit's" setup = [VignettingSequence, BayerMaps] begin
    using Test, StaticArrays
    import JSON
    using PanoramaStitching: PanoramaStitching, fit_vignetting, score_seams, render_panorama, panorama_view, SparseCholesky, ChannelGains
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: RadialVignetting, LogVignetting, pole_regular_basis, with_control_points, Trajectory, CameraFrame,
        WorldFrame, EuclideanMap, control_points, m
    using Rotations: RotMatrix
    c = SVector(430.0, 390.0)
    falloff(u) = -sum(abs2, u - c) / 400^2                                # log V: a quadratic fall-off about c
    G = [1.0 1.0 1.0; 1.10 1.12 1.15; 0.93 0.90 0.86; 1.25 1.20 1.12; 0.97 0.95 0.99; 1.02 1.07 1.11]
    q = vignetting_sequence(falloff; gains = G)
    v0 = LogVignetting(with_control_points(RadialVignetting(12, c, 420.0), zeros(12)))
    r = fit_vignetting(v0, q.camera, q.frames, q.R; free_directions = pole_regular_basis(v0), gain_correction = ChannelGains(),
                       saturated = q.saturated, frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0, plane = 2,
                       chart_spacing_deg = 1.0, penalty_order = 3, penalty_weight = 0.01, vignetting_penalty_weight = 1.0e-6,
                       solver = SparseCholesky())
    # the rotations as the panorama's trajectory: camera → world (d = R_k·f_k), zero translations, each frame's image
    tr = Trajectory(CameraFrame, WorldFrame, Dict(k => EuclideanMap(RotMatrix(R), zeros(3) .* m) for (k, R) in enumerate(q.R));
                    images = Dict(k => "frame_$(k).ARW" for k in eachindex(q.R)))
    mktempdir() do dir
        p = joinpath(dir, "panorama.json")
        VgtkIO.with_caller(PanoramaStitching) do                       # the generator names the calling package (the example's)
            VgtkIO.save_panorama(p, q.camera, tr, VgtkCore.estimate(r), r.metadata.gains)
        end
        rec = JSON.parsefile(p)
        @test rec["schema"] == "vgio/panorama/v2.0" && rec["generator"]["caller"]["name"] == "PanoramaStitching"
        b = VgtkIO.load_panorama(p)
        @test VgtkIO.camera_json(b.camera) == VgtkIO.camera_json(q.camera) && VgtkCore.lens_coverage(b.camera) == VgtkCore.lens_coverage(q.camera)
        @test typeof(b.vignetting) == typeof(VgtkCore.estimate(r)) && control_points(b.vignetting) == control_points(VgtkCore.estimate(r))
        @test b.gains == r.metadata.gains
        R = [SMatrix{3, 3, Float64, 9}(RotMatrix(b.rotations[k].R)) for k in eachindex(q.R)]
        @test all(maximum(abs, R[k] - q.R[k]) < 1.0e-15 for k in eachindex(q.R))       # a matrix written as its quaternion
        @test b.rotations.images == tr.images
        # the loaded results render and score the panorama as the fit's own values do
        view = panorama_view(VgtkCore.Equirectangular(), (-80.0, 140.0, -70.0, 70.0), 0.5)
        rk = (; frame_map = FRAME_MAP, rim_margin_px = 8.0, fov_deg = 120.0)
        a = render_panorama(q.camera, q.frames, q.R, view; vignetting = VgtkCore.estimate(r), gains = r.metadata.gains, rk...)
        Rq = [b.rotations[k].R for k in 1:length(b.rotations)]                          # the loaded QuatRotations, as the guide hands them over
        l = render_panorama(b.camera, q.frames, Rq, view; vignetting = b.vignetting, gains = b.gains, rk...)
        @test a.owner == l.owner && isapprox(a.image, l.image; nans = true, rtol = 1.0e-12)
        sk = (; saturated = q.saturated, chunk = 20_000, plane = 2, rk...)
        sa = score_seams(q.camera, q.frames, q.R, view; vignetting = VgtkCore.estimate(r), gains = r.metadata.gains, sk...)
        sl = score_seams(b.camera, q.frames, Rq, view; vignetting = b.vignetting, gains = b.gains, sk...)
        @test sa.n == sl.n && isapprox(sa.rms, sl.rms; rtol = 1.0e-10)
        # no gains fitted: the file states none, and reads back `nothing`
        VgtkIO.save_panorama(p, q.camera, tr, VgtkCore.estimate(r), nothing)
        @test VgtkIO.load_panorama(p).gains === nothing
    end
end
