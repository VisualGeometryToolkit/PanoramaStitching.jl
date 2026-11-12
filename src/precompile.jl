# The precompile workload (PrecompileTools): tiny but real instances of the package's entry points on the host, run once while the
# package image is built, so that their native code is cached in it and a fresh process does not compile it again. CPU only, no
# data files, no output: two 40 × 40 frames of a smooth synthetic scene through a Kannala–Brandt fisheye, rotated in yaw.

@setup_workload begin
    camera = VgtkCore.BoundedCamera(VgtkCore.CameraModel(40.0 * VgtkCore.px, SVector(40.0 * VgtkCore.px, 40.0 * VgtkCore.px);
                                                         lens = VgtkCore.KannalaBrandt(-0.02)),
                                    VgtkCore.Size2(80, 80); lens_coverage = VgtkCore.Circle(VgtkCore.Point2(40.0, 40.0), 38.0))
    rotations = [SMatrix{3, 3}(VgtkCore.exp_so3(SVector(0.0, deg2rad(y), 0.0))) for y in (0.0, 30.0)]
    # the frames' map onto the camera's pixels: a Bayer cell grid's (Core's raw_frame_map, x ↦ 2x), from a synthetic 2 × 2 mosaic
    mosaic = VgtkCore.RawMosaic(zeros(UInt16, 2, 2), UInt8[0 1; 3 2], (0.0, 0.0, 0.0, 0.0), 1.0, (0, 0), (2, 2), (; model = "synthetic"))
    frame_map = VgtkCore.raw_frame_map(VgtkCore.LinearRaw(mosaic; mode = :bin2))
    scene(d) = 0.05 * exp(0.6 * sin(3d[1] + 2d[2]) * cos(2d[3]))
    frames = map(rotations) do R
        img = zeros(Float32, 40, 40, 3)
        for j in 1:40, i in 1:40
            u = frame_map(SVector(j - 0.5, i - 0.5))                    # the frame pixel's centre in the camera's pixels
            hypot(u[1] - 40.0, u[2] - 40.0) < 38.0 || continue
            img[i, j, :] .= Float32(scene(R * VgtkCore.backproject(camera, u)) * exp(-0.3 * (hypot(u[1] - 40.0, u[2] - 40.0) / 40)^2))
        end
        img
    end
    saturated = [falses(40, 40) for _ in rotations]
    @compile_workload begin
        kw = (; frame_map, rim_margin_px = 2.0, fov_deg = 100.0)
        fit = (; saturated, kw..., plane = 2, chart_spacing_deg = 4.0, penalty_order = 3, penalty_weight = 0.01,
               vignetting_penalty_weight = 1.0e-6, solver = SparseCholesky())
        # the radial and the tensor log-vignetting, each from log V ≡ 0
        radial = VgtkCore.LogVignetting(VgtkCore.with_control_points(VgtkCore.RadialVignetting(6, SVector(40.0, 40.0), 42.0), zeros(6)))
        result = fit_vignetting(radial, camera, frames, rotations; free_directions = VgtkCore.pole_regular_basis(radial), gain_correction = ChannelGains(), fit...)
        tensor = VgtkCore.LogVignetting(VgtkCore.with_control_points(VgtkCore.TensorProductVignetting((4, 4), VgtkCore.Rect(2.0, 2.0, 74.0, 74.0)), zeros(16)))
        fit_vignetting(tensor, camera, frames, rotations; free_directions = Matrix(1.0I, 16, 16), gain_correction = IntensityGains(), fit...)
        v = VgtkCore.estimate(result)
        view = panorama_view(VgtkCore.Equirectangular(), (-60.0, 90.0, -40.0, 40.0), 5.0)
        render_panorama(camera, frames, rotations, view; vignetting = v, kw...)
        render_cube_map(camera, frames, rotations, 8; vignetting = v, kw...)
        score_seams(camera, frames, rotations, view; vignetting = v, saturated, kw..., plane = 2, chunk = 1000)
        sample_frames(camera, frames, rotations, fibonacci_directions(rotations, 50; fov_deg = 100.0); saturated, vignetting = v, kw...)
        # (not refine_rotations: an NLLSsolver solve evaluates into NLLSsolver, which a precompiling package may not do)
    end
end
