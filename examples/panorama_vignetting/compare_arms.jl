# The comparison of fit_vignetting's settings: every arm of arms.toml (V free or fixed × the gain correction × the observation noise) on
# every set of record, each run keyed by its configs (README.md, "The comparison of the fit's settings").
#
#     julia --project=<env> examples/panorama_vignetting/compare_arms.jl examples/panorama_vignetting/arms.toml
#
# Per set: the decode, the registration and the domain are the set config's (run.jl's stages, cached as there); per arm the fit
# (warm: the second of two runs is timed), then the seam score of every colour plane through the arm's V and gains, pooled over
# the planes too, and V's summary on the domain. Writes <set output>/<arms output>/<key>/results.json and results.md; a key
# that exists is reused (delete its folder to recompute).

include(joinpath(@__DIR__, "run.jl"))


# V's mean in 100 px rings about the lens circle's centre (camera pixels), relative to the first ring: the profile summary.
function ring_profile(v, center, rmax)
    nb = ceil(Int, rmax / 100); acc = zeros(nb); n = zeros(Int, nb)
    for r in 5.0:10.0:rmax, a in range(0, 2π; length = 73)[1:72]
        b = min(floor(Int, r / 100) + 1, nb)
        acc[b] += v(center + r * SVector(cos(a), sin(a)), nothing); n[b] += 1
    end
    p = acc ./ n
    return p ./ p[1]
end

function compare_arms(arms_path)
    A = TOML.parsefile(arms_path)
    refuse_renamed_keys(A, arms_path)                                               # run.jl's: the keys of earlier releases
    foreach(a -> refuse_renamed_keys(a, "$arms_path, arm $(get(a, "name", "?"))"), A["arms"])
    names = [a["name"] for a in A["arms"]]
    allunique(names) || error("arms.toml: arm names repeat")
    A["fixed_from"] in names || error("arms.toml: fixed_from = $(A["fixed_from"]) names no arm")
    ref = only(a for a in A["arms"] if a["name"] == A["fixed_from"])
    ref["vignetting"] == "free" || error("arms.toml: the V of record's arm $(ref["name"]) does not fit V")
    for a in A["arms"]
        a["vignetting"] in ("free", "fixed") || error("arms.toml: arm $(a["name"]): vignetting = $(a["vignetting"])")
        PanoramaStitching.GAIN_CORRECTIONS[a["gain_correction"]]                    # an ArgumentError names the tags
        PanoramaStitching.OBSERVATION_NOISES[a["observation_noise"]]
    end
    # the V of record first: the fixed arms hold its V
    order = vcat(ref, [a for a in A["arms"] if a !== ref])
    for s in A["sets"]
        set_path = joinpath(dirname(abspath(arms_path)), s)
        C = load_config(set_path)
        inp = decoded_frames(C; masks = true)
        Hf, Wf, _ = size(inp.frames[1]); W, H = Int.(PanoramaStitching._covered_extent(inp.frame_map, Wf, Hf))   # the frames' map: (2Wf, 2Hf) for :bin2
        lens = load_lens(C; raw_width = W, raw_height = H)
        reg = registration(C, inp, lens; log = devnull)
        # the run key: Eval's of this comparison's config, the set's config and its registration record folded in by content
        key = VgtkEval.Experiments.run_key(A; files = Dict("set" => set_path, "registration" => reg.record))
        dir = joinpath(C.out, A["output"], key); mkpath(dir)
        isfile(joinpath(dir, "results.json")) && (println("$(C.cfg["name"]): reused $dir"); continue)
        device = get(ENV, "PANORAMA_VIGNETTING_DEVICE", C.cfg["compute"]["device"])
        move = device == "cuda" ? CuArray : identity
        dev = (; frames = move.(inp.frames), saturated = move.(inp.saturated), inp.frame_map, inp.plane_maps)
        up = up_frame(sweep_up(reg.rotations).up)
        sc = C.cfg["seams"]; d = C.cfg["domain"]
        ext = sc["extent"] == "coverage" ? coverage_extent(lens, dev.frames, reg.rotations, up, dev.frame_map) : Tuple(sc["extent"])
        sview = panorama_view(VgtkCore.Equirectangular(), ext, sc["deg_per_pixel"]; up)
        skw = (; saturated = dev.saturated, frame_map = dev.frame_map, rim_margin_px = d["rim_margin_px"], fov_deg = d["fov_deg"],
               plane_maps = dev.plane_maps, chunk = sc["chunk"])
        rim = Float64(d["rim_margin_px"])
        domain = PanoramaStitching.frame_region(lens, W, H, rim)
        in_domain(u) = VgtkCore.Point2(u[1], u[2]) ∈ domain
        pp = SVector{2, Float64}(VgtkCore.principal_point(VgtkCore.model(lens).intrinsics))
        # the profile about the image circle's centre, or about the principal point when the camera has none; out to the domain's edge
        cc = VgtkCore.lens_coverage(lens) isa Union{VgtkCore.Circle, VgtkCore.Ellipse} ? coverage_center(lens) : pp
        rr = domain_radius(lens, cc, W, H, rim)
        Vref = nothing
        rows = Dict{String, Any}[]
        for a in order
            cfg = deepcopy(C.cfg); v = cfg["vignetting"]
            v["gain_correction"] = a["gain_correction"]; v["observation_noise"] = a["observation_noise"]
            v["log_intensity_scale"] = A["log_intensity_scale"]; v["observation_noise_iterations"] = A["observation_noise_iterations"]
            Ca = (; cfg, C.out, run = dir, C.path)
            fkw = (; vignetting = a["vignetting"] == "fixed" ? Vref : nothing)
            fit(A["kind"], Ca, lens, dev, reg.rotations, W, H; fkw...)                       # compile, warm
            t = time(); r = fit(A["kind"], Ca, lens, dev, reg.rotations, W, H; fkw...); dt = time() - t
            V = VgtkCore.estimate(r); m = r.metadata
            a === ref && (Vref = V)
            G = fit_gains(Ca, r)
            seams = [score_seams(lens, dev.frames, reg.rotations, sview; vignetting = V, gains = G, plane = p, skw...) for p in 1:3]
            pooled = vcat([filter(!isnan, vec(Array(s.ratio))) for s in seams]...)
            summ = vignetting_summary(V, pp, inp.frame_map, (Hf, Wf); mask = in_domain)
            row = Dict{String, Any}("arm" => a["name"], "vignetting" => a["vignetting"], "gain_correction" => a["gain_correction"],
                                    "observation_noise" => a["observation_noise"], "seconds" => dt,
                                    "solves" => m.observation_noise isa IntensityAndPositionNoise ? 1 + m.observation_noise.iterations : 1,
                                    "rows" => r.n,
                                    "residual_mad" => m.residual_mad, "weighted_rms" => r.rms,
                                    "gains" => [collect(m.gains[k, :]) for k in axes(m.gains, 1)],
                                    "seam_mad" => [s.mad for s in seams], "seam_rms" => [s.rms for s in seams], "seam_n" => [s.n for s in seams],
                                    "seam_mad_all" => StatsBase.mad(Float64.(pooled); normalize = true), "seam_rms_all" => sqrt(mean(abs2, pooled)),
                                    "V_min" => summ.V_min, "V_max" => summ.V_max, "V_max_offset_px" => summ.max_offset_px,
                                    "V_asymmetry" => summ.asymmetry, "V_rings_100px" => ring_profile(V, cc, rr))
            push!(rows, row)
            println("$(C.cfg["name"]) $(a["name"]): seams 1.4826·MAD R/G/B $(round.(row["seam_mad"]; digits = 4)), all " *
                    "$(round(row["seam_mad_all"]; digits = 4)); $(round(dt; digits = 2)) s"); flush(stdout)
        end
        rec = Dict("set" => C.cfg["name"], "key" => key, "arms_config" => abspath(arms_path), "set_config" => set_path,
                   "fixed_from" => A["fixed_from"], "kind" => A["kind"],
                   "date" => string(Dates.now()), "arms" => rows)
        open(io -> JSON.print(io, rec, 2), joinpath(dir, "results.json"), "w")
        open(joinpath(dir, "results.md"), "w") do io
            println(io, "# fit_vignetting settings: $(C.cfg["name"])\n")
            println(io, "Key $key; the V of record (fixed arms): $(A["fixed_from"]); seams 1.4826·MAD of the log ratio.\n")
            println(io, "| arm | V | gains | observation noise | seam MAD R | G | B | all | seam rms all | V_min | V asym % | gain range (G) | s |")
            println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            f4(x) = @sprintf("%.4f", x)
            for r in rows
                g = reduce(hcat, r["gains"])'                    # K × 3
                println(io, "| $(r["arm"]) | $(r["vignetting"]) | $(r["gain_correction"]) | $(r["observation_noise"]) | " *
                            join(f4.(r["seam_mad"]), " | ") * " | $(f4(r["seam_mad_all"])) | $(f4(r["seam_rms_all"])) | " *
                            "$(@sprintf("%.3f", r["V_min"])) | $(@sprintf("%.1f", 100r["V_asymmetry"])) | " *
                            "$(@sprintf("%.3f", minimum(g[:, 2])))–$(@sprintf("%.3f", maximum(g[:, 2]))) | $(@sprintf("%.2f", r["seconds"])) |")
            end
            println(io, "\nPer-channel gains (g_k R / G / B) of each channel arm:\n")
            for r in rows
                r["gain_correction"] == "channel" || continue
                println(io, "- $(r["arm"]): " * join([join([@sprintf("%.4f", x) for x in gk], " / ") for gk in r["gains"]], ", "))
            end
        end
        println("$(C.cfg["name"]): $(joinpath(dir, "results.md"))")
    end
end

import Dates
import StatsBase
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: julia --project=<env> compare_arms.jl <arms.toml>")
    compare_arms(ARGS[1])
end
