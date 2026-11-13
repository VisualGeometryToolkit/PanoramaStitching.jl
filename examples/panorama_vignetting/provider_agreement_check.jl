# LoMa's two ONNX Runtime providers agree on the registration — a CHECK (exit 1 on failure), run in this environment on a GPU
# (it needs CUDA) by run_checks.jl, the CPU half in the same run:
#
#   julia --project=examples/panorama_vignetting examples/panorama_vignetting/provider_agreement_check.jl
#
# The 14 mm config of record (11 frames) registered twice, LoMa on the CPU and on CUDA, each from scratch (its own results
# folder: the provider keys the pairs and the registration): the pairs the view-graph filters KEEP are the same set, and the
# solved rotations agree within ROTATION_TOL_DEG. A kept pair that flips between the providers is printed by name.
include(joinpath(@__DIR__, "run.jl"))
using Test

# The tolerance (degrees). The bundle's rotations are fixed by the frames' tracks: a frame's rotation has a standard deviation of
# about σ̂ / (f √n) rad — σ̂ ≈ 0.5 px (the pooled RANSAC scale), f ≈ 2360 px (14 mm at the FX3's 5.93 µm pixels), n ≥ 300 kept
# observations per frame — ≈ 1.2e-5 rad = 7e-4°. Matches that differ between the providers at the network's threshold (a few per
# pair: CUDA's and the CPU's floating-point order) move the solution by a few such deviations; 0.01°, ~14 deviations and 0.4 px at
# the frame's edge, bounds that and fails a pair that flips (a weak pair's rotation differs by degrees between the providers
# without the track check: 9.8° on frames 4–7, 2026-11-13).
const ROTATION_TOL_DEG = 0.01

function provider_agreement_check(config = joinpath(@__DIR__, "config_fx3_14mm.toml"))
    CUDA.functional() || error("MISSING INPUT [a CUDA device]: the check runs LoMa on CUDA and on the CPU")
    C0 = load_config(config)
    raw = input_path(C0.cfg["inputs"]["raw_dir"])
    isdir(raw) || error("MISSING INPUT [raw_dir]: $raw")
    inp = decoded_frames(C0; masks = true)
    Hf, Wf, _ = size(inp.frames[1]); W, H = Int.(PanoramaStitching._covered_extent(inp.frame_map, Wf, Hf))
    lens = load_lens(C0; raw_width = W, raw_height = H)
    function registered(provider)
        cfg = deepcopy(C0.cfg); cfg["registration"]["execution_provider"] = provider
        return registration((; C0..., cfg, out = mktempdir()), inp, lens; log = stdout)
    end
    cpu, cuda = registered("cpu"), registered("cuda")
    kc, kg = Set(cpu.pairs_kept), Set(cuda.pairs_kept)
    flips = sort!(collect(symdiff(kc, kg)))
    for p in flips
        println("kept pair $(p[1])-$(p[2]) flips: kept on $(p in kc ? "the CPU" : "CUDA") only")
    end
    angle(A, B) = rad2deg(norm(VgtkCore.log_so3(A' * B)))             # the relative rotation's angle (Core's log map)
    dR = [angle(cpu.rotations[k], cuda.rotations[k]) for k in eachindex(cpu.rotations)]
    println("rotations, CPU vs CUDA: max $(round(maximum(dR); sigdigits = 3))°, per frame ", join(round.(dR; sigdigits = 2), ", "),
            "; kept pairs $(length(kc)) / $(length(kg)), track check removed $(sum(h -> h["removed"], cpu.track_check; init = 0)) / " *
            "$(sum(h -> h["removed"], cuda.track_check; init = 0)) observations")
    ok = try
        @testset "LoMa's providers agree on the registration" begin
            @test isempty(flips)
            @test maximum(dR) <= ROTATION_TOL_DEG
        end
        true
    catch e
        e isa Test.TestSetException || rethrow()
        false
    end
    return ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    provider_agreement_check() || exit(1)
end
