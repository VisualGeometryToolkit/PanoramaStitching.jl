# PanoramaStitching's suite (ReTestItems): every `*_tests.jl` under test/ holds `@testitem`s, each evaluated in its own module on
# a pool of workers. Named by the module (from a path ReTestItems walks up to the unnamed workspace member test/Project.toml).
#   Pkg.test()                                   # every item
#   Pkg.test(; test_args = ["bundle|sample_frames"])   # items whose name matches
#   RETESTITEMS_NWORKERS=0 → serial, in-process
#   PS_TEST_RSS_LOG=<file> → each item's worker resident memory (GiB, after the release below) appended to <file>
using ReTestItems
using PanoramaStitching

# After each item, what it held is released. ReTestItems evaluates an item into a module bound in Main (anonymous modules are
# not reliably collected, julia#48711), so an item's globals — its scenes, frames and LoMa's ONNX Runtime sessions, ~1.6 GiB
# each — would live for the rest of the run: on a 4-core CI runner the one worker grew to its 16 GB (PanoramaStitching
# 0.8.1, GitHub run 43572189718; reproduced on dgx01 in a 16 GiB cgroup, the worker killed at 15.2 GiB). The finished
# item's own non-constant globals are set to `nothing` (no item reads another's: each is its own module; the setups'
# modules are untouched), the GC runs, and the C heap is handed back to the OS (glibc's malloc_trim: ONNX Runtime frees its
# arena there, and a freed heap otherwise stays resident).
const RELEASE_ITEM = quote
    let me = string(nameof(@__MODULE__)), item = match(r"^##(.*) test_end#\d+$", me)
        if item !== nothing
            prefix = "##" * item.captures[1] * "#"
            for n in names(Main; all = true)
                (startswith(string(n), prefix) && isdefined(Main, n)) || continue
                m = getfield(Main, n)
                m isa Module || continue
                for g in names(m; all = true)
                    (isdefined(m, g) && !isconst(m, g)) && Core.eval(m, :($g = nothing))
                end
            end
        end
    end
    GC.gc(true)
    Sys.islinux() && try
        ccall(:malloc_trim, Cint, (Csize_t,), 0)
    catch
    end
    if haskey(ENV, "PS_TEST_RSS_LOG")
        rss = parse(Int, match(r"VmRSS:\s+(\d+) kB", read("/proc/self/status", String)).captures[1]) / 2^20
        open(io -> println(io, round(rss; digits = 2)), ENV["PS_TEST_RSS_LOG"], "a")
    end
end

runtests(
    PanoramaStitching;
    nworkers = parse(Int, get(ENV, "RETESTITEMS_NWORKERS", string(clamp(Sys.CPU_THREADS ÷ 4, 1, 4)))),
    nworker_threads = 4, testitem_timeout = 3600, test_end_expr = RELEASE_ITEM,
    name = isempty(ARGS) ? nothing : Regex(join(ARGS, "|")),
)
