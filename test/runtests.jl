# PanoramaStitching's suite (ReTestItems): every `*_tests.jl` under test/ holds `@testitem`s, each evaluated in its own module on
# a pool of workers. Named by the module (from a path ReTestItems walks up to the unnamed workspace member test/Project.toml).
#   Pkg.test()                                   # every item
#   Pkg.test(; test_args = ["bundle|sample_frames"])   # items whose name matches
#   RETESTITEMS_NWORKERS=0 → serial, in-process
using ReTestItems
using PanoramaStitching

runtests(
    PanoramaStitching;
    nworkers = parse(Int, get(ENV, "RETESTITEMS_NWORKERS", string(clamp(Sys.CPU_THREADS ÷ 4, 1, 4)))),
    nworker_threads = 4, testitem_timeout = 3600,
    name = isempty(ARGS) ? nothing : Regex(join(ARGS, "|")),
)
