# The example's checks — its test entry point, run in this environment (each exits 1 on failure):
#
#   julia --project=examples/panorama_vignetting examples/panorama_vignetting/run_checks.jl
#
# stage_cache_check.jl (the verified pairs reused when only the averaging or the bundle changes, recomputed when a pair
# setting changes: Eval's keyed cache) and autocalibration_check.jl (the 14 mm bundle-equivalence check; it needs its declared
# inputs under the data root, and is reported as not run where they are absent). README.md, "Maintainers".
const CHECKS = ("stage_cache_check.jl" => :stage_cache_check, "autocalibration_check.jl" => :autocalibration_check)
failed = String[]
for (file, f) in CHECKS
    m = Module(Symbol(splitext(file)[1], "_sandbox"))
    Core.eval(m, :(include(p) = Base.include($m, p)))
    Base.include(m, joinpath(@__DIR__, file))
    ok = try
        Base.invokelatest(getfield(m, f))
    catch e
        e isa ErrorException && startswith(e.msg, "MISSING INPUT") || rethrow()
        println("$file: NOT RUN, ", first(split(e.msg, '\n')))
        nothing
    end
    ok === nothing || println(file, ": ", ok ? "ok" : "FAILED")
    ok === false && push!(failed, file)
end
isempty(failed) || (println("failed: ", join(failed, ", ")); exit(1))
