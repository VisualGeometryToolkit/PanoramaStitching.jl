# readme_tests.jl — the README's example, the documentation's first example and the Getting started page run as written,
# in a clean environment that holds exactly what the README's install line adds. A reader's first contact with the package
# is that code; the General registration review asked for this test (#174496: the install line missed
# VisualGeometryToolkitCore and StaticArrays; #173031, #174579 for the stack).
#
# The install line is a `pkg>` block (```julia-repl), so it is not run as code: the item checks that it names every
# non-standard-library package the examples load, and nothing else. The examples then run in a child Julia process in a
# temporary environment (this package by path, the rest of the install line from the registries) with none of the test
# run's load path: the README's example first, then the guide's blocks after it, in order, in one module, as a reader
# continuing from the example would. A block that starts with `# Illustrative` (one more package) is counted, not run.

@testitem "README: the install line names what the examples load; the examples run in a fresh environment" begin
    using Test, TOML, PanoramaStitching
    root = pkgdir(PanoramaStitching)
    julia_blocks(text) = [m[1] for m in eachmatch(r"^```julia\n(.*?)^```"ms, text)]
    illustrative(block) = startswith(block, "# Illustrative")
    readme = read(joinpath(root, "README.md"), String)
    index = read(joinpath(root, "docs", "src", "index.md"), String)
    example = julia_blocks(readme)
    landing = julia_blocks(index)
    guide_all = julia_blocks(read(joinpath(root, "docs", "src", "guide.md"), String))
    guide = filter(!illustrative, guide_all)
    @test length(example) == 1 && !illustrative(only(example))   # pin the counts
    @test landing == example                                    # one example, shown twice
    @test length(guide) == 6 && count(illustrative, guide_all) == 2

    # The install line adds exactly the packages the examples load, and no registry line precedes it.
    install = [m[1] for m in eachmatch(r"^pkg> add (.+)$"m, readme)]
    @test length(install) == 1
    packages = split(only(install))
    loaded = Set{String}()
    for code in [example; guide], m in eachmatch(r"^(?:using|import)\s+([^:\n]+)"m, code)
        for part in split(m[1], ',')
            push!(loaded, first(split(strip(part), r"[.\s]")))
        end
    end
    stdlib(name) = isdir(joinpath(Sys.STDLIB, name))
    @test Set(filter(!stdlib, loaded)) == Set(packages)
    @test occursin("pkg> add $(only(install))", index)
    for text in (readme, index)
        @test !occursin(r"^pkg> registry add"m, text)      # the install block is the add line alone
    end

    env = mktempdir()
    write(joinpath(env, "example.jl"), only(example))
    write(joinpath(env, "guide.jl"), join(guide, "\n"))
    registered = filter(!=("PanoramaStitching"), String.(packages))
    # the package's own [sources] paths (a stack not yet registered, held there TEMPORARILY) come along: the fresh environment
    # resolves the stack the package does; with none, everything but the package comes from the registries
    sources = get(TOML.parsefile(joinpath(root, "Project.toml")), "sources", Dict{String, Any}())
    sourced = [normpath(joinpath(root, s["path"])) for s in values(sources) if s isa AbstractDict && haskey(s, "path")]
    registered = filter(∉(keys(sources)), registered)                        # a sourced package is developed, not added
    script = joinpath(env, "run.jl")
    write(script, """
    using Pkg
    Pkg.develop([Pkg.PackageSpec(path = p) for p in $(repr([sourced; root]))]; io = devnull)
    Pkg.add($(repr(registered)); io = devnull)
    readme = Module(:ReadmeSandbox)
    using Test
    Base.include(readme, $(repr(joinpath(env, "example.jl"))))
    @testset "README example" begin
        # the comments' claims: the rotations within 0.05°, the bundle below half a pixel (0.24 px, every kept observation a
        # true inlier of the scene's truth, 2026-11-12), the panorama's size and NaN where
        # no frame sees the pixel
        @test length(readme.reg.rotations) == 4 && maximum(readme.errors) < 0.05
        @test readme.reg.bundle.rms < 0.5
        @test size(readme.pano.image) == (458, 1220, 3)
        @test size(readme.pano.owner) == size(readme.pano.coverage) == size(readme.pano.image)[1:2]
        @test eltype(readme.frames[1]) == Float32
        @test any(>(0), readme.pano.owner) && all(i -> isnan(readme.pano.image[i, 2]), findall(==(0), readme.pano.owner))
    end
    Base.include(readme, $(repr(joinpath(env, "guide.jl"))))
    @testset "Getting started" begin
        @test readme.reg.pairs[(1, 2)].inliers >= 40
        @test readme.S.seen isa AbstractMatrix{Bool} && size(readme.S.seen, 2) == 4 && any(readme.S.seen)
        @test readme.rect.size == (1080, 1920) && readme.planet.size == (2000, 2000)
        # frames without vignetting at one exposure: V and every gain close to 1, the seams small
        @test readme.VgtkCore.converged(readme.res)
        @test readme.g[1] == 1 && all(g -> isapprox(g, 1; atol = 0.01), readme.g)
        @test all(r -> isapprox(readme.V(readme.SA[500.0 + r, 500.0], nothing), 1; atol = 0.01), 0:50:400)
        @test readme.seams.rms < 0.05
        @test size(readme.corrected.image) == size(readme.pano.image) && size(readme.pano.image, 3) == 3
    end
    """)
    # Base.julia_cmd() carries the test run's flags (--depwarn=error, coverage). The load path is the project and the
    # standard library only: no default environment (@v1.12) can supply a package the install line does not add.
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$env $script`
    proc = withenv("JULIA_LOAD_PATH" => "@:@stdlib") do   # --project overrides JULIA_PROJECT
        run(ignorestatus(cmd))
    end
    @test proc.exitcode == 0
end
