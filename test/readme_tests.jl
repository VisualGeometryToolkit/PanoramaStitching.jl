# The README's code runs and its installation is complete (the General registration review of the stack, 2026-10, found
# README examples that did not run and an installation that missed a package the examples name). The installation is fenced
# ```julia-repl (a `pkg>` line, not run: it would modify the test environment); every ```julia block runs, in order, in one
# fresh module, as a reader pasting them into a REPL would. The usage's one placeholder, `my_frames` (the reader's images), is
# the synthetic sequence the README describes: a Kannala–Brandt fisheye of 300 px on a 1000 × 1000 sensor, frames 2 × 2 binned.

@testitem "README: the installation names every package the usage loads; the usage runs" setup = [SyntheticSequence] begin
    using Test, LinearAlgebra, StaticArrays
    using PanoramaStitching
    import VisualGeometryToolkitCore as VgtkCore
    readme = read(joinpath(pkgdir(PanoramaStitching), "README.md"), String)
    install = [m[1] for m in eachmatch(r"^```julia-repl\n(.*?)^```"ms, readme)]
    blocks = [m[1] for m in eachmatch(r"^```julia\n(.*?)^```"ms, readme)]
    @test length(install) == 1 && !isempty(blocks)
    # the registry that holds the package, and every non-stdlib package a block loads
    @test occursin(r"^pkg> registry add .*CauRegistry\.jl"m, install[1])
    loaded = unique(m[1] for b in blocks for m in eachmatch(r"^(?:using|import) (\w+)"m, b))
    @test !isempty(loaded)
    for name in loaded
        @test occursin(Regex("^pkg> add .*\\b$name\\b", "m"), install[1])
    end

    q = sequence()
    sandbox = Module(:ReadmeExamples)
    Core.eval(sandbox, :(my_frames = $(q.frames)))
    for (i, code) in enumerate(blocks)
        @test (include_string(sandbox, code, "README.md (julia block $i)"); true)
    end
    # the usage's commented claims hold
    reg = Core.eval(sandbox, :reg); pano = Core.eval(sandbox, :pano); res = Core.eval(sandbox, :res)
    K = length(q.frames)
    errs = [rad2deg(norm(VgtkCore.log_so3(SMatrix{3, 3}(reg.rotations[k])' * q.R[k]))) for k in 1:K]
    @test length(reg.rotations) == K && maximum(errs) < 0.05 && reg.bundle.rms < 1.0         # pixels
    @test VgtkCore.converged(res) && length(res.metadata.gains) == K && res.metadata.gains[1] == 1
    @test ndims(pano.image) == 3 && size(pano.image, 3) == 3 && size(pano.owner) == size(pano.coverage) == size(pano.image)[1:2]
    @test any(>(0), pano.owner) && all(i -> isnan(pano.image[i, 2]), findall(==(0), pano.owner))
end
