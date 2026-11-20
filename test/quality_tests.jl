# The stack's quality gates (VisualGeometryToolkitCore's `test_quality_gates`: tags, the dependency allow-list, public names
# across package boundaries, the contract generics, Aqua, ExplicitImports, JET) and the agent-docs policy.

@testitem "quality gates" begin
    using Test, Aqua, ExplicitImports, JET
    import VisualGeometryToolkitCore as VgtkCore
    using PanoramaStitching
    import VisualRadiometryToolkitCore as VrtkCore
    using TOML: TOML
    # Atomix, Adapt, ForwardDiff and JSON declare no public names: `Atomix.@atomic` (the vignetting fit's scatter-add),
    # `Adapt.@adapt_structure` (the kernels' functors moved to a device), `ForwardDiff.gradient` (the
    # PositionNoise weights' observation gradient), `ForwardDiff.jacobian` (the autocalibration bundle's camera
    # block) and `JSON.lower`/`JSON.lift` (the panorama record's bodies on VisualGeometryToolkitIO's style) are their
    # documented API
    #
    # The allow-list: this package's row (configs/stack_dependencies.toml) with the Visual Radiometry Toolkit's own
    # rows, which make its packages stack packages (VisualGeometryToolkitCore's table names none of them).
    rows(dir) = TOML.parsefile(joinpath(dir, "configs", "stack_dependencies.toml"))
    VgtkCore.test_quality_gates(PanoramaStitching; jet_bound = 0,
                                qualified_ignore = (Symbol("@atomic"), Symbol("@adapt_structure"), :gradient, :jacobian, :lower, :lift),
                                stack_extra = merge(rows(pkgdir(VrtkCore)), rows(pkgdir(PanoramaStitching))))
end

@testitem "agent docs and the conventions pointer" begin
    using Test
    using VisualGeometryToolkitCore: agent_doc_violations
    using PanoramaStitching
    root = pkgdir(PanoramaStitching)
    @test isempty(agent_doc_violations(root; extra = get(ENV, "AGENT_DOC_PATTERNS", nothing)))
    claude = read(joinpath(root, "CLAUDE.md"), String)
    @test occursin(r"^#+ Stack Conventions$"m, claude)
    @test occursin("VisualGeometryToolkitCore/conventions/", claude)
    @test occursin("VisualGeometryToolkitCore.conventions_dir()", claude)
    for copied in ("Frame in = frame out.", "Iterative-estimator vocabulary", "Correspondences in, dicts for groups")
        @test !occursin(copied, claude)
    end
end

@testitem "naming" begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    using PanoramaStitching
    found = VgtkCore.check_naming(pkgdir(PanoramaStitching))
    foreach(v -> @info(sprint(show, v)), found)
    @test isempty(found)
end

@testitem "duplication: one implementation per concept" begin
    # VisualGeometryToolkitCore's `check_duplication` over the package (names an upstream stack package owns, restated function
    # bodies, copied files, retired patterns), against the stack repositories in ENV["VGTK_STACK_PATHS"] (absent ones skipped)
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    using PanoramaStitching
    VgtkCore.test_duplication(PanoramaStitching)
end
