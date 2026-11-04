using Documenter, PanoramaStitching

makedocs(;
    sitename = "PanoramaStitching.jl",
    modules = [PanoramaStitching],
    authors = "James Pritts <jbpritts@gmail.com>",
    repo = Documenter.Remotes.GitHub("VisualGeometryToolkit", "PanoramaStitching.jl"),
    format = Documenter.HTML(; canonical = "https://visualgeometrytoolkit.github.io/PanoramaStitching.jl"),
    # The docs gates (General review, #171997, #173031, #173032): every public docstring is in the manual, the doctests
    # run, every external link resolves, and nothing is warnonly: a warning fails the build. That every public name HAS a
    # docstring is test/docs_tests.jl's job (the missing-docs check only sees docstrings that exist).
    checkdocs = :public,
    doctest = true,
    linkcheck = true,
    # github.com rate-limits a crawler (429 on the stack's builds, 2026-11-04); its links are read by hand before a
    # registration (the General checklist's R8).
    linkcheck_ignore = [r"^https://github\.com/"],
    pages = ["Home" => "index.md", "Getting started" => "guide.md", "API" => "api.md"],
)

# CI deploys to this repository's gh-pages branch (GitHub Pages:
# https://visualgeometrytoolkit.github.io/PanoramaStitching.jl); `dev` from main, a version from each tag TagBot makes.
deploydocs(; repo = "github.com/VisualGeometryToolkit/PanoramaStitching.jl.git", devbranch = "main", push_preview = false)
