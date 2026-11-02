using Documenter, PanoramaStitching

makedocs(;
    sitename = "PanoramaStitching.jl",
    modules = [PanoramaStitching],
    authors = "James Pritts <jbpritts@gmail.com>",
    repo = Documenter.Remotes.GitHub("VisualGeometryToolkit", "PanoramaStitching.jl"),
    format = Documenter.HTML(; canonical = "https://visualgeometrytoolkit.github.io/PanoramaStitching.jl"),
    pages = ["Home" => "index.md", "Getting started" => "guide.md", "API" => "api.md"],
)

# CI deploys to this repository's gh-pages branch (GitHub Pages:
# https://visualgeometrytoolkit.github.io/PanoramaStitching.jl); `dev` from main, a version from each tag TagBot makes.
deploydocs(; repo = "github.com/VisualGeometryToolkit/PanoramaStitching.jl.git", devbranch = "main", push_preview = false)
