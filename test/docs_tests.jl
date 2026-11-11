# docs_tests.jl — the docstring gates the Documenter build cannot run (VisualGeometryToolkitCore's `test_docs`, the stack's one
# implementation): every public name has a docstring, attached to its definition, and its signature lines match the methods.
# The General registration review asks for every public name to be documented (#173032). Every exception list is empty: every
# public name of PanoramaStitching has a docstring (the docs review of 2026-11-04).

@testitem "docs: every public name has a docstring, attached, and its signatures match the methods" begin
    using Test, PanoramaStitching
    using VisualGeometryToolkitCore: test_docs
    test_docs(PanoramaStitching)
end
