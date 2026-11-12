# compare_makie.jl — view a run's outputs uncorrected vs corrected in GLMakie, the two axes linked (pan and zoom together):
# VisualGeometryToolkitFeatures' `compare_images` (its Makie extension: the menu, the linked axes, Core's image orientation).
#
#   julia --project=examples/panorama_vignetting examples/panorama_vignetting/compare_makie.jl $VGTK_RESULTS_ROOT/panorama_vignetting/fx3_14mm_indoor/run-<key>
#
# or from a REPL: include("compare_makie.jl"); compare_makie("<results>/panorama_vignetting/fx3_14mm_indoor/run-<key>"). The menu switches the pair shown: each
# projection's renders and the seam-ratio maps (run.jl's `<view>_uncorrected.png` and `<view>_corrected.png`). The renders of one
# pair are the same size and the same exposure, so the linked axes compare pixel for pixel. `save_comparison` writes the
# figure as a PNG (for a reader without a display). What is the example's own: which renders form a pair.
using GLMakie
using VisualGeometryToolkitFeatures: compare_images

const VIEWS = ("cylindrical", "equal_area", "rectilinear", "seams")

# The views of `dir` that run.jl wrote, both halves present.
available_views(dir) = [v for v in VIEWS if all(isfile(joinpath(dir, "$(v)_$(s).png")) for s in ("uncorrected", "corrected"))]

function compare_figure(dir::AbstractString; view = first(available_views(dir)), size = (1800, 1000))
    views = available_views(dir)
    isempty(views) && throw(ArgumentError("no <view>_uncorrected.png / <view>_corrected.png pair in $dir"))
    view in views || throw(ArgumentError("view $view not in $dir (has $(join(views, ", ")))"))
    load(v, s) = GLMakie.FileIO.load(joinpath(dir, "$(v)_$(s).png"))
    entries = [v => () -> (load(v, "uncorrected"), load(v, "corrected")) for v in views]   # each pair loaded when chosen
    fig = compare_images(entries; titles = ("uncorrected", "corrected (÷ V)"), default = view, size)
    Label(fig[0, 1], basename(normpath(dir)); fontsize = 20, tellwidth = false)
    return fig
end

compare_makie(dir::AbstractString; kwargs...) = (fig = compare_figure(dir; kwargs...); display(fig); fig)

save_comparison(path, dir::AbstractString; kwargs...) = save(path, compare_figure(dir; kwargs...))

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: compare_makie.jl <output folder of run.jl>")
    screen = display(compare_figure(ARGS[1]))
    wait(screen)
end
