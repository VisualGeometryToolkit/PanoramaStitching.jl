# compare_makie.jl — view a run's outputs uncorrected vs corrected in GLMakie, the two axes linked (pan and zoom together).
#
#   julia --project=examples/panorama_vignetting examples/panorama_vignetting/compare_makie.jl examples/panorama_vignetting/out/fx3_14mm_indoor
#
# or from a REPL: include("compare_makie.jl"); compare_makie("out/fx3_14mm_indoor"). The menu switches the pair shown: each
# projection's renders and the seam-ratio maps (run.jl's `<view>_uncorrected.png` and `<view>_corrected.png`). The renders of one
# pair are the same size and the same exposure, so the linked axes compare pixel for pixel. `save_comparison` writes the
# figure as a PNG (for a reader without a display).
using GLMakie

const VIEWS = ("cylindrical", "equal_area", "rectilinear", "seams")

# A PNG (Makie's FileIO `load`) in Makie's image orientation: x right, y up, the file's top row at the top.
_load(path) = rotr90(GLMakie.FileIO.load(path))

# The views of `dir` that run.jl wrote, both halves present.
available_views(dir) = [v for v in VIEWS if all(isfile(joinpath(dir, "$(v)_$(s).png")) for s in ("uncorrected", "corrected"))]

function compare_figure(dir::AbstractString; view = first(available_views(dir)), size = (1800, 1000))
    views = available_views(dir)
    isempty(views) && throw(ArgumentError("no <view>_uncorrected.png / <view>_corrected.png pair in $dir"))
    view in views || throw(ArgumentError("view $view not in $dir (has $(join(views, ", ")))"))
    pair(v) = (_load(joinpath(dir, "$(v)_uncorrected.png")), _load(joinpath(dir, "$(v)_corrected.png")))
    fig = Figure(; size)
    menu = Menu(fig[1, 1]; options = views, default = view, tellwidth = false)
    imgs = Observable(pair(view))
    axes = [Axis(fig[i + 1, 1]; title = t, aspect = DataAspect()) for (i, t) in enumerate(("uncorrected", "corrected (÷ V)"))]
    for (i, ax) in enumerate(axes)
        hidedecorations!(ax)
        image!(ax, lift(p -> p[i], imgs); interpolate = false)
    end
    linkaxes!(axes...)
    Label(fig[0, 1], basename(normpath(dir)); fontsize = 20, tellwidth = false)
    on(menu.selection) do v
        imgs[] = pair(v)
        reset_limits!(axes[1])
    end
    return fig
end

compare_makie(dir::AbstractString; kwargs...) = (fig = compare_figure(dir; kwargs...); display(fig); fig)

save_comparison(path, dir::AbstractString; kwargs...) = save(path, compare_figure(dir; kwargs...))

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: compare_makie.jl <output folder of run.jl>")
    screen = display(compare_figure(ARGS[1]))
    wait(screen)
end
