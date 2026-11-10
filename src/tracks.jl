# =============================================================================
# The tracks of a pure-rotation sequence
# =============================================================================
#
# The view graph of the verified pairs is VisualGeometryToolkitEstimation's `ViewGraph` (built in registration.jl); the
# tracks are the connected components of the match graph — one vertex per observation (frame, keypoint), one edge per inlier
# match — found by the same `ViewGraph` and its `components`. What is written here is the one rule a track adds: a component
# holding two keypoints of one frame is not a track.
# =============================================================================

"""
    chain_tracks(matches) -> (tracks, n_dropped)

The tracks of a sequence: the connected sets of observations `(frame, keypoint)`
through every pair's inlier matches, `matches` an `AbstractDict` keyed by frame
pairs `(i, j)` whose value is a vector of keypoint index pairs `(a, b)` (keypoint
`a` of frame `i` ↔ keypoint `b` of frame `j`). The connected components of the
match graph, VisualGeometryToolkitEstimation's `ViewGraph` on the observations
(`components`). A track that holds two different keypoints of one frame is
inconsistent (a wrong match chained through the graph) and is DROPPED whole;
their count is `n_dropped`. Every track returned has observations in at least
two frames, sorted by frame then keypoint; the tracks are sorted by their first
observation.
"""
function chain_tracks(matches::AbstractDict{<:Tuple{K, K}}) where {K}
    O = Tuple{K, Int}
    links = Dict{Tuple{O, O}, Nothing}()
    for ((i, j), ms) in matches, (a, b) in ms
        o, p = (i, Int(a)), (j, Int(b))
        links[isless(o, p) ? (o, p) : (p, o)] = nothing                          # a ViewGraph's pairs are canonical
    end
    observations = sort!(unique!(O[o for p in keys(links) for o in p]))
    out = Vector{Vector{O}}(); dropped = 0
    for t in VgtkEstimation.components(VgtkEstimation.ViewGraph(observations, links))   # each sorted by frame, keypoint
        allunique(first.(t)) ? push!(out, t) : (dropped += 1)
    end
    sort!(out; by = first)
    return out, dropped
end
