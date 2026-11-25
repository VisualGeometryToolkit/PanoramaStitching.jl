# =============================================================================
# vgio/panorama/v2.0 — a panorama's results: the camera, the frames' rotations, the vignetting and the frames' gains
# =============================================================================
#
# {"schema": "vgio/panorama/v2.0", <envelope>,
#  "camera": {bounded camera body},                                  (a camera store's entry without `name`)
#  "rotations": {"from": "camera", "to": "world", "frames": [{"frame_index": 1, "image": "DSC00001.ARW",
#                "pose": {"rotation": [w, x, y, z], "translation": [0.0, 0.0, 0.0]}}, …]},   (the trajectory body)
#  "vignetting": {vignetting body},                                  (optional: the vgio/vignetting body; none stated:
#                                                                     no vignetting field — a candidate with no correction or
#                                                                     one of its own, a flat field's binned radial profile)
#  "gains": [{"frame_index": 1, "gain": [g_R, g_G, g_B]}, …],        (optional: none, every frame at one exposure)
#  "fit": {"retcode": "Success", "stop_rule": "…", "iterations": 12, "rms": 0.41, "n": 31580},
#                                                                    (optional: the bundle's termination record; none:
#                                                                     not recorded)
#  "provenance": {…}}                                                (optional)
#
# The record is this package's and holds only what is its own: which parts a panorama's results are, the frames' gains
# and the rules that tie the parts together. Every part is written and read by the kind that owns it, through that
# owner's functions, and stated in the table by that owner's table:
#
#   camera      VisualGeometryToolkitIO     body_spec(BoundedCamera)   camera_json / camera_from_json
#   rotations   VisualGeometryToolkitIO     body_spec(Trajectory)      JSON.lower(style, ·) / JSON.lift(style, Trajectory, ·)
#   vignetting  VisualRadiometryToolkitIO   VIGNETTING_BODY            JSON.lower(style, ·) / JSON.lift(style, AbstractVignetting, ·)
#   fit         VisualGeometryToolkitIO     body_spec(FitResult)       JSON.lower(style, ·) / JSON.lift(style, FitResult, ·)
#   gains       this record's               (frame_index, gain) per frame, a matrix as its rows (VisualGeometryToolkitIO's)
#
# No body holds a rotation alone, so the rotations are the trajectory body with every translation zero (a camera that
# only rotated: its centre does not move), the rotation exact as the stored quaternion, the zero a length in
# `units.length` like any translation; a file or a value with a translation that is not zero is refused. A frame's gains
# are plain numbers, its exposure gain per colour plane (I = g·L·V), keyed by the frame index `frame_index`, the join
# key of the rotations' frames.
#
# The envelope (VisualGeometryToolkitIO's) declares the units once: the pixel frame of the camera's K and lens coverage
# and of the vignetting's positions (each body moves its numbers by the style), the length unit of the translations and
# of a sensor's pitch. The kind was VisualGeometryToolkitIO's up to its 0.10 and keeps its schema name: a file written
# there loads here, and a vgio/panorama/v1 file (IO 0.9.4) reads through its step to v2.0.
#
# The value has no type of its own: it is the NamedTuple `(; camera, rotations, vignetting, gains, fit)`
# (`vignetting`, `gains`, `fit` each `nothing` when absent), written by `save_panorama` and read by `load_panorama`.

const _PANORAMA_SCHEMA = "vgio/panorama/v2.0"
# Where the published JSON Schema of the kind lives (docs/src/schemas, at the path of the schema string).
const _PANORAMA_SCHEMA_SITE = "https://visualgeometrytoolkit.github.io/PanoramaStitching.jl/schemas/"

const _GAIN_RECORD = VgtkIO.object_spec(:frame_index => VgtkIO.integer_spec(), :gain => VgtkIO.array_spec(VgtkIO.real_spec()))

# The rules that tie the parts together (the whole record's check: `nothing`, or the refusal's message).
function _panorama_check(x)
    r = x[:rotations]
    camera, world = VgtkIO.frame_name(VgtkCore.CameraFrame), VgtkIO.frame_name(VgtkCore.WorldFrame)
    (r[:from] == camera && r[:to] == world) ||
        return "rotations: a panorama's rotations map the camera to the world (`from` $(repr(camera)), `to` $(repr(world))), got $(repr(r[:from])) → $(repr(r[:to]))"
    for f in r[:frames]
        all(iszero, f[:pose][:translation]) ||
            return "rotations: frame $(f[:frame_index]): a panorama's camera only rotates; its translation is zero, got $(collect(f[:pose][:translation]))"
    end
    haskey(x, :gains) || return nothing
    g = x[:gains]
    isempty(g) && return "gains: one record per frame of the rotations (none stated: omit `gains`)"
    Set(q[:frame_index] for q in g) == Set(f[:frame_index] for f in r[:frames]) == Set(1:length(g)) ||
        return "gains: one record per frame of the rotations, whose frame indices are then 1:K (row k of the gains is frame index k)"
    C = length(first(g)[:gain])
    C > 0 && all(q -> length(q[:gain]) == C, g) || return "gains: every frame has the same number ($(C) > 0) of colour planes"
    all(q -> all(>(0), q[:gain]), g) || return "gains: an exposure gain is positive"
    return nothing
end

"""
    PANORAMA_KIND

The `vgio/panorama/v2.0` kind, a `VisualGeometryToolkitIO.KindSpec`: its schema
string and the table of its files — the envelope, then `camera` (the bounded
camera body) and `rotations` (the trajectory body), and where stated
`vignetting` (VisualRadiometryToolkitIO's vignetting body), `gains` (one
`{"frame_index", "gain"}` record per frame) and `fit` (a fit's termination
record). Its check: the rotations map `camera` to `world`, each translation
zero; the gains, where stated, are one record per frame of the rotations (frame
indices `1:K`), each with the same number (> 0) of positive gains.
[`save_panorama`](@ref) writes its files and [`load_panorama`](@ref) reads
them; [`panorama_json_schema`](@ref) is its JSON Schema.
"""
const PANORAMA_KIND = VgtkIO.KindSpec(
    _PANORAMA_SCHEMA,
    [:camera => VgtkIO.body_spec(VgtkCore.BoundedCamera), :rotations => VgtkIO.body_spec(VgtkCore.Trajectory)],
    [:vignetting => VrtkIO.VIGNETTING_BODY, :gains => VgtkIO.array_spec(_GAIN_RECORD), :fit => VgtkIO.body_spec(VgtkCore.FitResult)];
    package = "PanoramaStitching",
    check = _panorama_check,
    rule = "the rotations map `camera` to `world`, each translation zero; `gains`, where stated, one record per frame of the " *
           "rotations (frame indices 1:K), each with the same number (> 0) of positive gains",
    schema_keywords = [Dict("properties" => Dict(
        "rotations" => Dict("properties" => Dict("from" => Dict("const" => VgtkIO.frame_name(VgtkCore.CameraFrame)),
                                                 "to" => Dict("const" => VgtkIO.frame_name(VgtkCore.WorldFrame)))),
        "gains" => Dict("minItems" => 1, "items" => Dict("properties" => Dict("gain" => Dict("minItems" => 1, "items" => Dict("exclusiveMinimum" => 0))))),
    ))],
)

"""
    panorama_json_schema() -> JSON.Object

The JSON Schema (draft 2020-12) of [`PANORAMA_KIND`](@ref), generated from its
table (`VisualGeometryToolkitIO.json_schema`); its `\$id` is
`https://visualgeometrytoolkit.github.io/PanoramaStitching.jl/schemas/vgio/panorama/v2.0.json`.
`JSON.json(panorama_json_schema(); pretty = 4)` is the published file
(docs/src/schemas/vgio/panorama/v2.0.json).
"""
panorama_json_schema() = VgtkIO.json_schema(PANORAMA_KIND; id_base = _PANORAMA_SCHEMA_SITE)

"""
    save_panorama(path, camera::BoundedCamera, rotations::Trajectory, vignetting, gains, fit = nothing;
                  length_unit = mm, provenance = (;))

Write a panorama's results — the camera of a sweep that only rotated, its frames'
rotations, its vignetting and its frames' exposure gains — as a
`vgio/panorama/v2.0` file ([`PANORAMA_KIND`](@ref)), each part through the body
the kind that owns it writes: VisualGeometryToolkitIO's bounded camera body (K,
lens, a sensor's `pitch` in `length_unit`, its lens coverage) and trajectory
body (`rotations`: per frame index its pose `from → to`, its `image` and `time`
where the trajectory has them), and VisualRadiometryToolkitIO's vignetting
body, positions in the canonical (colmap) pixel frame. The rotations map the
camera to the world (`Trajectory{CameraFrame, WorldFrame}`) and every pose's
translation is a zero length: other frames, a translation that is not zero (the
camera only rotates) and a plain one (attach its unit,
`EuclideanMap(R, zeros(3) .* m)`) are an `ArgumentError`, nothing written.

`vignetting` is a VisualRadiometryToolkitCore `AbstractVignetting`, or
`nothing`: a panorama with no vignetting field (a candidate with no correction,
or one of its own — a sampled map, a flat field's binned radial profile), which
writes none and reads back `nothing`.

`fit` is the bundle's `FitResult` (any estimate and metadata: only its
termination record is written — `retcode`, `stop_rule`, `iterations` (its
`stats.nsteps`), `rms`, `n`) or `nothing` (none recorded).

`gains` is `nothing` (every frame at one exposure) or a `K × C` matrix,
`gains[k, c]` the exposure gain of frame index `k` in colour plane `c`
([`VignettingSelfCalibration`](@ref)'s `metadata.gains`), every one positive; the
rotations' frame indices are then `1:K`. Numbers of a narrower type (Float32)
are written exactly. The file's `generator` names PanoramaStitching as the
calling package. [`load_panorama`](@ref) reads the file back.
"""
function save_panorama(path::AbstractString, camera::VgtkCore.BoundedCamera, rotations::VgtkCore.Trajectory,
                       vignetting::Union{Nothing, VrtkCore.AbstractVignetting}, gains::Union{Nothing, AbstractMatrix{<:Real}},
                       fit::Union{Nothing, VgtkCore.FitResult} = nothing; length_unit = VgtkCore.mm, provenance::NamedTuple = (;))
    s = VgtkIO.VgioStyle(; length_unit)
    body = merge((camera = VgtkIO.camera_json(camera, VgtkIO.pixel_frame(s); length_unit), rotations = JSON.lower(s, rotations)),
                 vignetting === nothing ? (;) : (vignetting = JSON.lower(s, vignetting),),
                 _gains_json(gains), fit === nothing ? (;) : (fit = JSON.lower(s, fit),))
    VgtkIO.with_caller(@__MODULE__) do
        VgtkIO.write_kind(path, PANORAMA_KIND, body; style = s, provenance)
    end
    return nothing
end

_gains_json(::Nothing) = (;)
# Row k of the gains is frame index k: the table refuses gains whose frame indices are not the rotations', 1:K.
_gains_json(G::AbstractMatrix{<:Real}) = (gains = [(frame_index = k, gain = row) for (k, row) in enumerate(VgtkIO.matrix_rows(G))],)

"""
    load_panorama(path) -> (; camera, rotations, vignetting, gains, fit)

Read a `vgio/panorama/v2.0` file ([`save_panorama`](@ref)'s): `camera` the
`BoundedCamera` (its K and lens coverage moved into the canonical pixel frame),
`rotations` the `Trajectory` of the frames' poses (`QuatRotation`s, the stored
quaternions exactly; translations zero metres; labeled by the file's `from` and
`to`), `vignetting` the `AbstractVignetting`, rebuilt bit for bit (`nothing` when the
file states none), `fit` the bundle's termination record as a `FitResult{Nothing, Nothing}`
(its estimate and metadata `nothing`; `nothing` when the file records none — every
`vgio/panorama/v1` file), and `gains` the
`K × C` `Matrix{Float64}` of the frames' exposure gains (row `k` frame index `k`),
or `nothing` when the file states none. Every number reads back as the Float64 it
was written as. A file written by VisualGeometryToolkitIO 0.10 (the kind's owner
then) reads the same, and a `vgio/panorama/v1` file through its version step.
"""
function load_panorama(path::AbstractString)
    d = VgtkIO.read_kind(path, PANORAMA_KIND)
    s = VgtkIO.VgioStyle(d)
    return (
        camera = VgtkIO.camera_from_json(VgtkCore.BoundedCamera, d.camera, VgtkIO.pixel_frame(s); length_unit = VgtkIO.length_unit(s)),
        rotations = JSON.lift(s, VgtkCore.Trajectory, d.rotations::VgtkIO.Record),
        vignetting = haskey(d, :vignetting) ? JSON.lift(s, VrtkCore.AbstractVignetting, d.vignetting::VgtkIO.Record) : nothing,
        gains = _gains_from_json(get(d, :gains, nothing)),
        fit = haskey(d, :fit) ? JSON.lift(s, VgtkCore.FitResult, d.fit::VgtkIO.Record) : nothing,
    )
end

_gains_from_json(::Nothing) = nothing
_gains_from_json(g::AbstractVector) = VgtkIO.rows_matrix(r.gain for r in sort(g; by = r -> r.frame_index::Integer))

# vgio/panorama/v1 (VisualGeometryToolkitIO 0.9.4) → v2.0, each part by its owner's step: the envelope (its
# `pixel_frame`, colmap, and pose declarations into `units` and `conventions`), the camera body and the rotations'
# trajectory body (VisualGeometryToolkitIO's), the vignetting body (VisualRadiometryToolkitIO's); the record's own: the
# gains' `t` → `frame_index`. No number moves but a w < 0 quaternion's signs (the envelope step's).
function VgtkIO.migrate_schema(::Val{Symbol("vgio/panorama/v1")}, old::AbstractDict)
    t = VgtkIO.migrate_envelope(old, _PANORAMA_SCHEMA)
    t["camera"] = VgtkIO.migrate_body(VgtkCore.BoundedCamera, old["camera"])
    t["rotations"] = VgtkIO.migrate_body(VgtkCore.Trajectory, t["rotations"])
    t["vignetting"] = VrtkIO.migrate_vignetting_body(old["vignetting"])
    haskey(old, "gains") && (t["gains"] = Any[VgtkIO.renamed_key(g, "t", "frame_index") for g in old["gains"]])
    return t
end

# A listed file of the kind, for `VisualGeometryToolkitIO.validate` and a calibration manifest: the owner of a `vgio`
# kind that moved is keyed by the kind (VisualGeometryToolkitIO's `validate_record`).
VgtkIO.validate_record(::Val{Symbol("vgio/panorama")}, path::AbstractString, schema::AbstractString) = (VgtkIO.read_kind(path, PANORAMA_KIND); nothing)
