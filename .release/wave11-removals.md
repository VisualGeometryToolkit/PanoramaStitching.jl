# Wave 11 removals

The names and keywords PanoramaStitching drops in wave 11 (the stack's next breaking wave), as the monorepo's
`.release/wave11-removals.md` lists its own. Each one still works until then, as its row says. The CHANGELOG points here; add a
row when you deprecate or retire something.

| package | name | deprecated in | replacement / reason | removal |
|---|---|---|---|---|
| PanoramaStitching | `refine_rotations(…; eliminate)`, `refine_camera_and_rotations(…; eliminate)` | 0.8.2 (no deprecation warning: `eliminate = true` is refused by name where the loaded NLLSsolver has no Schur system, `MultiVariateLSschur`, which no installable NLLSsolver has; `false` is the default and the only path) | none: drop the keyword. The bundle solves every unknown together; a Schur elimination comes back, if NLLSsolver ships one, as the bundle's own choice rather than a setting | wave 11 |
