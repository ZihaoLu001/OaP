# Objective generation and simulation

Run the commands below from the source checkout after installing the package.
Scene data, generated programs, model responses and execution outputs belong
in a separate workspace. The pipeline rejects workspaces and output directories
inside the checkout. These entry points run offline simulation, not a robot.

## Configure external inputs

```bash
python -B -m oap_pipeline.generation --root ../oap-work init
```

This creates only `../oap-work/config.json` from
[`config.example.json`](config.example.json). It does not create scene anchors,
meshes, observations or objective programs.

Set `input_root` to your external data directory. Relative values are resolved
against the workspace. Every task's `context_path` and `inputs` paths are
resolved against `input_root`; absolute paths are also accepted. The executor
passes the resolved directory as `OAP_INPUT_ROOT`.

Supply:

- A scene bundle manifest with schema `real2sim2real_scene_manifest_v1`,
  its reconstructed object assets and referenced calibration files.
- A calibrated MuJoCo robot/scene XML, initial observation and scene RGB image.
- A context JSON for each selected task, with `instruction`, `body_names`
  and `anchors`. The instruction must match `config.json`.
- For the Cup profile, a fixture JSON with schema `oap_lab_cup_fixture_v1`
  and the visual mesh and texture referenced by that fixture. The configuration's
  `OAP_LAB_CUP_FIXTURE` is relative to `input_root`.

The default Pick, Push and Tool-push profiles also validate their external
scene identities. Under `input_root`, provide:

- `authorities/joint_velocity_force_scene_manifest.json`: the complete
  declared scene manifest, including `camera_calibration`. The loaded bundle
  is compared against this object.
- `authorities/pick_input_identity.json`: `camera_sha256`,
  `eraser_refined_sha256` and `box_refined_sha256` describing the supplied
  calibration and refined object poses.

The separate Pick-v9 S0 diagnostic additionally requires
`authorities/pick_v9_s0_stage.json` containing its declared stage object,
and `initial_pos_base`, `initial_size_lwh_m` and
`initial_yaw_base_rad` in the identity file. That diagnostic stage is not
needed by the default generated-program pipeline. The Cup profile instead
validates its explicit fixture, initial observation and referenced assets.
None of these data or authority files is bundled with the code.

`anchors` maps literal anchor names to fields accepted by
[`Anchor`](../src/oap/program/anchors.py). Each anchor has a measured
three-element `point`. Optional fields include `axis`, `region_half`,
`kind`, `attached_to`, `confidence`, `visibility`, `age_since_seen`,
`last_pose` and `dynamic`. Geometry, axes and regions must describe the
same scene as the image and simulator. No example coordinates are bundled.
Initial-observation fields and path suffixes are validated by the selected
profile loader. Pick and Tool-push use `eraser/pose/eraser_refined.json`;
Push uses `spacemouse_box/pose/spacemouse_box_refined.json`; Cup uses
`experiments/initial_observations/lab_cup_offline.json`. Configure these
external paths to describe the registered subject and initial pose.

Set `robot_calibration.real_table_z_m` and `sim_tcp_m` from your own
calibration before execution. They are intentionally unset in the template.
The supplied values must agree with the selected controller profile.
`sim_tcp_anchor` selects the configured TCP anchor. External robot capability
configuration can be located with `OAP_CONFIG_ROOT`, which is forwarded to
the executor.

The four task keys are `pick`, `cup`, `push` and `toolpush`. The
`canonical_program` values are registration path names, not bundled cost
programs. A generated program is copied under that relative name within its
external execution directory. Generated-program execution does not require
a bundled reference cost program. Non-generated reference profiles obtain
their additional program files from `OAP_INPUT_ROOT`.

## Generate and self-review

Set `OPENAI_API_KEY` in the environment and choose the model, reasoning effort
and response length in `config.json`. The default request ceiling is seven
attempts per workspace: draft plus one possible format repair, one self-review,
and two feedback revisions with one possible format repair each. Increase this
explicit ceiling when scheduling multiple trials. Transport failures remain
incomplete; the pipeline does not silently request another generation.

Preview prompts without sending a request:

```bash
python -B -m oap_pipeline.generation --root ../oap-work preview --trial pick_t01
```

Generate one draft and self-review:

```bash
python -B -m oap_pipeline.generation --root ../oap-work generate --trial pick_t01
```

The workspace receives `generation/pick_t01/draft.json`, `review.json`,
request records and a receipt. An invalid or unchanged self-review keeps the
draft. Check the receipt before execution. All tasks use the same twelve
residual types and shared objective-design rules.

## Execute and refine

Execute the reviewed program from the configured initial scene:

```bash
python -B -m oap_pipeline.execute --root ../oap-work --trial pick_t01 \
  --program ../oap-work/generation/pick_t01/review.json \
  --out ../oap-work/execution/pick_t01/a0 --seed 970001 --attempt 0 --run
```

Without `--run`, the command prints the execution specification. Each run
allows at most 250 MPC updates, with a 600-step horizon and 40 applied controls
per update. The configured closing-force cap is 25 N.

Inspect `completion.json` for run status and `evidence.json` for measured
execution feedback. Assess task success using your own task-specific criteria;
completed stages alone do not establish task success. If the task remains
incomplete, request a feedback revision:

```bash
python -B -m oap_pipeline.generation --root ../oap-work feedback \
  --trial pick_t01 --arm review_feedback --attempt 1 \
  --program ../oap-work/generation/pick_t01/review.json \
  --evidence ../oap-work/execution/pick_t01/a0/evidence.json

python -B -m oap_pipeline.execute --root ../oap-work --trial pick_t01 \
  --program ../oap-work/feedback/pick_t01/review_feedback/r1/program.json \
  --out ../oap-work/execution/pick_t01/a1 --seed 970002 --attempt 1 --run
```

The revised program is executable only if its receipt records
`execution_eligible: true`. An invalid or unchanged revision keeps the
preceding program and does not grant another execution.

After another failure, one final feedback round is available:

```bash
python -B -m oap_pipeline.generation --root ../oap-work feedback \
  --trial pick_t01 --arm review_feedback --attempt 2 \
  --program ../oap-work/feedback/pick_t01/review_feedback/r1/program.json \
  --evidence ../oap-work/execution/pick_t01/a1/evidence.json

python -B -m oap_pipeline.execute --root ../oap-work --trial pick_t01 \
  --program ../oap-work/feedback/pick_t01/review_feedback/r2/program.json \
  --out ../oap-work/execution/pick_t01/a2 --seed 970003 --attempt 2 --run
```

For a no-self-review arm, begin with `draft.json` and use
`--arm draft_feedback` throughout. Use a predetermined controller-seed
schedule shared across compared methods.

Feedback contains up to six measured records uniformly spaced in record
index across the whole execution, including the first and last available
records. Fewer than six records are all retained. Stage history separately
preserves visited stages and their last terminal measurements.
