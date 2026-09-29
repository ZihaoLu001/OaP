# Objectives as Policies (OaP)

**Objectives as Policies: Vision-Language Model Cost Programs for Contact-Rich Manipulation**

OaP uses a vision-language model to express a manipulation task as a sequence
of objectives. A shared physics-based MPPI controller executes the objective
program. Simulation feedback supports program revision before deployment.

## Installation

Use Python 3.11. GPU simulation requires Linux, an NVIDIA GPU, and a compatible
CUDA driver. Create the environment outside the source checkout:

```bash
git clone https://github.com/ZihaoLu001/OaP.git
cd OaP
conda create -n oap python=3.11
conda activate oap
python -m pip install "oap[synthesis,gpu-screen] @ git+https://github.com/ZihaoLu001/OaP.git"
```

The Python namespace is `oap`; command-line tools use the `oap-` prefix.
Optional robot support is available through the `robot` extra. Camera,
reconstruction, and tracking dependencies are installed separately.

## Inputs and outputs

Keep scene meshes, observations, calibration, generated programs, and execution
outputs in a separate workspace. This repository contains source code and
configuration templates, not experiment data.

- `OAP_INPUT_ROOT`: external scene and runtime input directory.
- `OAP_CONFIG_ROOT`: external configuration directory; defaults to
  `$OAP_INPUT_ROOT/configs`.
- `OAP_EXTERNAL_ENVS`: external tool-environment configuration, based on
  [external_envs.example.yaml](external_envs.example.yaml).

Supply your robot assets, calibrated scene, initial observation, scene image,
and grounded anchors. The generation pipeline supports a draft, one
self-review, and up to two execution-feedback rounds. See the
[pipeline guide](oap_pipeline/README.md) for configuration and commands.

The robot adapters target Flexiv Rizon4 and the GN01 gripper. Hardware
execution requires robot-specific calibration and authorization.
Simulated success and soft cost penalties are not safety guarantees.

## Source layout

| Directory | Component |
|---|---|
| `src/oap/program/` | Residuals, cost evaluation, synthesis, and stage verification |
| `src/oap/twin/` | Scene construction and batched physics rollouts |
| `src/oap/loop/` | MPC execution and measured observations |
| `src/oap/reconstruct/` | Scene reconstruction interfaces |
| `oap_pipeline/` | Generation, self-review, and execution feedback |

## License

[Apache-2.0](LICENSE). External software dependencies are
described in [third-party notices](THIRD_PARTY_NOTICES.md).
