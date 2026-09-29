"""Create a verified release descriptor for one relocated planning XML."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oap.loop.planner_profile import PlannerProfile
from oap.remote_planner.deployment import create_release_bundle
from oap.twin.batched_rollout import (
    DEFAULT_CEM_ELITE_FRACTION,
    DEFAULT_CEM_MIN_STD_FRACTION,
    DEFAULT_CEM_ROUNDS,
    PREDICTIVE_SAMPLER_MODES,
    PREDICTIVE_SAMPLER_SINGLE_SCALE,
    PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oap-planner-bundle")
    parser.add_argument("--plan-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, required=True)
    parser.add_argument("--horizon-steps", type=int, required=True)
    parser.add_argument("--num-knots", type=int, required=True)
    parser.add_argument(
        "--sigma-fraction", type=float, required=True
    )
    parser.add_argument(
        "--sampler-mode",
        choices=PREDICTIVE_SAMPLER_MODES,
        default=PREDICTIVE_SAMPLER_SINGLE_SCALE,
    )
    parser.add_argument(
        "--local-sigma-fraction",
        type=float,
        default=PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION,
    )
    parser.add_argument(
        "--cem-rounds", type=int, default=DEFAULT_CEM_ROUNDS
    )
    parser.add_argument(
        "--cem-elite-fraction",
        type=float,
        default=DEFAULT_CEM_ELITE_FRACTION,
    )
    parser.add_argument(
        "--cem-min-std-fraction",
        type=float,
        default=DEFAULT_CEM_MIN_STD_FRACTION,
    )
    parser.add_argument(
        "--execution-prefix-steps", type=int, required=True
    )
    parser.add_argument("--max-stage-cycles", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    release = create_release_bundle(
        plan_xml=args.plan_xml,
        output=args.output,
        planner_profile=PlannerProfile(
            pool_size=args.pool_size,
            horizon_steps=args.horizon_steps,
            num_knots=args.num_knots,
            sigma_fraction=args.sigma_fraction,
            sampler_mode=args.sampler_mode,
            local_sigma_fraction=args.local_sigma_fraction,
            cem_rounds=args.cem_rounds,
            cem_elite_fraction=args.cem_elite_fraction,
            cem_min_std_fraction=args.cem_min_std_fraction,
            execution_prefix_steps=args.execution_prefix_steps,
            max_stage_cycles=args.max_stage_cycles,
        ),
    )
    print(json.dumps(release.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
