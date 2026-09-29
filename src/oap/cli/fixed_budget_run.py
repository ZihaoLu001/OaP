"""Run a sealed unified-MPPI profile under the paper's global budget."""
from __future__ import annotations

import sys
from collections.abc import Sequence

from oap.cli import run
from oap.execution_budget import PAPER_GLOBAL_250_V1
from oap.twin.unified_mppi_effort import (
    split_unified_mppi_effort_ablation,
)


_PROFILE_OPTION = "--unified-mppi-effort-profile"
_PROTOCOL_OPTION = "--execution-budget-protocol"


def _add_budget_protocol(argv: Sequence[str]) -> list[str]:
    """Bind the paper budget to one N=1000 unified controller profile."""
    rewritten = list(argv)
    if "-h" in rewritten or "--help" in rewritten:
        return rewritten

    matches: list[tuple[int, str, bool]] = []
    for index, token in enumerate(rewritten):
        if token == _PROFILE_OPTION:
            if index + 1 >= len(rewritten):
                raise ValueError(f"{_PROFILE_OPTION} requires a value")
            matches.append((index + 1, rewritten[index + 1], False))
        elif token.startswith(f"{_PROFILE_OPTION}="):
            matches.append((index, token.split("=", 1)[1], True))

    if len(matches) != 1:
        raise ValueError(
            "fixed-budget execution requires exactly one "
            f"{_PROFILE_OPTION}"
        )

    _, profile, _ = matches[0]
    _, modifier = split_unified_mppi_effort_ablation(profile)
    if modifier != "n1001":
        raise ValueError(
            "fixed-budget execution requires the paper controller modifier "
            "'n1001' (1000 optimization proposals, one mean verification, "
            "and at most one conditional sample verification)"
        )
    if any(
        token == _PROTOCOL_OPTION or token.startswith(f"{_PROTOCOL_OPTION}=")
        for token in rewritten
    ):
        raise ValueError(f"{_PROTOCOL_OPTION} is entry-point controlled")
    rewritten.extend([_PROTOCOL_OPTION, PAPER_GLOBAL_250_V1])
    return rewritten


def main(argv: list[str] | None = None) -> int:
    """The ``oap-run-fixed-budget`` console entry point."""
    source = sys.argv[1:] if argv is None else argv
    try:
        rewritten = _add_budget_protocol(source)
    except ValueError as exc:
        print(f"FIXED_BUDGET_REFUSAL: {exc}", file=sys.stderr)
        return 2
    return run.main(rewritten)


if __name__ == "__main__":
    raise SystemExit(main())
