"""Table I allowlist, shared by all generation tasks."""
from oap.program.synthesis import PAPER_TERM_TYPES

ALLOWED = PAPER_TERM_TYPES


def trim_system(system):
    """The synthesis prompt already contains exactly Table I."""
    return system


def trim_user(user):
    return user


def allowlist_errors(data):
    errors = []
    for index, stage in enumerate(data.get("stages") or []):
        if not isinstance(stage, dict):
            continue
        for scope in ("running", "terminal"):
            for term in stage.get(scope) or []:
                kind = term.get("type") if isinstance(term, dict) else None
                if kind not in ALLOWED:
                    errors.append(f"stage[{index}].{scope}: {kind!r} is not a Table I residual")
    return errors
