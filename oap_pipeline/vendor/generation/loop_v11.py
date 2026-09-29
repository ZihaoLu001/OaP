"""Model requests and Table I validation shared by all four tasks."""
import argparse, base64, hashlib, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trim_schema_v11 as trim_schema

def _held_on_entry(stages: list, stage_index: int) -> set[str]:
    """Only the immediately preceding measured terminal can establish entry hold."""
    if stage_index == 0 or not isinstance(stages[stage_index - 1], dict):
        return set()
    terminal = stages[stage_index - 1].get("terminal") or []
    held = {x["object_anchor"] for x in terminal if isinstance(x, dict)
            and x.get("type") == "object_held"
            and isinstance(x.get("object_anchor"), str) and x["object_anchor"]}
    released = {x.get("object_anchor") for x in terminal if isinstance(x, dict)
                and x.get("type") == "not_held"
                and isinstance(x.get("object_anchor"), str)}
    return held - released


def normalize_patterns(prog: dict) -> dict:
    """Keep model-authored objectives unchanged; validation reports errors."""
    return json.loads(json.dumps(prog))



MODEL = os.environ.get("OAP_VLM_MODEL", "gpt-5.6-sol")
EFFORT = os.environ.get("FR_REASONING_EFFORT", "high")
MAX_TOKENS = int(os.environ.get("FR_MAX_TOKENS", "32768"))
CALL_CAP = 7
LEDGER = HERE / "logs" / "api_calls.jsonl"

def n_calls() -> int:
    return sum(1 for _ in LEDGER.open()) if LEDGER.exists() else 0


def ask(system: str, user: str, image: Path, tag: str) -> tuple[str, dict]:
    """One OpenAI call, high reasoning, image before text.  Refuses past the cap."""
    if n_calls() >= CALL_CAP:
        raise SystemExit(f"API call cap reached ({CALL_CAP}); refusing {tag}")
    import openai
    assert MODEL.lower().startswith("gpt"), MODEL
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in the environment before generation.")
    client = openai.OpenAI(max_retries=0, timeout=float(os.environ.get("OAP_VLM_TIMEOUT", "900")))
    b64 = base64.b64encode(image.read_bytes()).decode("ascii")
    parts = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
             {"type": "text", "text": user}]
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL, reasoning_effort=EFFORT, max_completion_tokens=MAX_TOKENS,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": parts}])
    text = resp.choices[0].message.content or ""
    u = resp.usage
    rec = {"n": n_calls() + 1, "tag": tag, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "model_requested": MODEL, "model_answered": resp.model, "response_id": resp.id,
           "reasoning_effort": EFFORT, "latency_s": round(time.time() - t0, 1),
           "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
           "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None),
                                       "reasoning_tokens", None),
           "system_sha16": hashlib.sha256(system.encode()).hexdigest()[:16],
           "user_chars": len(user), "empty": not text}
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    if not text:
        raise RuntimeError(f"empty completion for {tag} (call {rec['n']})")
    return text, rec


# ------------------------------------------------------------------ parsing
def first_json_object(text: str, want_key: str) -> dict | None:
    import re
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            cand, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict) and want_key in cand:
            return cand
    return None


def release_hard_rules(data: dict) -> list:
    """The shared schema validator owns runtime constraints."""
    return []


def movable_field_errors(data: dict, known_bodies) -> list:
    """Every stage declares the scene bodies it intends to move."""
    errs = []
    for i, s in enumerate(data.get("stages") or []):
        if not isinstance(s, dict):
            continue
        m = s.get("movable_objects")
        if m is None:
            errs.append(f'stage[{i}]: missing "movable_objects" -- list the object '
                        f'names this stage intends to move, [] if none')
        elif not (isinstance(m, list) and all(isinstance(x, str) for x in m)):
            errs.append(f'stage[{i}].movable_objects: must be a JSON list of object names')
        elif known_bodies and not set(m) <= set(known_bodies):
            errs.append(f'stage[{i}].movable_objects: unknown names '
                        f'{sorted(set(m) - set(known_bodies))}; known objects: {sorted(known_bodies)}')
    return errs


def to_program(data: dict, instruction: str, provenance: str, anchors, known_bodies=()):
    """Validate without silently adding, removing or changing objectives."""
    from oap.program.synthesis import validate_program
    from oap.program import TaskProgram
    data = normalize_patterns(data)
    mov_errs = movable_field_errors(data, known_bodies)
    # Runtime-only material-frame anchors are validated after scene grounding.
    # Do not invent raw orientation axes in the VLM scene context.
    errs = (validate_program(data, anchors, require_grounded_frame_contract=False)
            + release_hard_rules(data)
            + trim_schema.allowlist_errors(data) + mov_errs)
    if errs:
        return None, errs
    d = dict(data)
    d["instruction"] = instruction
    d["provenance"] = provenance
    try:
        return TaskProgram.from_dict(d), []
    except (ValueError, TypeError, KeyError) as e:
        return None, [f"{type(e).__name__}: {e}"]


def attach_movables(final_program: dict, source_program: dict) -> dict:
    """Re-attach each stage's movable_objects (VLM-authored, validated by
    movable_field_errors) onto the canonical validated program dict."""
    src = (source_program or {}).get("stages") or []
    for fs, ss in zip(final_program.get("stages", []), src):
        if isinstance(fs, dict) and isinstance(ss, dict) and "movable_objects" in ss:
            fs["movable_objects"] = list(ss["movable_objects"])
    return final_program
