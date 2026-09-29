"""``oap-reconstruct``: the Stage-1 offline asset pipeline CLI.

Role in the two-stage pipeline: one orchestrated executable (replacing a pile
of loose scripts) that turns a lab-host RGB-D capture into a portable scene
bundle for ``oap-run``. Subcommands map 1:1 onto the reconstruction DAG::

    capture | mesh | scale | collision | pose | tracking | priors | manifest | all

``capture`` is LAB HOST ONLY (the ZED lives there); every other step runs
wherever the required external environment exists (selected with
``--external-envs <profile>``). All steps are idempotent and
resumable; ``all`` walks the whole DAG and re-runs only stale steps.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from oap.reconstruct import bundle as bundle_mod
from oap.reconstruct import capture as capture_mod
from oap.reconstruct.external import ExternalEnvError, ExternalEnvs

logger = logging.getLogger("oap.cli.reconstruct")

__all__ = ["build_parser", "main"]

_PER_OBJECT_STEPS = (
    "mesh",
    "scale",
    "collision",
    "pose",
    "tracking",
    "priors",
)


def _parse_object_pairs(pairs: list[str]) -> dict[str, str]:
    """Parse ``name=prompt`` strings into an ordered mapping."""
    objects: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(
                f"--objects entries must be 'name=prompt', got {pair!r}"
            )
        name, prompt = pair.split("=", 1)
        name, prompt = name.strip(), prompt.strip()
        if not name or not prompt:
            raise argparse.ArgumentTypeError(
                f"--objects entries must be 'name=prompt', got {pair!r}"
            )
        objects[name] = prompt
    return objects


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--external-envs",
        default=None,
        help="host profile from your external external_envs.yaml "
             "(for example, local) or a path to a YAML with the same schema; "
             "default: $OAP_EXTERNAL_ENVS, then external config root",
    )
    parser.add_argument("--force", action="store_true",
                        help="re-run even when step inputs/params are unchanged")
    # Per-tool interpreter/root overrides (highest precedence).
    parser.add_argument("--observer-python", type=Path, default=None)
    parser.add_argument("--sam3d-python", type=Path, default=None)
    parser.add_argument("--sam3d-root", type=Path, default=None)
    parser.add_argument("--foundationpose-python", type=Path, default=None)
    parser.add_argument("--any6d-root", type=Path, default=None)
    parser.add_argument("--qwen-python", type=Path, default=None)


def _add_bundle_object(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", type=Path, required=True,
                        help="scene-bundle directory")
    parser.add_argument("--object", action="append", default=None,
                        help="object name (repeatable; default: every object "
                             "registered in <bundle>/bundle.json)")
    parser.add_argument("--capture-dir", type=Path, default=None,
                        help="source capture dir (only needed if the packet "
                             "was not imported into the bundle yet)")


def _add_mesh_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-texture-baking", action="store_true",
                        help="geometry-only SAM3D diagnostic; production bundle "
                             "assembly still applies the automatic pose-quality gate")
    parser.add_argument("--mesh-only-lowmem", action="store_true",
                        help="16GB fallback: mesh decoder only, raw vertex-color OBJ")
    parser.add_argument("--mesh-extract-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-sparse-coords", type=int, default=None)
    parser.add_argument("--checkpoint-tag", default="hf")
    parser.add_argument("--seed", type=int, default=42)


def _add_roles_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--static", action="append", default=None, metavar="NAME",
                        help="mark NAME movable:false in the manifest (repeatable)")
    parser.add_argument("--open-top", action="append", default=None, metavar="NAME",
                        help="model NAME as an open-top container (repeatable)")
    parser.add_argument("--label", action="append", default=None, metavar="NAME=TEXT",
                        help="label handed to the VLM for PHYSICAL PRIORS, when it "
                             "must differ from the SAM3 prompt. The two pull "
                             "opposite ways: segmentation wants a broad phrase "
                             "that matches, priors want a specific noun that "
                             "identifies the material. Defaults to the prompt.")
    parser.add_argument("--wall-thickness-m", type=float, default=0.006)
    parser.add_argument("--description", default=None)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``oap-reconstruct`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="oap-reconstruct",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_capture = sub.add_parser(
        "capture",
        help="LAB HOST ONLY: capture one ZED frame + SAM3 masks into observation packets",
    )
    p_capture.add_argument("--capture-dir", type=Path, required=True)
    p_capture.add_argument("--object", default=None, help="object name (with --prompt)")
    p_capture.add_argument("--prompt", default=None, help="SAM3 text prompt for --object")
    p_capture.add_argument("--objects", nargs="+", default=None, metavar="NAME=PROMPT",
                           help="multiple objects segmented from the SAME frame")
    p_capture.add_argument("--settle-frames", type=int, default=15)
    _add_common(p_capture)

    for step in _PER_OBJECT_STEPS:
        p_step = sub.add_parser(step, help=f"run the {step.upper()} step for one or all objects")
        _add_bundle_object(p_step)
        _add_common(p_step)
        if step == "mesh":
            _add_mesh_flags(p_step)
        elif step == "pose":
            p_step.add_argument("--downscale", type=float, default=0.5,
                                help="RGB-D downscale for Any6D (0.5 = the 16GB recipe)")
            p_step.add_argument("--iterations", type=int, default=5)
        elif step == "priors":
            p_step.add_argument("--model-id", default=None,
                                help="transformers model id (default: profile model_id)")

    p_manifest = sub.add_parser("manifest", help="assemble <bundle>/manifest.json")
    p_manifest.add_argument("--bundle", type=Path, required=True)
    _add_roles_flags(p_manifest)
    _add_common(p_manifest)

    p_all = sub.add_parser(
        "all",
        help="walk the full DAG capture->mesh->scale->collision->pose->tracking"
             "->priors->manifest "
             "(idempotent + resume-safe)",
    )
    p_all.add_argument("--bundle", type=Path, required=True)
    p_all.add_argument("--capture-dir", type=Path, default=None,
                       help="source capture dir (required on the first run)")
    p_all.add_argument("--objects", nargs="+", default=None, metavar="NAME=PROMPT",
                       help='e.g. --objects "zed_box=black rectangular box" '
                            '"amazon_box=brown cardboard box"')
    p_all.add_argument("--downscale", type=float, default=0.5,
                       help="forwarded to the pose step")
    p_all.add_argument("--model-id", default=None, help="forwarded to the priors step")
    _add_mesh_flags(p_all)
    _add_roles_flags(p_all)
    _add_common(p_all)

    return parser


def _load_envs(args: argparse.Namespace) -> ExternalEnvs:
    overrides: dict[str, Any] = {
        "observer.python": getattr(args, "observer_python", None),
        "sam3d.python": getattr(args, "sam3d_python", None),
        "sam3d.root": getattr(args, "sam3d_root", None),
        "foundationpose.python": getattr(args, "foundationpose_python", None),
        "foundationpose.any6d_root": getattr(args, "any6d_root", None),
        "qwen.python": getattr(args, "qwen_python", None),
    }
    return ExternalEnvs.load(
        getattr(args, "external_envs", None),
        overrides={k: str(v) for k, v in overrides.items() if v},
    )


def _object_names(args: argparse.Namespace) -> list[str]:
    if getattr(args, "object", None):
        return [str(n) for n in args.object]
    reg = bundle_mod.load_registry(args.bundle)
    names = list(reg.get("objects", {}))
    if not names:
        raise RuntimeError(
            f"bundle {args.bundle} has no registered objects; run "
            f"`oap-reconstruct all --objects \"name=prompt\" ...` once, or "
            f"pass --object NAME."
        )
    return names


def _step_options(args: argparse.Namespace, step: str) -> dict[str, Any]:
    if step == "mesh":
        return {
            "texture_baking": not args.no_texture_baking,
            "mesh_only_lowmem": args.mesh_only_lowmem,
            "mesh_extract_device": args.mesh_extract_device,
            "max_sparse_coords": args.max_sparse_coords,
            "checkpoint_tag": args.checkpoint_tag,
            "seed": args.seed,
            "allow_untextured": False,
        }
    if step == "pose":
        return {
            "downscale": args.downscale,
            "iterations": args.iterations,
        }
    if step == "priors":
        return {"model_id": args.model_id}
    return {}


def _cmd_capture(args: argparse.Namespace) -> int:
    objects: dict[str, str] = {}
    if args.objects:
        objects.update(_parse_object_pairs(list(args.objects)))
    if args.object or args.prompt:
        if not (args.object and args.prompt):
            raise RuntimeError("--object and --prompt must be given together")
        objects[str(args.object)] = str(args.prompt)
    if not objects:
        raise RuntimeError('give --object NAME --prompt "..." or --objects "name=prompt" ...')
    envs = _load_envs(args)
    packets = capture_mod.capture_objects(
        args.capture_dir, objects, envs, force=args.force,
        settle_frames=args.settle_frames,
    )
    for p in packets:
        print(p)
    return 0


def _cmd_step(args: argparse.Namespace, step: str) -> int:
    envs = _load_envs(args)
    options = _step_options(args, step)
    for name in _object_names(args):
        # The capture import is cheap and safe to run implicitly so per-step
        # commands work right after a capture-dir sync.
        packet = capture_mod.packet_dir_in_bundle(args.bundle, name)
        if not (packet / "observation_manifest.json").exists():
            bundle_mod.run_step(args.bundle, name, "capture", envs,
                                capture_dir=args.capture_dir)
        result = bundle_mod.run_step(
            args.bundle, name, step, envs,
            capture_dir=args.capture_dir, force=args.force, options=options,
        )
        status = "skipped (current)" if result.get("skipped") else "done"
        print(f"{step} {name}: {status}")
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    bundle_mod.update_registry(
        args.bundle,
        labels=(_parse_object_pairs(list(args.label)) if getattr(args, "label", None)
                else None),
        static=args.static or None,
        open_top=args.open_top or None,
    )
    path = bundle_mod.assemble_manifest(
        args.bundle,
        wall_thickness_m=args.wall_thickness_m,
        description=args.description,
    )
    print(path)
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    envs = _load_envs(args)
    objects = _parse_object_pairs(list(args.objects)) if args.objects else None
    labels = _parse_object_pairs(list(args.label)) if getattr(args, "label", None) else None
    step_options = {
        "mesh": {
            "texture_baking": not args.no_texture_baking,
            "mesh_only_lowmem": args.mesh_only_lowmem,
            "mesh_extract_device": args.mesh_extract_device,
            "max_sparse_coords": args.max_sparse_coords,
            "checkpoint_tag": args.checkpoint_tag,
            "seed": args.seed,
            "allow_untextured": False,
        },
        "pose": {"downscale": args.downscale},
        "priors": {"model_id": args.model_id},
    }
    manifest = bundle_mod.run_all(
        args.bundle,
        envs,
        capture_dir=args.capture_dir,
        objects=objects,
        labels=labels,
        static=args.static or (),
        open_top=args.open_top or (),
        force=args.force,
        step_options=step_options,
        wall_thickness_m=args.wall_thickness_m,
        description=args.description,
    )
    print(manifest)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``oap-reconstruct`` console script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    try:
        if args.command == "capture":
            return _cmd_capture(args)
        if args.command in _PER_OBJECT_STEPS:
            return _cmd_step(args, args.command)
        if args.command == "manifest":
            return _cmd_manifest(args)
        if args.command == "all":
            return _cmd_all(args)
        raise RuntimeError(f"unknown command {args.command!r}")
    except (ExternalEnvError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"oap-reconstruct: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("oap-reconstruct: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
