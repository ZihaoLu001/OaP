"""Match the sim GN01 grasp-center site to the real calibrated 198.12 mm tool.

Role in the two-stage pipeline: the planning twin's TCP site must sit exactly
where the REAL calibrated tool's grasp center is (0.19812 m from the flange),
or every certified grasp lands ~24 mm off on hardware. This module patches a
built W_plan XML in place. The safe anchor is ``"site"`` (move only the site
and trim the fingertip collision by the measured 2.12 mm): the legacy
``"gripper"`` anchor shifted the whole finger mechanism, left sim fingertips
~28 mm longer than the real hand, and its rollouts exploded.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from oap.twin.assets import gripper_joint_targets

logger = logging.getLogger("oap.twin.gripper_tcp")

__all__ = ["REAL_GN01_TCP_M", "patch_sim_gripper_tcp"]

# The real calibrated flange->grasp-center tool length (teach-and-measure).
REAL_GN01_TCP_M = 0.19812


def patch_sim_gripper_tcp(xml_path: Path | str, target_tcp_m: float, anchor: str = "site") -> None:
    """Lengthen the sim GN01 so its grasp-center sits ``target_tcp_m`` from the flange.

    Matches the real calibrated 0.19812 m tool, closing the [tcp-site] -24 mm gap.

    anchor="gripper" (legacy): shifts BOTH ``gn01_grasp_center_site`` (the
    planning TCP) AND ``gn01_articulated_root`` (the whole finger mechanism) by
    the SAME delta, keeping the site at the sim pads' CENTER. Side effect
    (measured): the sim fingertips then reach 226 mm from the flange vs the
    REAL gripper's 198 mm total length -- the sim hand is ~28 mm too long, so
    collision checks are conservative and sim finger contact lands ~28 mm
    lower on the object than real contact. This anchor's rollouts exploded;
    it is kept only for A/B.

    anchor="site" (default): moves ONLY the site. The base GN01 model's pads
    already sit at the physically correct heights (tips 202.3 mm from flange
    vs real 198.1); the real 4-bar closes tip-first so the real grasp-center
    IS at the tip (measured: 199 closed / 197.8 @46mm), i.e. the site belongs
    near the pads' distal end -- matching reality -- not at their center. Also
    trims the fingertip collision boxes by the measured 2.12 mm half-z
    (derivation: closed-tip overshoot 4.13 mm / (2*cos 13.1deg tilt)) so the
    CLOSED collision tip lands at ``target_tcp_m`` exactly. The visual mesh
    keeps a ~1.2 mm non-colliding overhang. Self-verifies by recompiling and
    logging the closed-tip extent.

    The finger LINKAGE calibration (width->gap) is unchanged in both modes.
    Per-run twin only; idempotent-safe (computes the delta from the site's
    current z each call).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    site = next((s for s in root.iter("site") if s.get("name") == "gn01_grasp_center_site"), None)
    artic = next((b for b in root.iter("body") if b.get("name") == "gn01_articulated_root"), None)
    if site is None or artic is None:
        logger.warning("[sim-tcp] gn01_grasp_center_site / gn01_articulated_root not found -> skipped")
        return
    sx, sy, sz = (float(v) for v in site.attrib["pos"].split())
    delta = float(target_tcp_m) - sz
    if abs(delta) < 1e-6:
        return
    site.set("pos", f"{sx} {sy} {sz + delta:.6f}")
    if anchor == "gripper":
        ax, ay, az = (float(v) for v in artic.attrib["pos"].split())
        artic.set("pos", f"{ax} {ay} {az + delta:.6f}")
        tree.write(xml_path)
        logger.info(
            "[sim-tcp] lengthened sim GN01 grasp-center %.5f->%.5f m (shifted "
            "fingers+site %+.1f mm) to match the real tool",
            sz, float(target_tcp_m), 1000 * delta)
        return
    # anchor == "site": pads stay at their (physically correct) place; trim tip boxes.
    _TIP_TRIM_M = 0.00212
    trimmed = 0
    for geom in root.iter("geom"):
        if geom.get("name") in ("gn01_left_finger_tip_collision", "gn01_right_finger_tip_collision"):
            size = [float(v) for v in geom.attrib["size"].split()]
            pos = [float(v) for v in geom.attrib["pos"].split()]
            q = [float(v) for v in (geom.get("quat") or "1 0 0 0").split()]
            w, x, y, z = q
            # box local +z axis in body frame (distal direction, verified cos=+0.974)
            zl = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
            size[2] -= _TIP_TRIM_M
            pos = [pos[i] - _TIP_TRIM_M * zl[i] for i in range(3)]  # keep the proximal face fixed
            geom.set("size", " ".join(f"{v:.6f}" for v in size))
            geom.set("pos", " ".join(f"{v:.6f}" for v in pos))
            trimmed += 1
    tree.write(xml_path)
    logger.info(
        "[sim-tcp] site-only: grasp-center %.5f->%.5f m (fingers NOT shifted; pads "
        "stay at the real heights); trimmed %d tip boxes by %.2f mm half-z",
        sz, float(target_tcp_m), trimmed, _TIP_TRIM_M * 1000)
    try:  # self-check: closed-config collision tip from the flange (target = target_tcp_m)
        import mujoco

        m = mujoco.MjModel.from_xml_path(str(xml_path))
        d = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "lab_snapshot")
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(m, d, kid)
        gj = [j for j in range(m.njnt) if "gn01" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "")]
        for j, v in zip(gj, gripper_joint_targets(0.0)):
            d.qpos[m.jnt_qposadr[j]] = v
        mujoco.mj_forward(m, d)
        l7 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link7")
        R7, p7 = d.xmat[l7].reshape(3, 3), d.xpos[l7]
        tip = -1.0
        for g in range(m.ngeom):
            gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if gn.endswith("finger_tip_collision"):
                s = m.geom_size[g]
                c = np.array([[a, b, cc] for a in (-s[0], s[0]) for b in (-s[1], s[1]) for cc in (-s[2], s[2])])
                Pw = (d.geom_xmat[g].reshape(3, 3) @ c.T).T + d.geom_xpos[g]
                tip = max(tip, float(((Pw - p7) @ R7)[:, 2].max()))
        flange = float(((d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lab_gn01_tool")] - p7) @ R7)[2])
        logger.info(
            "[sim-tcp] self-check: CLOSED collision tip = %.1f mm from flange "
            "(real gripper total = 198.1 mm)", (tip - flange) * 1000)
    except Exception as e:  # noqa: BLE001 - the self-check must never break a build
        logger.warning("[sim-tcp] self-check skipped (%s)", e)
