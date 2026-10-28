"""A second grounded body: Roboparty's RPO (roboto_origin), as a stand-in in the same format as the tendon
humanoid's, so one environment, one trainer and one policy network serve both.

Source: github.com/Roboparty/roboto_origin, modules/rpo_description (CERN-OHL-W-2.0), pinned to the commit in
`RPO_COMMIT`. `fetch()` downloads the MJCF and the 24 STL meshes into third_party/roboparty_rpo (the meshes,
31 MB, are not committed). `build()` writes third_party/roboparty_rpo/mjcf/rpo_standin.xml and .json:

- the original model with its torque motors replaced by position servos using the PD gains and torque limits
  of roboparty_train's actuator configuration (BSD-3-Clause, robolab/assets/robots/roboparty.py): hips 100,
  knees and torso 150, ankles and shoulders 40, elbows 30 and 20 N m/rad; 120 N m on hips, knees and torso,
  27 N m elsewhere;
- a 5 ms timestep, the implicit-fast integrator and softer contacts, as the tendon stand-in uses;
- the same sensors: the IMU (pelvis_quat, pelvis_gyro, pelvis_acc at the model's own `imu` site), a touch
  sensor under each sole, impact sensors around the trunk and the base, servo torques and joint-stop forces;
- its standing pose: roboparty_train's default joint positions at touchdown (it does not stand on its
  servos alone; the policy balances it).

RPO has quasi-direct-drive motors, no lines: its DOFs are of class "QDD", which the line gate does not cover;
its damage gates are the impacts and, in the stand task, falling.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

RPO_COMMIT = "c71fd9287d28104dca111d7c25b908fa3f45f8e4"
RPO_REPO = "Roboparty/roboto_origin"
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RPO_DIR = os.path.join(PROJECT, "third_party", "roboparty_rpo")

# roboparty_train robolab/assets/robots/roboparty.py (BSD-3-Clause): stiffness, damping, effort limit
GAINS = [
    (("thigh_yaw", "thigh_roll", "thigh_pitch"), 100.0, 3.3, 120.0),
    (("knee", "torso"), 150.0, 5.0, 120.0),
    (("ankle_pitch", "ankle_roll"), 40.0, 2.0, 27.0),
    (("arm_pitch", "arm_roll", "arm_yaw"), 40.0, 2.0, 27.0),
    (("elbow_pitch",), 30.0, 1.5, 27.0),
    (("elbow_yaw",), 20.0, 1.0, 27.0),
]
DEFAULT_POSE = {"left_thigh_pitch_joint": -0.1, "left_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2,
                "left_arm_pitch_joint": 0.18, "left_arm_roll_joint": 0.06, "left_elbow_pitch_joint": 0.78,
                "right_thigh_pitch_joint": -0.1, "right_knee_joint": 0.3, "right_ankle_pitch_joint": -0.2,
                "right_arm_pitch_joint": 0.18, "right_arm_roll_joint": -0.06, "right_elbow_pitch_joint": 0.78}
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
IMPACT_BODIES = ("torso_link", "base_link")


def fetch(force: bool = False) -> None:
    """Download the description at RPO_COMMIT with the GitHub CLI (`gh`)."""
    files = ["mjcf/rpo.xml", "LICENSE", "README.md"] + [
        "meshes/%s.STL" % n for n in
        ["base_link", "torso_link"] + ["%s_%s" % (s, p) for s in ("left", "right") for p in
                                       ("thigh_yaw_link", "thigh_roll_link", "thigh_pitch_link", "knee_link",
                                        "ankle_pitch_link", "ankle_roll_link", "arm_pitch_link", "arm_roll_link",
                                        "arm_yaw_link", "elbow_pitch_link", "elbow_yaw_link")]]
    for f in files:
        dest = os.path.join(RPO_DIR, f)
        if os.path.exists(dest) and os.path.getsize(dest) > 0 and not force:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = "repos/%s/contents/modules/rpo_description/%s?ref=%s" % (RPO_REPO, f, RPO_COMMIT)
        data = subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github.raw", url],
                              check=True, capture_output=True).stdout
        with open(dest, "wb") as fh:
            fh.write(data)


def _gains(joint):
    for keys, kp, kv, tmax in GAINS:
        if any(k in joint for k in keys):
            return kp, kv, tmax
    raise KeyError(joint)


def build() -> dict:
    import mujoco

    src = os.path.join(RPO_DIR, "mjcf", "rpo.xml")
    tree = ET.parse(src)
    root = tree.getroot()
    opt = root.find("option")
    for k in ("iterations", "solver"):
        opt.attrib.pop(k, None)
    opt.set("timestep", "0.005")
    opt.set("integrator", "implicitfast")
    for g in root.iter("geom"):
        if g.get("solref"):
            g.set("solref", "0.02 1")
    m0 = mujoco.MjModel.from_xml_path(src)
    joints = [m0.joint(i).name for i in range(m0.njnt) if m0.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    act = root.find("actuator")
    for c in list(act):
        act.remove(c)
    dofs = []
    for j in joints:
        kp, kv, tmax = _gains(j)
        lo, hi = m0.jnt_range[m0.joint(j).id]
        ET.SubElement(act, "position", name=j, joint=j, kp="%.2f" % kp, kv="%.3f" % kv, ctrllimited="true",
                      ctrlrange="%.4f %.4f" % (lo, hi), forcelimited="true", forcerange="%.2f %.2f" % (-tmax, tmax))
        dofs.append(dict(name=j, cls="QDD", kp=kp, kv=kv, torque_max=tmax, lo=float(lo), hi=float(hi),
                         moment_arm_m=1.0, pretension_n=0.0, linear_torque_nm=0.0, line_cont_n=tmax, line_break_n=0.0,
                         stall_nm=tmax, noload_rad_s=25.0 if tmax > 100 else 8.0))
    # sole and impact sites from the compiled meshes
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    bodies = {b.get("name"): b for b in root.iter("body")}
    for foot in FEET:
        bid = m0.body(foot).id
        gid = [g for g in range(m0.ngeom) if m0.geom_bodyid[g] == bid][0]
        mid = m0.geom_dataid[gid]
        v = m0.mesh_vert[m0.mesh_vertadr[mid]:m0.mesh_vertadr[mid] + m0.mesh_vertnum[mid]]
        # mesh vertices are in the geom frame; bring them to the body frame
        R = _quat_mat(m0.geom_quat[gid])
        vb = v @ R.T + m0.geom_pos[gid]
        zmin = vb[:, 2].min()
        bottom = vb[vb[:, 2] < zmin + 0.004]
        c = np.r_[(bottom[:, :2].min(0) + bottom[:, :2].max(0)) / 2, zmin + 0.003]
        half = (bottom[:, :2].max(0) - bottom[:, :2].min(0)) / 2
        ET.SubElement(bodies[foot], "site", name="%s_sole" % foot, type="box",
                      size="%.4f %.4f 0.006" % (half[0], half[1]), pos="%.4f %.4f %.4f" % tuple(c))
    for b in IMPACT_BODIES:
        bid = m0.body(b).id
        gid = [g for g in range(m0.ngeom) if m0.geom_bodyid[g] == bid][0]
        ET.SubElement(bodies[b], "site", name="%s_impact" % b, type="sphere",
                      size="%.4f" % (m0.geom_rbound[gid] + 0.01), pos="%.4f %.4f %.4f" % tuple(m0.geom_pos[gid]))
    sens = root.find("sensor")
    if sens is None:
        sens = ET.SubElement(root, "sensor")
    for c in list(sens):
        sens.remove(c)
    ET.SubElement(sens, "framequat", name="pelvis_quat", objtype="site", objname="imu")
    ET.SubElement(sens, "gyro", name="pelvis_gyro", site="imu")
    ET.SubElement(sens, "accelerometer", name="pelvis_acc", site="imu")
    for foot in FEET:
        ET.SubElement(sens, "touch", name="%s_touch" % foot, site="%s_sole" % foot)
    for b in IMPACT_BODIES:
        ET.SubElement(sens, "touch", name="%s_impact" % b, site="%s_impact" % b)
    for j in joints:
        ET.SubElement(sens, "actuatorfrc", name="%s_tau" % j, actuator=j)
    for j in joints:
        ET.SubElement(sens, "jointlimitfrc", name="%s_stop" % j, joint=j)
    out_xml = os.path.join(RPO_DIR, "mjcf", "rpo_standin.xml")
    tree.write(out_xml)
    # the standing pose, settled on the servos
    m = mujoco.MjModel.from_xml_path(out_xml)
    d = mujoco.MjData(m)
    ctrl = np.array([DEFAULT_POSE.get(j, 0.0) for j in joints])
    for j, q in zip(joints, ctrl):
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = q
    mujoco.mj_forward(m, d)
    # touch down only: RPO does not stand on its servos alone (its original model with roboparty_train's gains
    # falls within 1.5 s too), so the policy starts from the pose at touchdown and has to balance from there
    for _ in range(int(0.05 / m.opt.timestep)):
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)
    up = d.xmat[m.body("base_link").id].reshape(3, 3)[2, 2]
    spec = dict(model="third_party/roboparty_rpo/mjcf/rpo_standin.xml", source="rl/rpo_body.py",
                origin="%s@%s modules/rpo_description (CERN-OHL-W-2.0); gains from roboparty_train (BSD-3-Clause)"
                       % (RPO_REPO, RPO_COMMIT),
                timestep=m.opt.timestep, root_body="base_link", feet=list(FEET), imu_site="imu",
                impact_bodies=list(IMPACT_BODIES), dofs=dofs, default_ctrl=ctrl.tolist(),
                stand_qpos=d.qpos.tolist(), stand_height=float(d.qpos[2]), mass=float(m.body_subtreemass[0]))
    with open(out_xml.replace(".xml", ".json"), "w") as f:
        json.dump(spec, f, indent=1)
    return dict(xml=out_xml, dofs=len(dofs), stand_height=spec["stand_height"], upright=float(up),
                feet_load_n=[float(d.sensordata[m.sensor_adr[m.sensor("%s_touch" % f).id]]) for f in FEET],
                mass=spec["mass"])


def _quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


if __name__ == "__main__":
    fetch()
    print(json.dumps(build(), indent=1))
