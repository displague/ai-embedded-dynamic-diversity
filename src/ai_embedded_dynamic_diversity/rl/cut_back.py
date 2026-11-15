# python -m ai_embedded_dynamic_diversity.rl.cut_back <champion.pt> <policy.pt> <out.pt> [<policy.pt> <out.pt> ...]
"""Cut an RL-trained core (grown from a champion) back to the champion's channel counts, so the sibling's own
cross-eval can score it against the untouched champion: does learning a body make it forget its old world?

A rooted policy (rl/actor.RootedCore) keeps its body with it: the cut core carries `bodies`, the roots that were
cut off (the body's input columns and router rows, its noise and its input scaling), so `embody` can put them
back on this core or on one evolved from it.

    python -m ai_embedded_dynamic_diversity.rl.cut_back embody <core.pt> <body> <out policy.pt> [<roots core.pt>]
    python -m ai_embedded_dynamic_diversity.rl.cut_back pack <champion.pt> <core with bodies.pt> <out champion.pt> [note]
"""
import sys

import torch


def cut(champ_path, rl_path, out):
    champ = torch.load(champ_path, map_location="cpu", weights_only=False)
    cfg = dict(champ["model_config"])
    rl = torch.load(rl_path, map_location="cpu", weights_only=False)
    rooted = bool(rl.get("grow", {}).get("roots"))
    full = {(k[len("inner."):] if rooted else k): v for k, v in rl["core"].items()}
    sd = dict(full)
    sd["passive.encoder.0.weight"] = sd["passive.encoder.0.weight"][:, :cfg["signal_dim"]].clone()
    sd["router.base_map"] = sd["router.base_map"][:cfg["io_channels"]].clone()
    for k, v in champ["model"].items():
        assert sd[k].shape == v.shape, (k, sd[k].shape, v.shape)
    drift = {k: float((sd[k] - champ["model"][k]).norm() / (champ["model"][k].norm() + 1e-9)) for k in champ["model"]}
    top = sorted(drift.items(), key=lambda kv: -kv[1])[:4]
    saved = {"model": sd, "model_config": cfg, "profile": champ.get("profile"),
             "source": "cut back from %s (rl/ppo.py policy grown from %s)" % (rl_path, champ_path)}
    body = None
    if rooted:
        body = rl["env"].get("body", "tendon32")
        saved["bodies"] = dict(champ.get("bodies") or {})
        saved["bodies"][body] = dict(
            encoder_cols=full["passive.encoder.0.weight"][:, cfg["signal_dim"]:].clone(),
            router_rows=full["router.base_map"][cfg["io_channels"]:].clone(),
            log_std=rl["log_std"], obs_norm=rl["obs_norm"], env=rl["env"], ppo=rl["ppo"], dof_names=rl["dof_names"],
            grow=rl["grow"], trunk_config=rl.get("trunk_config"))
    torch.save(saved, out)
    print(out, "mean relative weight change %.3f, largest %s" % (sum(drift.values()) / len(drift),
          ", ".join("%s %.2f" % kv for kv in top)) + ("; roots kept for %s" % body if rooted else ""))


def embody(core_path, body, out, roots_path=None):
    """A policy rl/tendon_eval.py and rl/ppo.py can load, from a core checkpoint and a body's roots: the core's own
    (`bodies[body]`), or those of another core checkpoint `roots_path` (an evolved core tried on its parent's body)."""
    core = torch.load(core_path, map_location="cpu", weights_only=False)
    b = torch.load(roots_path or core_path, map_location="cpu", weights_only=False)["bodies"][body]
    sd = {k: v.clone() for k, v in core["model"].items()}
    sd["passive.encoder.0.weight"] = torch.cat([sd["passive.encoder.0.weight"], b["encoder_cols"]], dim=1)
    sd["router.base_map"] = torch.cat([sd["router.base_map"], b["router_rows"]], dim=0)
    torch.save(dict(core={"inner." + k: v for k, v in sd.items()}, log_std=b["log_std"], arch="core",
                    core_config=None, critic=None, obs_norm=b["obs_norm"], priv_norm=None, env=b["env"], ppo=b["ppo"],
                    grow=dict(b["grow"], checkpoint=core_path), trunk_config=b.get("trunk_config"),
                    dof_names=b["dof_names"]), out)
    print(out, "the core of %s with the %s roots of %s" % (core_path, body, roots_path or core_path))


def pack(champion_path, core_path, out, note=""):
    """A champion file with a body: everything `champion_path` holds (its core, config, training record), plus the
    `bodies` of `core_path`, a core cut from a policy whose trunk was this champion's, frozen. Refuses a trunk that
    differs: the roots were learned on that trunk and no other."""
    champ = torch.load(champion_path, map_location="cpu", weights_only=False)
    core = torch.load(core_path, map_location="cpu", weights_only=False)
    for k, v in champ["model"].items():
        if not torch.equal(v, core["model"][k]):
            raise ValueError("%s: %s differs from the champion's; pack only roots learned on its frozen trunk" % (core_path, k))
    champ["bodies"] = dict(champ.get("bodies") or {}, **core["bodies"])
    champ["body_note"] = note
    torch.save(champ, out)
    print(out, "bodies:", ", ".join(champ["bodies"]))


if __name__ == "__main__":
    if sys.argv[1] == "embody":
        embody(*sys.argv[2:])
    elif sys.argv[1] == "pack":
        pack(*sys.argv[2:])
    else:
        for rl_path, out in zip(sys.argv[2::2], sys.argv[3::2]):
            cut(sys.argv[1], rl_path, out)
