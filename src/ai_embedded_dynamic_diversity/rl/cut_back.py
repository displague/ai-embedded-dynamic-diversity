# python -m ai_embedded_dynamic_diversity.rl.cut_back <champion.pt> <policy.pt> <out.pt> [<policy.pt> <out.pt> ...]
"""Cut an RL-trained core (grown from a champion) back to the champion's channel counts, so the sibling's own
cross-eval can score it against the untouched champion: does learning a body make it forget its old world?"""
import sys, torch

champ_path = sys.argv[1]
champ = torch.load(champ_path, map_location="cpu", weights_only=False)
cfg = dict(champ["model_config"])
for rl_path, out in zip(sys.argv[2::2], sys.argv[3::2]):
    rl = torch.load(rl_path, map_location="cpu", weights_only=False)
    sd = dict(rl["core"])
    sd["passive.encoder.0.weight"] = sd["passive.encoder.0.weight"][:, :cfg["signal_dim"]].clone()
    sd["router.base_map"] = sd["router.base_map"][:cfg["io_channels"]].clone()
    for k, v in champ["model"].items():
        assert sd[k].shape == v.shape, (k, sd[k].shape, v.shape)
    drift = {k: float((sd[k] - champ["model"][k]).norm() / (champ["model"][k].norm() + 1e-9)) for k in champ["model"]}
    top = sorted(drift.items(), key=lambda kv: -kv[1])[:4]
    torch.save({"model": sd, "model_config": cfg, "profile": champ.get("profile"),
                "source": "cut back from %s (rl/ppo.py policy grown from %s)" % (rl_path, champ_path)}, out)
    print(out, "mean relative weight change %.3f, largest %s" % (sum(drift.values()) / len(drift),
          ", ".join("%s %.2f" % kv for kv in top)))
