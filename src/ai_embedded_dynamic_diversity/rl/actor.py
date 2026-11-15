"""The model core as a PPO actor, grown from a checkpoint to a new body's channel counts.

A champion checkpoint was trained with a fixed number of anonymous input signals (`signal_dim`) and output
channels (`io_channels`). A body with more of either gets a grown copy:

- the passive encoder's first layer gains input columns initialised to zero, so the grown core computes
  exactly what the champion did from the inputs it already had, and learns what the new ones mean;
- the router gains output rows initialised as the originals were (small random); its softmax mixing cannot
  output zero, so the new channels start near the mean readiness, close to the old ones' starting range;
- everything else (memory, gates, heads) is copied unchanged.

The router's outputs lie in (0, 1); the action mean is 2 * output - 1, so a core whose outputs sit near 0.5
starts out holding the standing pose. A learned log standard deviation per action makes the policy Gaussian.
The memory tensor is the recurrent state PPO carries between steps.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ai_embedded_dynamic_diversity.models.core import ModelCore

CORE_KEYS = ("edge_nodes", "io_channels", "hidden_dim", "signal_dim", "memory_slots", "memory_dim",
             "max_remap_groups", "gating_mode", "topk_gating", "enable_dmd_gating", "enable_phase_gating",
             "enable_multi_scale_gating", "emergent_signal_dim")
DEFAULT_CORE = dict(edge_nodes=64, io_channels=16, hidden_dim=64, signal_dim=48, memory_slots=24, memory_dim=32,
                    max_remap_groups=8, gating_mode="sigmoid", topk_gating=0, enable_dmd_gating=False,
                    enable_phase_gating=False, enable_multi_scale_gating=True, emergent_signal_dim=8)


def load_checkpoint(path: str) -> tuple[dict, dict]:
    """(core config, core state dict) from a champion checkpoint of this project, or from a policy saved by
    rl/ppo.py (its core, trained on some body), so a policy trained on one body can start another's training."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("arch") in ("mlp", "gru"):
        raise ValueError("%s is a plain-network policy; it cannot start a core's training" % path)
    if "core" in ck and "grow" in ck:                      # an rl/ppo.py policy
        g = ck["grow"]
        if g.get("roots"):
            raise ValueError("%s is a rooted policy: cut its roots off first (rl/cut_back.py)" % path)
        cfg = dict(load_checkpoint(g["checkpoint"])[0]) if g.get("checkpoint") else dict(DEFAULT_CORE)
        cfg.update(signal_dim=g["signal_dim"], io_channels=g["io_channels"])
        return cfg, ck["core"]
    cfg = {k: ck["model_config"][k] for k in CORE_KEYS if k in ck["model_config"]}
    return cfg, ck["model"]


class RootedCore(nn.Module):
    """A champion's core with a body's own roots: the body's observation enters through input columns of its own
    (after the champion's), and its action leaves through router rows of its own (after the champion's). The
    champion's own input columns see zero and its own output rows are not read, so learning the body sends no
    gradient to either: only the trunk between them is shared with the champion's world, and cutting the roots
    off (rl/cut_back.py) gives back a core of the champion's size whose first layer and router are untouched.
    The interface is the plain networks': `signal_dim` and `io_channels` are the body's."""

    arch = "core"

    def __init__(self, inner: ModelCore, sig_off: int, io_off: int):
        super().__init__()
        self.inner, self.sig_off, self.io_off = inner, sig_off, io_off
        self.signal_dim, self.io_channels = inner.signal_dim - sig_off, inner.io_channels - io_off
        for k in ("edge_nodes", "hidden_dim", "memory_slots", "memory_dim", "max_remap_groups"):
            setattr(self, k, getattr(inner, k))

    def init_memory(self, batch_size: int, memory_slots: int, memory_dim: int, device) -> torch.Tensor:
        return self.inner.init_memory(batch_size, memory_slots, memory_dim, device)

    def forward(self, signal: torch.Tensor, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        pad = signal.new_zeros(signal.shape[:-1] + (self.sig_off,))
        out = self.inner(torch.cat([pad, signal], dim=-1), memory)
        return dict(io=out["io"][..., self.io_off:], memory=out["memory"])


def grow_core(signal_dim: int, io_channels: int, checkpoint: str | None = None, seed: int = 0,
              roots: bool = False, roots_from: str | None = None) -> tuple[ModelCore, dict]:
    """A ModelCore for `signal_dim` inputs and `io_channels` outputs: grown from `checkpoint`, or freshly
    initialised with the same architecture (the comparison baseline) when `checkpoint` is None.
    `roots`: a RootedCore instead (the body's inputs and outputs beside the champion's, not over them);
    `roots_from`: a rooted policy saved by rl/ppo.py whose roots (input columns, router rows) are copied in, so
    a new trunk starts with the body an earlier one learned."""
    torch.manual_seed(seed)
    if roots:
        if not checkpoint:
            raise ValueError("roots need a checkpoint to root")
        old_cfg, _ = load_checkpoint(checkpoint)
        inner, report = grow_core(old_cfg["signal_dim"] + signal_dim, old_cfg["io_channels"] + io_channels,
                                  checkpoint, seed)
        report.update(roots=True, roots_from=roots_from, body_signal_dim=signal_dim, body_io_channels=io_channels,
                      copied=["inner." + k for k in report["copied"]], grown=["inner." + k for k in report["grown"]],
                      fresh=["inner." + k for k in report["fresh"]])
        if roots_from:
            src = torch.load(roots_from, map_location="cpu", weights_only=False)
            if not src.get("grow", {}).get("roots"):
                raise ValueError("%s has no roots to copy" % roots_from)
            sd, s, o = inner.state_dict(), old_cfg["signal_dim"], old_cfg["io_channels"]
            sd["passive.encoder.0.weight"][:, s:] = src["core"]["inner.passive.encoder.0.weight"][:, s:]
            sd["router.base_map"][o:] = src["core"]["inner.router.base_map"][o:]
            inner.load_state_dict(sd)
        return RootedCore(inner, old_cfg["signal_dim"], old_cfg["io_channels"]), report
    if checkpoint:
        old_cfg, old_sd = load_checkpoint(checkpoint)
    else:
        old_cfg, old_sd = dict(DEFAULT_CORE), None
    cfg = dict(old_cfg)
    cfg.update(signal_dim=signal_dim, io_channels=io_channels)
    core = ModelCore(**cfg)
    report = dict(checkpoint=checkpoint, old_signal_dim=old_cfg["signal_dim"], old_io_channels=old_cfg["io_channels"],
                  signal_dim=signal_dim, io_channels=io_channels, copied=[], grown=[], fresh=[])
    if old_sd is None:
        report["fresh"] = list(core.state_dict())
        return core, report
    new_sd = core.state_dict()
    with torch.no_grad():
        for k, v in new_sd.items():
            if k not in old_sd:
                report["fresh"].append(k)
                continue
            o = old_sd[k]
            if o.shape == v.shape:
                v.copy_(o)
                report["copied"].append(k)
            elif k == "passive.encoder.0.weight":
                v.zero_()
                n = min(o.shape[1], v.shape[1])
                v[:, :n] = o[:, :n]
                report["grown"].append(k)
            elif k == "router.base_map":
                n = min(o.shape[0], v.shape[0])
                v[:n] = o[:n]
                report["grown"].append(k)
            else:
                report["fresh"].append(k)
    core.load_state_dict(new_sd)
    return core, report


class PlainMLP(nn.Module):
    """The comparison baseline (issue #51): a plain feed-forward network with the core's interface, so the same
    actor, trainer, cloning and gate run it. Two ELU hidden layers; `hidden` 168 gives 51,440 parameters for the
    tendon humanoid's 104 inputs and 32 outputs, against the grown core's 51,484. No memory: the state tensor is
    passed through unchanged. The output layer starts near zero, so the sigmoid sits at 0.5 and the action mean
    at 0 (the standing pose), as the core's outputs do."""

    arch = "mlp"

    def __init__(self, signal_dim: int, io_channels: int, hidden: int = 168):
        super().__init__()
        self.signal_dim, self.io_channels, self.hidden = signal_dim, io_channels, hidden
        self.memory_slots, self.memory_dim = 1, 1
        self.net = nn.Sequential(nn.Linear(signal_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, io_channels))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def init_memory(self, batch_size: int, memory_slots: int, memory_dim: int, device) -> torch.Tensor:
        return torch.zeros(batch_size, memory_slots, memory_dim, device=device)

    def forward(self, signal: torch.Tensor, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        return dict(io=torch.sigmoid(self.net(signal)), memory=memory)


class PlainGRU(nn.Module):
    """A plain recurrent baseline (the core has memory, the MLP none): input layer, a GRU cell, output layer. `hidden`
    80 gives 49,872 parameters for 104 inputs and 32 outputs (the MLP 51,440, the grown core 51,484). The memory
    tensor is (batch, 1, hidden)."""

    arch = "gru"

    def __init__(self, signal_dim: int, io_channels: int, hidden: int = 80):
        super().__init__()
        self.signal_dim, self.io_channels, self.hidden = signal_dim, io_channels, hidden
        self.memory_slots, self.memory_dim = 1, hidden
        self.inp = nn.Sequential(nn.Linear(signal_dim, hidden), nn.ELU())
        self.cell = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, io_channels)
        with torch.no_grad():
            self.out.weight.mul_(0.01)
            self.out.bias.zero_()

    def init_memory(self, batch_size: int, memory_slots: int, memory_dim: int, device) -> torch.Tensor:
        return torch.zeros(batch_size, 1, self.hidden, device=device)

    def forward(self, signal: torch.Tensor, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.cell(self.inp(signal), memory[:, 0])
        return dict(io=torch.sigmoid(self.out(h)), memory=h[:, None, :])


def build_policy_net(saved: dict) -> nn.Module:
    """The actor's network from a policy saved by rl/ppo.py: the plain baseline or the model core."""
    if saved.get("arch") == "gru":
        c = saved["core_config"]
        net = PlainGRU(c["signal_dim"], c["io_channels"], c["hidden"])
        net.load_state_dict(saved["core"])
        return net
    if saved.get("arch") == "mlp":
        c = saved["core_config"]
        net = PlainMLP(c["signal_dim"], c["io_channels"], c["hidden"])
        net.load_state_dict(saved["core"])
        return net
    from ai_embedded_dynamic_diversity.models.core import ModelCore
    g = saved["grow"]
    cfg = dict(saved["trunk_config"]) if saved.get("trunk_config") else (
        dict(load_checkpoint(g["checkpoint"])[0]) if g.get("checkpoint") else dict(DEFAULT_CORE))
    cfg.update(signal_dim=g["signal_dim"], io_channels=g["io_channels"])
    core = ModelCore(**cfg)
    if g.get("roots"):
        core = RootedCore(core, g["old_signal_dim"], g["old_io_channels"])
    core.load_state_dict(saved["core"])
    return core


class CoreActor(nn.Module):
    """Gaussian policy whose mean is the model core's output, with the core's memory as recurrent state."""

    def __init__(self, core: ModelCore, init_log_std: float = -1.5):
        super().__init__()
        self.core = core
        self.log_std = nn.Parameter(torch.full((core.io_channels,), float(init_log_std)))

    def initial_state(self, batch: int, device) -> torch.Tensor:
        return self.core.init_memory(batch, self.core.memory_slots, self.core.memory_dim, device)

    def forward(self, obs: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.core(obs, memory)
        mean = 2.0 * out["io"] - 1.0
        return mean, out["memory"]

    def dist(self, mean: torch.Tensor, scale: torch.Tensor | None = None) -> torch.distributions.Normal:
        """`scale` (batch,): a fixed per-environment multiplier on the std (the stand-in's quiet envs explore
        less, so drums can lock there); the log-probabilities use it, so PPO stays on-policy."""
        std = self.log_std.exp().expand_as(mean)
        if scale is not None:
            std = std * scale[..., None]
        return torch.distributions.Normal(mean, std)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.core.parameters())


class Critic(nn.Module):
    """Value function on the privileged observation (training only; never deployed)."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RunningNorm(nn.Module):
    """Running mean and variance normaliser (saved with the policy, so deployment sees the same scaling)."""

    def __init__(self, dim: int, clip: float = 5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta.pow(2) * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


def gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    return (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum()
