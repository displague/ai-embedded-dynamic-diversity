"""Recurrent PPO for the model core on the stand-in (rsl_rl's recipe, written out so the core's memory can be
the recurrent state).

Each iteration collects `horizon` steps from every environment, keeping the core's memory at the start of the
window; the update replays each environment's window from that memory (truncated backpropagation through
time), resetting the memory where an episode ended. Advantages are GAE; the policy loss is the clipped ratio;
the value function is a separate network on the privileged observation; the learning rate adapts to keep the
KL divergence near a target, as rsl_rl does.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ai_embedded_dynamic_diversity.rl.actor import CoreActor, Critic, RunningNorm, gaussian_entropy, grow_core
from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, StandinEnv


@dataclass
class PPOConfig:
    iterations: int = 300
    horizon: int = 24
    epochs: int = 4
    minibatches: int = 2
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 1e-3
    desired_kl: float = 0.01
    entropy: float = 0.003
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    init_log_std: float = -2.0
    seed: int = 0
    device: str = "cuda"
    save_every: int = 50


def train(env_cfg: EnvConfig, ppo: PPOConfig, init: str | None, out_dir: str, label: str,
          teacher: str | None = None, bc_epochs: int = 30) -> dict:
    """Train a core actor; `init` is a checkpoint path to grow from, or None for random initialisation;
    `teacher` is a demonstrations file (displague/robot tools/record_teacher.py) to clone before PPO.
    Writes out_dir/<label>.log.json (one record per iteration) and out_dir/<label>.pt; returns the summary."""
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(ppo.device if torch.cuda.is_available() or ppo.device == "cpu" else "cpu")
    torch.manual_seed(ppo.seed)
    env = StandinEnv(env_cfg)
    core, grow_report = grow_core(env.obs_dim, env.act_dim, init, seed=ppo.seed)
    actor = CoreActor(core, ppo.init_log_std).to(dev)
    critic = Critic(env.priv_dim).to(dev)
    obs_norm, priv_norm = RunningNorm(env.obs_dim).to(dev), RunningNorm(env.priv_dim).to(dev)
    bc_log = None
    if teacher:
        from ai_embedded_dynamic_diversity.rl.bc import clone, load_teacher
        bc_log = clone(actor, obs_norm, load_teacher(teacher, env.half), dev, epochs=bc_epochs, seed=ppo.seed)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=ppo.lr)
    lr = ppo.lr
    N, T, A = env.N, ppo.horizon, env.act_dim
    env.reset()
    obs_np, priv_np = env.observe()
    memory = actor.initial_state(N, dev)
    log, t_start, total_steps = [], time.time(), 0
    ep_success, ep_damage, ep_count, ep_wmini = 0, 0, 0, 0
    for it in range(ppo.iterations):
        t0 = time.time()
        buf_obs = torch.zeros(T, N, env.obs_dim, device=dev)
        buf_priv = torch.zeros(T, N, env.priv_dim, device=dev)
        buf_act = torch.zeros(T, N, A, device=dev)
        buf_logp = torch.zeros(T, N, device=dev)
        buf_mean = torch.zeros(T, N, A, device=dev)
        buf_val = torch.zeros(T, N, device=dev)
        buf_rew = torch.zeros(T, N, device=dev)
        buf_done = torch.zeros(T, N, device=dev)
        mem0 = memory.detach().clone()
        it_success = it_damage = it_eps = it_wmini = 0
        stand_frac = []
        with torch.no_grad():
            for t in range(T):
                o = torch.as_tensor(obs_np, device=dev)
                p = torch.as_tensor(priv_np, device=dev)
                obs_norm.update(o)
                priv_norm.update(p)
                on, pn = obs_norm(o), priv_norm(p)
                mean, memory = actor(on, memory)
                d = actor.dist(mean)
                a = d.sample()
                buf_obs[t], buf_priv[t], buf_act[t] = on, pn, a
                buf_logp[t] = d.log_prob(a).sum(-1)
                buf_mean[t] = mean
                buf_val[t] = critic(pn)
                obs_np, priv_np, r, done, info = env.step(a.clamp(-1, 1).cpu().numpy())
                buf_rew[t] = torch.as_tensor(r, device=dev)
                dn = torch.as_tensor(done, device=dev, dtype=torch.float32)
                buf_done[t] = dn
                memory = memory * (1.0 - dn)[:, None, None]
                stand_frac.append(float(info.standing.mean()))
                if done.any():
                    fin = np.nonzero(done)[0]
                    it_eps += len(fin)
                    it_success += int(info.extras["ever_stood"][fin].sum())
                    it_damage += int(info.damage[fin].sum())
            last_val = critic(priv_norm(torch.as_tensor(priv_np, device=dev)))
        # GAE
        adv = torch.zeros(T, N, device=dev)
        gae = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nxt = last_val if t == T - 1 else buf_val[t + 1]
            nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + ppo.gamma * nxt * nonterm - buf_val[t]
            gae = delta + ppo.gamma * ppo.lam * nonterm * gae
            adv[t] = gae
        ret = adv + buf_val
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        # update: minibatches of environments, each replayed through its window from the stored memory
        envs = torch.randperm(N, device=dev)
        mb_size = N // ppo.minibatches
        kls, pls, vls = [], [], []
        std_old = actor.log_std.detach().exp()
        for _ in range(ppo.epochs):
            envs = envs[torch.randperm(N, device=dev)]
            for b in range(ppo.minibatches):
                idx = envs[b * mb_size:(b + 1) * mb_size]
                mem = mem0[idx]
                means = []
                for t in range(T):
                    if t > 0:
                        mem = mem * (1.0 - buf_done[t - 1, idx])[:, None, None]
                    m_t, mem = actor(buf_obs[t, idx], mem)
                    means.append(m_t)
                mean = torch.stack(means)
                d = actor.dist(mean)
                logp = d.log_prob(buf_act[:, idx]).sum(-1)
                ratio = torch.exp(logp - buf_logp[:, idx])
                a_mb = adv_n[:, idx]
                pl = -torch.min(ratio * a_mb, ratio.clamp(1 - ppo.clip, 1 + ppo.clip) * a_mb).mean()
                v = critic(buf_priv[:, idx])
                vl = (v - ret[:, idx]).pow(2).mean()
                ent = gaussian_entropy(actor.log_std)
                loss = pl + ppo.value_coef * vl - ppo.entropy * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, ppo.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    # the exact KL between the collecting and the updated Gaussian (rsl_rl's adaptive rule
                    # needs it; the sample estimate hovers near zero and drove the learning rate to its cap)
                    std_new = actor.log_std.exp()
                    kl = (torch.log(std_new / std_old) + (std_old.pow(2) + (buf_mean[:, idx] - mean).pow(2))
                          / (2.0 * std_new.pow(2)) - 0.5).sum(-1).mean().item()
                kls.append(kl)
                pls.append(pl.item())
                vls.append(vl.item())
                if kl > 2.0 * ppo.desired_kl:
                    lr = max(1e-5, lr / 1.5)
                elif 0.0 <= kl < 0.5 * ppo.desired_kl:
                    lr = min(1e-2, lr * 1.5)
                for g in opt.param_groups:
                    g["lr"] = lr
        total_steps += N * T
        ep_success += it_success
        ep_damage += it_damage
        ep_count += it_eps
        rec = dict(iteration=it, steps=total_steps, wall_s=round(time.time() - t_start, 1),
                   fps=round(N * T / (time.time() - t0)), reward=float(buf_rew.mean()),
                   standing_fraction=float(np.mean(stand_frac)), episodes=it_eps,
                   success_rate=(it_success / it_eps) if it_eps else None,
                   episode_s=(N * T / it_eps * env.dt_ctrl) if it_eps else None,
                   damage_rate=(it_damage / it_eps) if it_eps else None,
                   kl=float(np.mean(kls)), lr=lr, policy_loss=float(np.mean(pls)), value_loss=float(np.mean(vls)),
                   std=float(actor.log_std.detach().exp().mean()))
        log.append(rec)
        if it % 5 == 0 or it == ppo.iterations - 1:
            print("[%s] it %4d steps %9d fps %6d r %.3f stand %.3f eps %4d success %s damage %s std %.3f lr %.1e"
                  % (label, it, total_steps, rec["fps"], rec["reward"], rec["standing_fraction"], it_eps,
                     "%.2f" % rec["success_rate"] if it_eps else "-", "%.2f" % rec["damage_rate"] if it_eps else "-",
                     rec["std"], lr), flush=True)
        if (it + 1) % ppo.save_every == 0 or it == ppo.iterations - 1:
            _save(os.path.join(out_dir, label + ".pt"), actor, critic, obs_norm, priv_norm, env_cfg, ppo, grow_report, env)
            with open(os.path.join(out_dir, label + ".log.json"), "w") as f:
                json.dump(dict(label=label, init=init, teacher=teacher, bc=bc_log, grow=grow_report,
                               env=asdict(env_cfg), ppo=asdict(ppo), log=log), f, indent=1)
    env.close()
    return dict(label=label, init=init, iterations=ppo.iterations, steps=total_steps, wall_s=time.time() - t_start,
                final=log[-1], params=actor.num_params())


def _save(path, actor, critic, obs_norm, priv_norm, env_cfg, ppo, grow_report, env):
    torch.save(dict(core=actor.core.state_dict(), log_std=actor.log_std.detach().cpu(),
                    core_config={k: getattr(actor.core, k, None) for k in ("edge_nodes", "io_channels", "hidden_dim",
                                                                            "signal_dim", "memory_slots", "memory_dim",
                                                                            "max_remap_groups")},
                    critic=critic.state_dict(), obs_norm=obs_norm.state_dict(), priv_norm=priv_norm.state_dict(),
                    env=asdict(env_cfg), ppo=asdict(ppo), grow=grow_report, dof_names=env.dof_names), path)
