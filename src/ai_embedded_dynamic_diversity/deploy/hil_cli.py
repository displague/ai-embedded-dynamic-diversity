from __future__ import annotations

import json
import socket

import torch
import typer

from ai_embedded_dynamic_diversity.config import model_config_for_profile
from ai_embedded_dynamic_diversity.models import ModelCore

app = typer.Typer(add_completion=False)


def _load_model(weights: str | None, profile: str, device: torch.device) -> tuple[ModelCore, object]:
    cfg = model_config_for_profile(profile)
    model = ModelCore(**cfg.__dict__).to(device)
    if weights:
        ckpt = torch.load(weights, map_location=device)
        cfg = type(cfg)(**ckpt["model_config"])
        model = ModelCore(**cfg.__dict__).to(device)
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@app.command()
def udp_bridge(
    profile: str = "pi5",
    weights: str = "",
    listen_host: str = "127.0.0.1",
    listen_port: int = 45454,
    send_host: str = "127.0.0.1",
    send_port: int = 45455,
    device: str = "cpu",
) -> None:
    """Minimal hardware-in-the-loop adapter over UDP JSON payloads.

    Input packet: {"signal": [..float..], "remap": [..float..]}
    Output packet: {"io": [..float..], "readiness": [..float..]}
    """
    dev = torch.device(device)
    model, cfg = _load_model(weights if weights else None, profile, dev)
    memory = model.init_memory(1, cfg.memory_slots, cfg.memory_dim, dev)

    sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_in.bind((listen_host, listen_port))

    print({"hil_adapter": "udp", "listen": f"{listen_host}:{listen_port}", "send": f"{send_host}:{send_port}"})
    try:
        while True:
            data, _ = sock_in.recvfrom(65535)
            payload = json.loads(data.decode("utf-8"))

            signal = torch.tensor(payload.get("signal", []), dtype=torch.float32, device=dev).view(1, -1)
            if signal.size(1) < cfg.signal_dim:
                signal = torch.nn.functional.pad(signal, (0, cfg.signal_dim - signal.size(1)))
            signal = signal[:, : cfg.signal_dim]

            remap = torch.tensor(payload.get("remap", [0.0] * cfg.max_remap_groups), dtype=torch.float32, device=dev).view(1, -1)
            if remap.size(1) < cfg.max_remap_groups:
                remap = torch.nn.functional.pad(remap, (0, cfg.max_remap_groups - remap.size(1)))
            remap = remap[:, : cfg.max_remap_groups]

            with torch.no_grad():
                out = model(signal, memory, remap)
                memory = out["memory"]
                response = {
                    "io": out["io"][0].detach().cpu().tolist(),
                    "readiness": out["readiness"][0].detach().cpu().tolist(),
                }
            sock_out.sendto(json.dumps(response).encode("utf-8"), (send_host, send_port))
    finally:
        sock_in.close()
        sock_out.close()


if __name__ == "__main__":
    app()
