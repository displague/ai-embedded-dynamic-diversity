from __future__ import annotations

import torch
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.sim.world import EnvironmentControls
from ai_embedded_dynamic_diversity.sim.signaling import SignalingWorld
from ai_embedded_dynamic_diversity.config import model_config_for_profile

def run_shock_test(weights_path: str, profile: str = "pi5"):
    # Load model
    ckpt = torch.load(weights_path, map_location="cpu")
    cfg = model_config_for_profile(profile)
    model = ModelCore(**cfg.__dict__)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    
    # Setup World with High Volatility
    world = SignalingWorld(20, 20, 20, 4)
    batch_size = 8
    state = world.init(batch_size)
    memory = model.init_memory(batch_size, cfg.memory_slots, cfg.memory_dim, "cpu")
    
    print(f"Starting High-Volatility Shock Test for {weights_path}...")
    
    steps = 100
    vitality_logs = []
    stress_logs = []
    
    with torch.no_grad():
        for i in range(steps):
            # Normal volatility for first 20 steps
            # Shock from 20-40
            # Recovery after 40
            volatility = 0.85 if 20 <= i <= 40 else 0.1
            
            obs = world.encode_observation(state, signal_dim=cfg.signal_dim)
            out = model(obs, memory)
            memory = out["memory"]
            
            action_field = out["io"].mean(dim=1, keepdim=True).repeat(1, 20*20*20)
            controls = world.random_controls(batch_size, volatility=volatility, step_index=i)
            state = world.step(state, action_field, controls)

            vitality_logs.append(state.life.mean().item())
            stress_logs.append(state.stress.mean().item())
            
    print(f"Shock Test Complete.")
    print(f"Baseline Vitality: {sum(vitality_logs[:20])/20:.4f}")
    print(f"Shock Vitality (min): {min(vitality_logs[20:40]):.4f}")
    print(f"Recovery Vitality (final): {vitality_logs[-1]:.4f}")
    print(f"Max Stress: {max(stress_logs):.4f}")
    
    if vitality_logs[-1] > 0.8 * vitality_logs[19]:
        print("Conclusion: Model demonstrated robust recovery from high-volatility shocks.")
    else:
        print("Conclusion: Model failed to recover fully from extreme environmental volatility.")

if __name__ == "__main__":
    run_shock_test("artifacts/model-core-champion-v08.pt")
