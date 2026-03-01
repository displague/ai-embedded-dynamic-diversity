from __future__ import annotations

import time
import torch
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.deploy.io_adapter import IOAdapter
from ai_embedded_dynamic_diversity.config import model_config_for_profile

def run_pi_hardware_simulation(weights_path: str, profile: str = "pi5"):
    # Load model
    ckpt = torch.load(weights_path, map_location="cpu")
    cfg = model_config_for_profile(profile)
    model = ModelCore(**cfg.__dict__)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    
    # Setup IOAdapter
    adapter = IOAdapter(signal_dim=cfg.signal_dim)
    
    # Mock hardware sensor data (e.g. 12-bit ADC values 0-4095)
    raw_sensors = torch.rand(cfg.signal_dim) * 4095.0
    
    # Initialize memory
    memory = model.init_memory(1, cfg.memory_slots, cfg.memory_dim, "cpu")
    
    print(f"Starting Pi 5 hardware simulation for {weights_path}...")
    
    steps = 100
    latencies = []
    
    with torch.no_grad():
        for i in range(steps):
            t0 = time.perf_counter()
            
            # 1. Normalize
            normalized_obs = adapter.normalize(raw_sensors)
            
            # 2. Inference
            out = model(normalized_obs, memory)
            memory = out["memory"]
            
            # 3. Denormalize (e.g. to PWM 0-100)
            actuators = adapter.denormalize(out["io"], target_range=(0, 100))
            
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            
    avg_latency = sum(latencies) / steps
    p95_latency = sorted(latencies)[int(0.95 * steps)]
    
    print(f"Simulation Complete.")
    print(f"Avg Latency (IO + Inference): {avg_latency:.4f} ms")
    print(f"P95 Latency: {p95_latency:.4f} ms")
    print(f"Sample Actuator Output: {actuators[:5].tolist()}")

if __name__ == "__main__":
    run_pi_hardware_simulation("artifacts/model-core-champion-v06.pt")
