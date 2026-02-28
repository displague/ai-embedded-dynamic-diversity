from __future__ import annotations

import json
import time
from pathlib import Path
import torch

class TelemetryCollector:
    """Captures deployment inference data for curriculum feedback loops."""
    
    def __init__(self, log_dir: str = "artifacts/telemetry"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def capture(self, signal: torch.Tensor, output: torch.Tensor, metadata: dict | None = None) -> None:
        """Logs a single inference step to a JSON file."""
        self._counter += 1
        entry = {
            "timestamp": time.time(),
            "signal": signal.detach().cpu().numpy().tolist()[0],
            "output": output.detach().cpu().numpy().tolist()[0],
            "metadata": metadata or {}
        }
        
        # Use high-precision filename with counter to avoid collisions
        filename = f"telemetry_{time.time_ns()}_{self._counter:06d}.json"
        log_path = self.log_dir / filename
        log_path.write_text(json.dumps(entry), encoding="utf-8")

    def get_retraining_data(self) -> list[dict]:
        """Aggregates all telemetry logs into a list for retraining."""
        data = []
        for log_file in sorted(self.log_dir.glob("*.json")):
            try:
                data.append(json.loads(log_file.read_text(encoding="utf-8")))
            except Exception:
                continue
        return data

    def clear(self):
        """Removes all collected telemetry logs."""
        for log_file in self.log_dir.glob("*.json"):
            log_file.unlink()
