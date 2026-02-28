from __future__ import annotations

import torch

class IOAdapter:
    """Hardware-agnostic runtime I/O abstraction for ModelCore deployment."""
    
    def __init__(self, signal_dim: int):
        self.signal_dim = signal_dim

    def normalize(self, raw_input: torch.Tensor) -> torch.Tensor:
        """Normalizes arbitrary hardware input to [0, 1] for ModelCore."""
        # Anonymous-channel robust normalization: 
        # use per-sample min/max span or moving window statistics.
        # Here we use a simple min-max per sample for functional baseline.
        if raw_input.dim() == 1:
            raw_input = raw_input.unsqueeze(0)
            
        min_v = raw_input.min(dim=1, keepdim=True).values
        max_v = raw_input.max(dim=1, keepdim=True).values
        span = (max_v - min_v).clamp_min(1e-6)
        
        normalized = (raw_input - min_v) / span
        return normalized

    def denormalize(self, model_output: torch.Tensor, target_range: tuple[float, float] = (0.0, 1.0)) -> torch.Tensor:
        """Maps ModelCore output [0, 1] to specific hardware ranges."""
        low, high = target_range
        mapped = low + model_output * (high - low)
        
        if mapped.dim() > 1 and mapped.size(0) == 1:
            mapped = mapped.squeeze(0)
            
        return mapped

class I2CTransport:
    """Low-level I2C transport stub for physical hardware access."""
    def __init__(self, bus_id: int = 1):
        self.bus_id = bus_id
        # Placeholder for smbus or similar library
        self.bus = None

    def read_block(self, address: int, length: int) -> list[int]:
        """Reads a block of bytes from an I2C device."""
        return [0] * length

    def write_block(self, address: int, data: list[int]) -> None:
        """Writes a block of bytes to an I2C device."""
        pass

class I2CAdapter(IOAdapter):
    """Bridges ModelCore with physical I2C sensors and actuators."""
    def __init__(self, signal_dim: int, transport: I2CTransport):
        super().__init__(signal_dim)
        self.transport = transport

    def read_sensors(self, device_address: int) -> torch.Tensor:
        """Reads raw I2C data and returns a normalized tensor."""
        raw = self.transport.read_block(device_address, self.signal_dim * 2) # Assume 16-bit
        # Convert bytes to floats, then normalize
        raw_tensor = torch.tensor(raw, dtype=torch.float32).view(-1, 2).mean(dim=1)
        return self.normalize(raw_tensor)

    def write_actuators(self, device_address: int, model_output: torch.Tensor):
        """Maps model output to hardware ranges and writes via I2C."""
        raw_output = self.denormalize(model_output, target_range=(0, 255))
        data = raw_output.byte().tolist()
        self.transport.write_block(device_address, data)
