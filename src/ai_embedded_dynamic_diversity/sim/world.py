from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EnvironmentControls:
    wind: torch.Tensor
    light_position: torch.Tensor
    light_intensity: torch.Tensor
    force_position: torch.Tensor
    force_vector: torch.Tensor
    force_strength: torch.Tensor
    force_active: torch.Tensor
    move_object_delta: torch.Tensor


class WorldState:
    def __init__(
        self,
        life: torch.Tensor,
        resources: torch.Tensor,
        stress: torch.Tensor,
        object_pos: torch.Tensor,
        object_vel: torch.Tensor,
    ):
        self.life = life
        self.resources = resources
        self.stress = stress
        self.object_pos = object_pos
        self.object_vel = object_vel


class DynamicDiversityWorld:
    """3D cellular world with resources, stressors, and local flow constraints."""

    def __init__(self, x: int, y: int, z: int, resource_channels: int, decay: float = 0.03, device: str = "cpu"):
        self.x = x
        self.y = y
        self.z = z
        self.resource_channels = resource_channels
        self.decay = decay
        self.device = torch.device(device)
        self.kernel = self._kernel3d().to(self.device)
        self.coord_grid = self._coord_grid3d().to(self.device)

    def _kernel3d(self) -> torch.Tensor:
        k = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)
        k[:, :, 1, 1, 1] = 0.0
        return k

    def _coord_grid3d(self) -> torch.Tensor:
        z = torch.linspace(-1.0, 1.0, self.z)
        y = torch.linspace(-1.0, 1.0, self.y)
        x = torch.linspace(-1.0, 1.0, self.x)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([xx, yy, zz], dim=0).unsqueeze(0)

    def default_controls(self, batch_size: int) -> EnvironmentControls:
        return EnvironmentControls(
            wind=torch.zeros(batch_size, 3, device=self.device),
            light_position=torch.zeros(batch_size, 3, device=self.device),
            light_intensity=torch.full((batch_size, 1), 0.3, device=self.device),
            force_position=torch.zeros(batch_size, 3, device=self.device),
            force_vector=torch.zeros(batch_size, 3, device=self.device),
            force_strength=torch.zeros(batch_size, 1, device=self.device),
            force_active=torch.zeros(batch_size, 1, device=self.device),
            move_object_delta=torch.zeros(batch_size, 3, device=self.device),
        )

    def random_controls(self, batch_size: int, volatility: float, step_index: int = 0) -> EnvironmentControls:
        controls = self.default_controls(batch_size)
        if volatility <= 0.0:
            return controls

        vol = float(volatility)
        controls.wind = (torch.rand(batch_size, 3, device=self.device) * 2.0 - 1.0) * (0.6 * vol)
        controls.light_position = torch.rand(batch_size, 3, device=self.device) * 2.0 - 1.0
        controls.light_intensity = torch.full((batch_size, 1), 0.25 + 0.65 * vol, device=self.device)

        if step_index % max(2, int(8 / max(vol, 1e-3))) == 0:
            controls.force_active = (torch.rand(batch_size, 1, device=self.device) > 0.5).float()
            controls.force_strength = torch.rand(batch_size, 1, device=self.device) * (0.4 + 0.8 * vol)
            controls.force_vector = torch.randn(batch_size, 3, device=self.device) * vol
            controls.force_position = torch.rand(batch_size, 3, device=self.device) * 2.0 - 1.0
        return controls

    def init(self, batch_size: int) -> WorldState:
        life = (torch.rand(batch_size, 1, self.z, self.y, self.x, device=self.device) > 0.8).float()
        resources = torch.rand(batch_size, self.resource_channels, self.z, self.y, self.x, device=self.device)
        stress = torch.zeros(batch_size, 1, self.z, self.y, self.x, device=self.device)
        object_pos = torch.zeros(batch_size, 3, device=self.device)
        object_vel = torch.zeros(batch_size, 3, device=self.device)
        return WorldState(life, resources, stress, object_pos, object_vel)

    def _light_field(self, controls: EnvironmentControls) -> torch.Tensor:
        pos = controls.light_position.view(-1, 3, 1, 1, 1)
        dist2 = torch.sum((self.coord_grid - pos) ** 2, dim=1, keepdim=True)
        sigma2 = 0.25
        light = controls.light_intensity.view(-1, 1, 1, 1, 1) * torch.exp(-dist2 / sigma2)
        return light

    def _force_field(self, controls: EnvironmentControls) -> torch.Tensor:
        pos = controls.force_position.view(-1, 3, 1, 1, 1)
        dist2 = torch.sum((self.coord_grid - pos) ** 2, dim=1, keepdim=True)
        magnitude = torch.norm(controls.force_vector, dim=1, keepdim=True).view(-1, 1, 1, 1, 1)
        strength = controls.force_strength.view(-1, 1, 1, 1, 1) * controls.force_active.view(-1, 1, 1, 1, 1)
        return strength * magnitude * torch.exp(-dist2 / 0.12)

    def _apply_wind_flow(self, resources: torch.Tensor, wind: torch.Tensor) -> torch.Tensor:
        flowed = resources
        shifts = torch.round(wind).to(torch.int64)
        # Axis order in tensor is z, y, x.
        for b in range(resources.size(0)):
            dx = int(shifts[b, 0].item())
            dy = int(shifts[b, 1].item())
            dz = int(shifts[b, 2].item())
            flowed[b : b + 1] = torch.roll(resources[b : b + 1], shifts=(dz, dy, dx), dims=(-3, -2, -1))
        return flowed

    def step(self, state: WorldState, action_field: torch.Tensor, controls: EnvironmentControls | None = None) -> WorldState:
        if controls is None:
            controls = self.default_controls(action_field.size(0))

        neighbors = torch.conv3d(state.life, self.kernel, padding=1)
        survive = ((neighbors >= 5) & (neighbors <= 7)).float() * state.life
        born = ((neighbors == 6).float()) * (1.0 - state.life)

        action_field = action_field.view(action_field.size(0), 1, self.z, self.y, self.x)
        light_boost = self._light_field(controls)
        force_impact = self._force_field(controls)
        adaptive_bonus = torch.sigmoid(action_field + light_boost - 0.25 * force_impact) * (state.resources[:, :1] > 0.2).float()
        new_life = torch.clamp(survive + born + 0.25 * adaptive_bonus - 0.1 * force_impact, 0.0, 1.0)

        resource_use = 0.05 * new_life
        flow = self._apply_wind_flow(state.resources, controls.wind) * 0.02
        new_resources = torch.clamp(state.resources + flow + 0.03 * light_boost - resource_use - self.decay * state.resources, 0.0, 1.0)

        pressure = (neighbors / 26.0).clamp(0.0, 1.0)
        scarcity = 1.0 - new_resources[:, :1]
        wind_pressure = torch.norm(controls.wind, dim=1, keepdim=True).view(-1, 1, 1, 1, 1)
        new_stress = 0.55 * pressure + 0.35 * scarcity + 0.10 * wind_pressure + 0.2 * force_impact

        object_vel = 0.88 * state.object_vel + controls.force_vector * controls.force_strength * controls.force_active
        object_pos = state.object_pos + object_vel + controls.move_object_delta
        object_pos = object_pos.clamp(-1.0, 1.0)
        return WorldState(new_life, new_resources, new_stress.clamp(0.0, 1.0), object_pos, object_vel)

    def encode_observation(self, state: WorldState, signal_dim: int) -> torch.Tensor:
        # Aggregate anonymous signal channels; do not encode modality identity.
        pooled = torch.cat(
            [
                state.life.mean(dim=(2, 3, 4)),
                state.resources.mean(dim=(2, 3, 4)),
                state.stress.mean(dim=(2, 3, 4)),
                state.object_pos,
                state.object_vel,
            ],
            dim=1,
        )
        if pooled.size(1) >= signal_dim:
            return pooled[:, :signal_dim]
        pad = torch.zeros(pooled.size(0), signal_dim - pooled.size(1), device=pooled.device)
        return torch.cat([pooled, pad], dim=1)
