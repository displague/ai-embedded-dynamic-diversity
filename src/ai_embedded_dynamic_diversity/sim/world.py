from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from ai_embedded_dynamic_diversity.config import HazardZoneConfig


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
        occlusion_mask: torch.Tensor,
        phys_pos: torch.Tensor,
        phys_vel: torch.Tensor,
        phys_height: torch.Tensor,
    ):
        self.life = life
        self.resources = resources
        self.stress = stress
        self.object_pos = object_pos
        self.object_vel = object_vel
        self.occlusion_mask = occlusion_mask   # [B, 1, Z, Y, X] binary, permanent
        self.phys_pos = phys_pos               # [B, N, 3] normalised [-1,1]
        self.phys_vel = phys_vel               # [B, N, 3]
        self.phys_height = phys_height         # [B, N, 1] stack height (≥1.0)


class DynamicDiversityWorld:
    """3D toroidal cellular world with occlusion, physics objects, and hazard zones."""

    def __init__(
        self,
        x: int,
        y: int,
        z: int,
        resource_channels: int,
        decay: float = 0.03,
        device: str = "cpu",
        actuation_delay_steps: int = 0,
        actuation_noise_std: float = 0.0,
        sensor_latency_steps: int = 0,
        sensor_dropout_burst_prob: float = 0.0,
        surface_friction_scale: float = 1.0,
        disturbance_correlation_horizon: int = 0,
        num_occlusion_objects: int = 0,
        occlusion_seed: int = 0,
        num_physics_objects: int = 0,
        phys_mass: float = 1.0,
        phys_friction: float = 0.85,
        hazard_zones: List[HazardZoneConfig] | None = None,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.resource_channels = resource_channels
        self.decay = decay
        self.device = torch.device(device)
        self.kernel = self._kernel3d().to(self.device)
        self.coord_grid = self._coord_grid3d().to(self.device)
        self.actuation_delay_steps = max(0, int(actuation_delay_steps))
        self.actuation_noise_std = max(0.0, float(actuation_noise_std))
        self.sensor_latency_steps = max(0, int(sensor_latency_steps))
        self.sensor_dropout_burst_prob = max(0.0, float(sensor_dropout_burst_prob))
        self.surface_friction_scale = max(0.0, float(surface_friction_scale))
        self.disturbance_correlation_horizon = max(0, int(disturbance_correlation_horizon))
        self.num_occlusion_objects = max(0, int(num_occlusion_objects))
        self.occlusion_seed = int(occlusion_seed)
        self.num_physics_objects = max(0, int(num_physics_objects))
        self.phys_mass = max(1e-3, float(phys_mass))
        self.phys_friction = max(0.0, min(1.0, float(phys_friction)))
        self.hazard_zones: List[HazardZoneConfig] = hazard_zones or []
        self._action_buffer: list[torch.Tensor] = []
        self._obs_buffer: list[torch.Tensor] = []
        self._dropout_burst_steps_remaining = 0
        self._prev_controls: EnvironmentControls | None = None
        self._step_index: int = 0
        # Pre-build hazard masks (spatial footprints, shape [1,Z,Y,X])
        self._hazard_masks: list[torch.Tensor] = []
        for hz in self.hazard_zones:
            self._hazard_masks.append(self._build_hazard_mask(hz).to(self.device))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kernel3d(self) -> torch.Tensor:
        k = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)
        k[:, :, 1, 1, 1] = 0.0
        return k

    def _coord_grid3d(self) -> torch.Tensor:
        z = torch.linspace(-1.0, 1.0, self.z)
        y = torch.linspace(-1.0, 1.0, self.y)
        x = torch.linspace(-1.0, 1.0, self.x)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([xx, yy, zz], dim=0).unsqueeze(0)  # [1,3,Z,Y,X]

    def _build_hazard_mask(self, hz: HazardZoneConfig) -> torch.Tensor:
        """Build a binary [1,Z,Y,X] mask for a hazard zone from fractional bounds."""
        xc = int(hz.cx * self.x)
        yc = int(hz.cy * self.y)
        zc = int(hz.cz * self.z)
        rx = max(1, int(hz.rx * self.x))
        ry = max(1, int(hz.ry * self.y))
        rz = max(1, int(hz.rz * self.z))
        mask = torch.zeros(1, self.z, self.y, self.x)
        z0, z1 = max(0, zc - rz), min(self.z, zc + rz)
        y0, y1 = max(0, yc - ry), min(self.y, yc + ry)
        x0, x1 = max(0, xc - rx), min(self.x, xc + rx)
        mask[0, z0:z1, y0:y1, x0:x1] = 1.0
        return mask

    def _place_t_shapes(self, batch_size: int) -> torch.Tensor:
        """Return binary occlusion mask [B,1,Z,Y,X] with T-shaped occlusion objects."""
        mask = torch.zeros(batch_size, 1, self.z, self.y, self.x, device=self.device)
        if self.num_occlusion_objects == 0:
            return mask
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.occlusion_seed)
        # Arm thickness: fixed 1 cell in Z, ~6% of Y/X dimensions
        arm_thick_y = max(1, int(0.06 * self.y))
        arm_thick_x = max(1, int(0.06 * self.x))
        # Arm lengths: ~30% of Y/X
        arm_len_y = max(2, int(0.30 * self.y))
        arm_len_x = max(2, int(0.30 * self.x))
        for _ in range(self.num_occlusion_objects):
            cx = int(torch.randint(arm_len_x, self.x - arm_len_x, (1,), generator=gen).item())
            cy = int(torch.randint(arm_len_y, self.y - arm_len_y, (1,), generator=gen).item())
            cz = int(torch.randint(0, max(1, self.z // 2), (1,), generator=gen).item())
            orient = int(torch.randint(0, 2, (1,), generator=gen).item())  # 0=horizontal stem, 1=vertical stem
            # Stem (long bar)
            if orient == 0:
                # horizontal stem: extends in X
                sx0, sx1 = max(0, cx - arm_len_x), min(self.x, cx + arm_len_x)
                sy0, sy1 = max(0, cy - arm_thick_y), min(self.y, cy + arm_thick_y)
            else:
                # vertical stem: extends in Y
                sx0, sx1 = max(0, cx - arm_thick_x), min(self.x, cx + arm_thick_x)
                sy0, sy1 = max(0, cy - arm_len_y), min(self.y, cy + arm_len_y)
            sz0, sz1 = cz, min(self.z, cz + max(1, self.z // 5))
            mask[:, 0, sz0:sz1, sy0:sy1, sx0:sx1] = 1.0
            # Crossbar (perpendicular arm at centre)
            if orient == 0:
                bx0, bx1 = max(0, cx - arm_thick_x), min(self.x, cx + arm_thick_x)
                by0, by1 = max(0, cy - arm_len_y), min(self.y, cy + arm_len_y)
            else:
                bx0, bx1 = max(0, cx - arm_len_x), min(self.x, cx + arm_len_x)
                by0, by1 = max(0, cy - arm_thick_y), min(self.y, cy + arm_thick_y)
            mask[:, 0, sz0:sz1, by0:by1, bx0:bx1] = 1.0
        return mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        if self._prev_controls is not None and self.disturbance_correlation_horizon > 0:
            alpha = max(0.05, min(0.95, 1.0 / float(self.disturbance_correlation_horizon)))
            controls.wind = (1.0 - alpha) * self._prev_controls.wind + alpha * controls.wind
            controls.force_vector = (1.0 - alpha) * self._prev_controls.force_vector + alpha * controls.force_vector
            controls.force_strength = (1.0 - alpha) * self._prev_controls.force_strength + alpha * controls.force_strength
        self._prev_controls = EnvironmentControls(
            wind=controls.wind.clone(),
            light_position=controls.light_position.clone(),
            light_intensity=controls.light_intensity.clone(),
            force_position=controls.force_position.clone(),
            force_vector=controls.force_vector.clone(),
            force_strength=controls.force_strength.clone(),
            force_active=controls.force_active.clone(),
            move_object_delta=controls.move_object_delta.clone(),
        )
        return controls

    def init(self, batch_size: int) -> WorldState:
        life = (torch.rand(batch_size, 1, self.z, self.y, self.x, device=self.device) > 0.8).float()
        resources = torch.rand(batch_size, self.resource_channels, self.z, self.y, self.x, device=self.device)
        stress = torch.zeros(batch_size, 1, self.z, self.y, self.x, device=self.device)
        object_pos = torch.zeros(batch_size, 3, device=self.device)
        object_vel = torch.zeros(batch_size, 3, device=self.device)
        occlusion_mask = self._place_t_shapes(batch_size)
        # Physics objects: random positions in [-1,1], unit height
        if self.num_physics_objects > 0:
            phys_pos = torch.rand(batch_size, self.num_physics_objects, 3, device=self.device) * 2.0 - 1.0
            phys_vel = torch.zeros(batch_size, self.num_physics_objects, 3, device=self.device)
            phys_height = torch.ones(batch_size, self.num_physics_objects, 1, device=self.device)
        else:
            phys_pos = torch.zeros(batch_size, 0, 3, device=self.device)
            phys_vel = torch.zeros(batch_size, 0, 3, device=self.device)
            phys_height = torch.zeros(batch_size, 0, 1, device=self.device)
        self._action_buffer = []
        self._obs_buffer = []
        self._dropout_burst_steps_remaining = 0
        self._prev_controls = None
        self._step_index = 0
        return WorldState(life, resources, stress, object_pos, object_vel, occlusion_mask, phys_pos, phys_vel, phys_height)

    # ------------------------------------------------------------------
    # Field computations
    # ------------------------------------------------------------------

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
        for b in range(resources.size(0)):
            dx = int(shifts[b, 0].item())
            dy = int(shifts[b, 1].item())
            dz = int(shifts[b, 2].item())
            # torch.roll already wraps (toroidal), consistent with circular conv
            flowed[b : b + 1] = torch.roll(resources[b : b + 1], shifts=(dz, dy, dx), dims=(-3, -2, -1))
        return flowed

    def _shadow_mask(self, light_field: torch.Tensor, occlusion_mask: torch.Tensor) -> torch.Tensor:
        """Binary mask: 1 where cells are in shadow (low light AND near occlusion)."""
        return ((light_field < 0.05).float() * occlusion_mask).clamp(0.0, 1.0)

    def _compute_hazard_stress(
        self,
        light_field: torch.Tensor,
        occlusion_mask: torch.Tensor,
        wind: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        """Aggregate hazard stress across all configured hazard zones."""
        if not self.hazard_zones:
            return torch.zeros_like(light_field)

        shadow = self._shadow_mask(light_field, occlusion_mask)
        wind_mag = torch.norm(wind, dim=1, keepdim=True).view(-1, 1, 1, 1, 1)
        total = torch.zeros_like(light_field)

        for hz, hz_mask in zip(self.hazard_zones, self._hazard_masks):
            zone = hz_mask.to(light_field.device)  # [1,Z,Y,X]
            if hz.kind == "light_triggered":
                light_in_zone = (light_field * zone).mean(dim=(2, 3, 4), keepdim=True)
                active = (light_in_zone > hz.threshold).float()
                hazard_field = active * zone
                if hz.shadow_safe:
                    hazard_field = hazard_field * (1.0 - shadow)
            elif hz.kind == "airflow":
                active = (wind_mag > hz.threshold).float()
                hazard_field = active * zone
            elif hz.kind == "periodic":
                phase = math.sin(2.0 * math.pi * step_index / max(1, hz.period))
                active = 1.0 if phase > hz.threshold else 0.0
                hazard_field = active * zone
            else:
                hazard_field = torch.zeros_like(light_field)

            total = total + hz.hazard_weight * hazard_field

        return total.clamp(0.0, 1.0)

    def _physics_step(
        self,
        state: WorldState,
        action_field: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update physics object positions and velocities. Returns (phys_pos, phys_vel, phys_height)."""
        if self.num_physics_objects == 0:
            return state.phys_pos, state.phys_vel, state.phys_height

        # Agent "centre" approximated as mean coord_grid weighted by action field amplitude
        # Shape of action_field here: [B, 1, Z, Y, X]
        af_abs = action_field.abs()
        af_sum = af_abs.sum(dim=(2, 3, 4), keepdim=True).clamp(min=1e-6)
        # coord_grid: [1,3,Z,Y,X] → weighted mean → [B,3]
        agent_center = (self.coord_grid * af_abs / af_sum).sum(dim=(2, 3, 4))  # [B,3]

        action_scalar = action_field.mean(dim=(1, 2, 3, 4))  # [B]

        new_pos = state.phys_pos.clone()
        new_vel = state.phys_vel.clone()
        new_height = state.phys_height.clone()

        for i in range(self.num_physics_objects):
            obj_pos = state.phys_pos[:, i, :]   # [B,3]
            obj_vel = state.phys_vel[:, i, :]   # [B,3]

            disp = obj_pos - agent_center         # direction agent→object [B,3]
            dist2 = (disp ** 2).sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B,1]
            prox = torch.exp(-dist2 / 0.08)       # [B,1]

            # Push/pull: sign of action_scalar dot sign of disp determines direction
            # Positive action + positive disp → push away; negative → pull toward
            force_dir = disp / (dist2.sqrt() + 1e-6)  # unit vector [B,3]
            force_mag = (action_scalar.unsqueeze(1) * prox) / self.phys_mass  # [B,1]
            force = force_mag * force_dir  # [B,3]

            obj_vel = self.phys_friction * obj_vel + force
            obj_pos = (obj_pos + obj_vel).clamp(-1.0, 1.0)

            new_vel[:, i, :] = obj_vel
            new_pos[:, i, :] = obj_pos

        # Stacking: if two objects are very close in XY, accumulate height
        stack_threshold = 0.05
        for i in range(self.num_physics_objects):
            for j in range(i + 1, self.num_physics_objects):
                xy_dist = (new_pos[:, i, :2] - new_pos[:, j, :2]).norm(dim=1)  # [B]
                close = (xy_dist < stack_threshold).float().unsqueeze(1)  # [B,1]
                new_height[:, i, :] = new_height[:, i, :] + close * 0.5
                new_height[:, j, :] = new_height[:, j, :] + close * 0.5

        new_height = new_height.clamp(1.0, float(self.z))
        return new_pos, new_vel, new_height

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, state: WorldState, action_field: torch.Tensor, controls: EnvironmentControls | None = None) -> WorldState:
        if controls is None:
            controls = self.default_controls(action_field.size(0))
        if self.actuation_noise_std > 0.0:
            action_field = action_field + self.actuation_noise_std * torch.randn_like(action_field)
        if self.actuation_delay_steps > 0:
            self._action_buffer.append(action_field.clone())
            if len(self._action_buffer) <= self.actuation_delay_steps:
                delayed_action = torch.zeros_like(action_field)
            else:
                delayed_action = self._action_buffer.pop(0)
            action_field = delayed_action

        # Toroidal convolution: circular padding removes hard walls
        padded_life = F.pad(state.life, (1, 1, 1, 1, 1, 1), mode="circular")
        neighbors = F.conv3d(padded_life, self.kernel, padding=0)

        # Occlusion suppresses neighbour propagation across walls
        if self.num_occlusion_objects > 0:
            neighbors = neighbors * (1.0 - state.occlusion_mask)

        survive = ((neighbors >= 5) & (neighbors <= 7)).float() * state.life
        born = ((neighbors == 6).float()) * (1.0 - state.life)

        action_field_vol = action_field.view(action_field.size(0), 1, self.z, self.y, self.x)
        light_boost = self._light_field(controls)
        force_impact = self._force_field(controls)

        # Occlusion attenuates resources
        resource_occ_factor = 1.0 - 0.6 * state.occlusion_mask if self.num_occlusion_objects > 0 else 1.0

        adaptive_bonus = torch.sigmoid(action_field_vol + light_boost - 0.25 * force_impact) * (state.resources[:, :1] > 0.2).float()
        new_life = torch.clamp(survive + born + 0.25 * adaptive_bonus - 0.1 * force_impact, 0.0, 1.0)

        resource_use = 0.05 * new_life
        flow = self._apply_wind_flow(state.resources, controls.wind) * 0.02
        new_resources = torch.clamp(
            (state.resources + flow + 0.03 * light_boost - resource_use - self.decay * state.resources) * resource_occ_factor,
            0.0,
            1.0,
        )

        pressure = (neighbors / 26.0).clamp(0.0, 1.0)
        scarcity = 1.0 - new_resources[:, :1]
        wind_pressure = torch.norm(controls.wind, dim=1, keepdim=True).view(-1, 1, 1, 1, 1)
        hazard_stress = self._compute_hazard_stress(light_boost, state.occlusion_mask, controls.wind, self._step_index)

        # Bridging: physics objects near hazard zones reduce local stress
        bridge_reduction = torch.zeros_like(hazard_stress)
        if self.num_physics_objects > 0 and self.hazard_zones:
            for i in range(self.num_physics_objects):
                obj_pos = state.phys_pos[:, i, :]     # [B,3]
                obj_height = state.phys_height[:, i, :]  # [B,1]
                # Project object position onto coord_grid distance
                obj_xyz = obj_pos.view(-1, 3, 1, 1, 1)
                obj_dist2 = ((self.coord_grid - obj_xyz) ** 2).sum(dim=1, keepdim=True)
                span_factor = obj_height.view(-1, 1, 1, 1, 1) * 0.15
                bridge_reduction = bridge_reduction + span_factor * torch.exp(-obj_dist2 / 0.06)
            bridge_reduction = bridge_reduction.clamp(0.0, 0.5)

        new_stress = (
            0.55 * pressure
            + 0.35 * scarcity
            + 0.10 * wind_pressure
            + 0.2 * force_impact
            + hazard_stress
            - bridge_reduction
        ).clamp(0.0, 1.0)

        drag = max(0.0, min(1.0, 0.88 * self.surface_friction_scale))
        object_vel = drag * state.object_vel + controls.force_vector * controls.force_strength * controls.force_active
        object_pos = (state.object_pos + object_vel + controls.move_object_delta).clamp(-1.0, 1.0)

        new_phys_pos, new_phys_vel, new_phys_height = self._physics_step(state, action_field_vol)

        self._step_index += 1
        return WorldState(
            new_life,
            new_resources,
            new_stress,
            object_pos,
            object_vel,
            state.occlusion_mask,   # permanent — unchanged each step
            new_phys_pos,
            new_phys_vel,
            new_phys_height,
        )

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def encode_observation(self, state: WorldState, signal_dim: int) -> torch.Tensor:
        # Aggregate anonymous signal channels; do not encode modality identity.
        light_mean = self._light_field(self.default_controls(state.life.size(0))).mean(dim=(2, 3, 4))  # [B,1]
        wind_mag = torch.zeros(state.life.size(0), 1, device=self.device)  # placeholder; populated from controls in step

        # Hazard hint channels: pooled light field and step-phase sine
        step_phase = torch.full((state.life.size(0), 1), math.sin(2.0 * math.pi * self._step_index / 16.0), device=self.device)

        # Physics object proximity channels (anonymous distances — no object ID)
        if self.num_physics_objects > 0:
            # Agent reference: origin (0,0,0) in normalised space
            origin = torch.zeros(state.life.size(0), 3, device=self.device)
            dists = []
            for i in range(self.num_physics_objects):
                d = (state.phys_pos[:, i, :] - origin).norm(dim=1, keepdim=True)  # [B,1]
                dists.append(d)
            phys_proximity = torch.cat(dists, dim=1)  # [B, N]
        else:
            phys_proximity = torch.zeros(state.life.size(0), 0, device=self.device)

        pooled = torch.cat(
            [
                state.life.mean(dim=(2, 3, 4)),
                state.resources.mean(dim=(2, 3, 4)),
                state.stress.mean(dim=(2, 3, 4)),
                state.object_pos,
                state.object_vel,
                light_mean,
                wind_mag,
                step_phase,
                phys_proximity,
            ],
            dim=1,
        )
        if pooled.size(1) >= signal_dim:
            obs = pooled[:, :signal_dim]
        else:
            pad = torch.zeros(pooled.size(0), signal_dim - pooled.size(1), device=pooled.device)
            obs = torch.cat([pooled, pad], dim=1)
        if self.sensor_latency_steps > 0:
            self._obs_buffer.append(obs.clone())
            if len(self._obs_buffer) <= self.sensor_latency_steps:
                delayed_obs = torch.zeros_like(obs)
            else:
                delayed_obs = self._obs_buffer.pop(0)
            obs = delayed_obs
        if self.sensor_dropout_burst_prob > 0.0:
            if self._dropout_burst_steps_remaining <= 0 and torch.rand(1, device=self.device).item() < self.sensor_dropout_burst_prob:
                self._dropout_burst_steps_remaining = int(torch.randint(2, 6, (1,), device=self.device).item())
            if self._dropout_burst_steps_remaining > 0:
                keep = (torch.rand_like(obs) > 0.45).float()
                obs = obs * keep
                self._dropout_burst_steps_remaining -= 1
        return obs
