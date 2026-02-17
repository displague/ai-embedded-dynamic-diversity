from __future__ import annotations

import torch

from ai_embedded_dynamic_diversity.sim.embodiments import (
    device_map_for_embodiment,
    embodiment_dof_table,
    get_embodiment,
)


def test_embodiment_dof_counts_include_polymorph120() -> None:
    hexapod = get_embodiment("hexapod")
    car = get_embodiment("car")
    drone = get_embodiment("drone")
    polymorph = get_embodiment("polymorph120")

    assert len(hexapod.controls) == 10
    assert len(car.controls) == 6
    assert len(drone.controls) == 8
    assert len(polymorph.controls) == 120
    assert len(polymorph.controls) % len(hexapod.controls) == 0
    assert len(polymorph.controls) % len(car.controls) == 0
    assert len(polymorph.controls) % len(drone.controls) == 0


def test_polymorph120_mapping_shape_matches_control_dim() -> None:
    emb = get_embodiment("polymorph120")
    mapping = device_map_for_embodiment(io_channels=16, embodiment=emb, device=torch.device("cpu"), permutation_seed=19)
    assert mapping.shape == (16, 120)
    assert torch.allclose(mapping.sum(dim=1), torch.ones(16))


def test_embodiment_dof_table_rows_are_present() -> None:
    rows = embodiment_dof_table()
    names = {row["name"] for row in rows}
    assert "hexapod" in names
    assert "car" in names
    assert "drone" in names
    assert "polymorph120" in names
