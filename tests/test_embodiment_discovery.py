import pytest
from ai_embedded_dynamic_diversity.sim.embodiments import register_embodiment, get_embodiment, discover_from_spec, Embodiment, EMBODIMENTS

def test_register_embodiment_adds_to_global_map():
    new_emb = Embodiment(name="custom", controls=("c1", "c2"), sensors=("s1",))
    register_embodiment(new_emb)
    
    retrieved = get_embodiment("custom")
    assert retrieved == new_emb
    assert "custom" in EMBODIMENTS

def test_discover_from_spec_with_explicit_lists():
    spec = {
        "name": "explicit",
        "controls": ["joint1", "joint2"],
        "sensors": ["eye"]
    }
    emb = discover_from_spec(spec)
    assert emb.name == "explicit"
    assert emb.controls == ("joint1", "joint2")
    assert emb.sensors == ("eye",)

def test_discover_from_spec_with_counts():
    spec = {
        "name": "counted",
        "control_count": 4,
        "sensor_count": 2
    }
    emb = discover_from_spec(spec)
    assert emb.name == "counted"
    assert len(emb.controls) == 4
    assert len(emb.sensors) == 2
    assert emb.controls[0] == "control_000"
    assert emb.sensors[0] == "sensor_000"

def test_get_embodiment_is_case_insensitive():
    new_emb = Embodiment(name="CaseTest", controls=("c1",), sensors=("s1",))
    register_embodiment(new_emb)
    assert get_embodiment("casetest").name == "CaseTest"
    assert get_embodiment("CASETEST").name == "CaseTest"
