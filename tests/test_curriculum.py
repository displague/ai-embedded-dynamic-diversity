import pytest
from ai_embedded_dynamic_diversity.train.curriculum import AdaptiveLossController

def test_adaptive_loss_controller_increase():
    initial_weights = {"detection_loss_weight": 0.1, "remap_loss_weight": 0.1}
    # Target ratios: detection=0.15, remap=0.15
    controller = AdaptiveLossController(initial_weights, alpha=0.1)
    
    # Scenario: detection contribution is too low
    # loss=1.0, detection_loss=0.1 -> contribution = 0.1 * 0.1 = 0.01 (ratio 0.01 < 0.15)
    logs = {"loss": 1.0, "detection_loss": 0.1, "remap_loss": 1.0}
    
    new_weights = controller.step(logs)
    assert new_weights["detection_loss_weight"] > 0.1
    # Remap: contribution = 0.1 * 1.0 = 0.1 (ratio 0.1 < 0.15)
    assert new_weights["remap_loss_weight"] > 0.1

def test_adaptive_loss_controller_decrease():
    initial_weights = {"detection_loss_weight": 0.5, "remap_loss_weight": 0.1}
    controller = AdaptiveLossController(initial_weights, alpha=0.1)
    
    # Scenario: detection contribution is too high
    # loss=1.0, detection_loss=1.0 -> contribution = 0.5 * 1.0 = 0.5 (ratio 0.5 > 0.15 * 2.0 = 0.30)
    logs = {"loss": 1.0, "detection_loss": 1.0, "remap_loss": 0.1}
    
    new_weights = controller.step(logs)
    assert new_weights["detection_loss_weight"] < 0.5

def test_adaptive_loss_controller_clamping():
    initial_weights = {"detection_loss_weight": 1.95}
    controller = AdaptiveLossController(initial_weights, alpha=0.1)
    
    # Force increase
    logs = {"loss": 10.0, "detection_loss": 0.001}
    new_weights = controller.step(logs)
    assert new_weights["detection_loss_weight"] <= 2.0
