# World Harness Gap Analysis v1

## Goal

Reduce overly optimistic success predictions by hardening simulator realism and promotion evaluation.

## Current Strengths

- Scenarioized wind/light/force disturbances in `cross_eval`.
- Noise profiles (`dropout-quant-v1/v2`) in training + evaluation.
- Cross-embodiment transfer and capability scoring.
- Convergence gates for symbio/autopoiesis/checkmate.

## Main Gaps

1. Limited temporal dynamics in control and sensing path.
2. Disturbance changes are often too independent step-to-step.
3. Low friction/terrain regime diversity.
4. Promotion pipeline lacked explicit optimism-gap penalty between standard and harsher settings.

## Implemented in v1

1. Added world realism knobs (default-off for compatibility):
   - `actuation_delay_steps`
   - `actuation_noise_std`
   - `sensor_latency_steps`
   - `sensor_dropout_burst_prob`
   - `surface_friction_scale`
   - `disturbance_correlation_horizon`
2. Added world profiles:
   - `large_v1`
   - `large_v1_extreme`
3. Added calibrated scenarios:
   - `latency-storm`
   - `friction-shift`
   - `persistent-gust`
4. Added optimism-gap ranking/gate fields in cross-eval/report:
   - standard vs calibrated transfer split
   - optimism gap
   - optimism penalty component
   - optimism gate pass

## Next Calibration Work

1. Tune `large_v1_extreme` defaults against failure signatures in `add-viz batch-force`.
2. Promote only from candidates that keep rank under calibrated profile and satisfy optimism gate.
3. Add heldout calibration profiles with unseen disturbance combinations.
