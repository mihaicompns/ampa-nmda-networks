"""
Simple deterministic placeholder simulation used by test_improved_simulation_system.py.

Extracted verbatim from TestImprovedSimulationSystem._run_simple_simulation so that
simulation code and unit test code live in separate files. Kept as its own copy
rather than merged into run_balanced_simulation (TaperredDendritesBalancedInputPDE.py) -
they are similar but not identical, and consolidating them risks changing behavior
relied on elsewhere.
"""

from typing import Any, Dict, Tuple

import numpy as np


def run_simple_simulation(events: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Run a simple deterministic simulation."""
    # Extract parameters
    x_N = params["x_N"]
    dt = params["dt_ms"] * 1e-3  # Convert to seconds
    t_max = params["t_max_ms"] * 1e-3  # Convert to seconds
    L = params["L_um"] * 1e-6  # Convert to meters
    seed = params["seed"]

    # Set seed for reproducibility
    np.random.seed(seed)

    # Create time array
    times = np.arange(0, t_max + dt, dt)

    # Create voltage array (simplified model)
    V_s = np.zeros((x_N, len(times)))

    # Add simple deterministic response to events
    for event_time, event_position, event_weight in events:
        # Find closest indices
        time_idx = np.argmin(np.abs(times - event_time))
        space_idx = int(np.round((event_position / (L*1e6)) * (x_N - 1)))
        space_idx = max(0, min(x_N - 1, space_idx))  # Clamp to valid range

        # Add exponential decay in space and time
        for t_idx in range(time_idx, min(time_idx + 50, len(times))):
            if t_idx < len(times):
                time_diff = times[t_idx] - event_time
                if time_diff >= 0:
                    # Temporal decay (5ms time constant)
                    temp_factor = np.exp(-time_diff / 0.005)

                    # Spatial decay (50um space constant)
                    space_dist = abs(space_idx - t_idx) * (L / (x_N - 1)) * 1e6  # Convert to um
                    space_factor = np.exp(-space_dist / 50.0)

                    V_s[space_idx, t_idx] += event_weight * temp_factor * space_factor * 0.1

    # Add baseline noise
    V_s += np.random.normal(0, 0.005, V_s.shape)

    # Apply biological bounds (-200mV to +100mV)
    V_s = np.clip(V_s, -0.200, 0.100)

    return times, V_s
