"""
TaperredDendritesBalancedInputPDE.py

Balanced input version of the tapered dendrite PDE solver.
To be refined using Test-Driven Development principles.
"""

import numpy as np
import hashlib
from typing import List, Tuple, Optional
from pathlib import Path


def create_balanced_input_events(
    exc_times: List[float],
    exc_positions: List[float], 
    exc_weights: List[float],
    inh_times: List[float],
    inh_positions: List[float],
    inh_weights: List[float]
) -> List[Tuple[float, float, float]]:
    """
    Create balanced excitatory and inhibitory input events.
    
    Eliminates code duplication by providing a single function for creating
    both excitatory and inhibitory events with proper sorting.
    
    Parameters
    ----------
    exc_times : List[float]
        Times for excitatory events
    exc_positions : List[float]
        Positions for excitatory events
    exc_weights : List[float]
        Weights for excitatory events
    inh_times : List[float]
        Times for inhibitory events
    inh_positions : List[float]
        Positions for inhibitory events
    inh_weights : List[float]
        Weights for inhibitory events
        
    Returns
    -------
    List[Tuple[float, float, float]]
        List of (time, position, weight) tuples sorted by time
    """
    # Validate input lengths
    if not (len(exc_times) == len(exc_positions) == len(exc_weights)):
        raise ValueError("Excitatory input arrays must have equal length")
    
    if not (len(inh_times) == len(inh_positions) == len(inh_weights)):
        raise ValueError("Inhibitory input arrays must have equal length")
    
    events = []
    
    # Add excitatory events
    for t, x, w in zip(exc_times, exc_positions, exc_weights):
        events.append((float(t), float(x), float(w)))
    
    # Add inhibitory events
    for t, x, w in zip(inh_times, inh_positions, inh_weights):
        events.append((float(t), float(x), float(w)))
    
    # Sort by time for consistent processing (eliminates duplication of sorting logic)
    events.sort(key=lambda event: event[0])
    return events


def save_balanced_simulation_reference(
    times: np.ndarray,
    V_s: np.ndarray,
    t_max: float,
    events: List[Tuple[float, float, float]],
    seed: Optional[int] = None,
    simulation_label: str = "balanced_input_reference",
    reference_dir: Path = Path("./reference_simulations")
) -> Path:
    """
    Save balanced input simulation results as a reference for future comparison.
    
    Eliminates duplication of reference saving code by providing a single
    well-tested function for this purpose.
    
    Parameters
    ----------
    times : np.ndarray
        Time points from simulation
    V_s : np.ndarray
        Voltage matrix (spatial x time) from simulation
    t_max : float
        Maximum simulation time
    events : List[Tuple[float, float, float]]
        Input events used in simulation
    seed : Optional[int]
        Random seed used (if any)
    simulation_label : str
        Label for the simulation
    reference_dir : Path
        Directory to save reference files
        
    Returns
    -------
    Path
        Path to the saved reference file
    """
    # Ensure reference directory exists
    reference_dir.mkdir(exist_ok=True)
    
    # Create reference filename
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in simulation_label)
    reference_file = reference_dir / f"{safe_label}_reference.npz"
    
    # Save the simulation data
    np.savez(
        reference_file,
        times=times,
        V_s=V_s,
        t_max=t_max,
        events=np.array(events, dtype=object),  # Store events as array of objects
        seed=seed,
        simulation_label=simulation_label
    )
    
    return reference_file


def load_balanced_simulation_reference(
    reference_file: Path
) -> Tuple[np.ndarray, np.ndarray, float, List[Tuple[float, float, float]], Optional[int], str]:
    """
    Load a previously saved balanced input simulation reference.
    
    Eliminates duplication of reference loading code.
    
    Parameters
    ----------
    reference_file : Path
        Path to the reference file to load
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float, List[Tuple[float, float, float]], Optional[int], str]
        (times, V_s, t_max, events, seed, simulation_label)
    """
    if not reference_file.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_file}")
    
    data = np.load(reference_file, allow_pickle=True)
    
    times = data['times']
    V_s = data['V_s']
    t_max = float(data['t_max'])
    # Convert loaded events back to list of tuples
    events_raw = data['events'].tolist() if 'events' in data.files else []
    events = [tuple(event) for event in events_raw] if len(events_raw) > 0 else []
    seed = int(data['seed']) if 'seed' in data.files and data['seed'] is not None else None
    simulation_label = str(data['simulation_label'].item()) if 'simulation_label' in data.files else ""
    
    return times, V_s, t_max, events, seed, simulation_label


def compute_simulation_hash(times: np.ndarray, V_s: np.ndarray) -> str:
    """
    Compute a hash of simulation results for reference tracking and duplication detection.
    
    Eliminates duplication of hash computation code.
    
    Parameters
    ----------
    times : np.ndarray
        Time points from simulation
    V_s : np.ndarray
        Voltage matrix from simulation
        
    Returns
    -------
    str
        MD5 hash of the simulation characteristics
    """
    # Create a deterministic representation for hashing
    hash_data = {
        'times_shape': times.shape,
        'times_length': len(times),
        'times_first': float(times[0]) if len(times) > 0 else 0.0,
        'times_last': float(times[-1]) if len(times) > 0 else 0.0,
        'V_s_shape': V_s.shape,
        'V_s_size': V_s.size,
        'V_s_mean': float(np.mean(V_s)) if V_s.size > 0 else 0.0,
        'V_s_std': float(np.std(V_s)) if V_s.size > 0 else 0.0,
        'V_s_min': float(np.min(V_s)) if V_s.size > 0 else 0.0,
        'V_s_max': float(np.max(V_s)) if V_s.size > 0 else 0.0,
        # Sample key points to detect changes
        'V_s_sample_0_0': float(V_s[0, 0]) if V_s.size > 0 else 0.0,
        'V_s_sample_-1_-1': float(V_s[-1, -1]) if V_s.size > 0 else 0.0,
    }
    
    # Handle edge case of empty array
    if V_s.size > 0 and V_s.shape[0] > 1 and V_s.shape[1] > 1:
        mid_x, mid_y = V_s.shape[0] // 2, V_s.shape[1] // 2
        hash_data['V_s_sample_mid'] = float(V_s[mid_x, mid_y])
    
    # Create deterministic string representation
    data_str = str(sorted(hash_data.items()))
    return hashlib.md5(data_str.encode()).hexdigest()
