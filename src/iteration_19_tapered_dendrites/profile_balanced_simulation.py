"""
Profile one balanced tapered-dendrite simulation save workflow.

Example:
    ../../agentic-ampa/bin/python profile_balanced_simulation.py --t-max-ms 20 --dt-us 10
"""

import argparse
import cProfile
from pathlib import Path
import pstats
import sys

from brian2 import Hz, second, um
from brian2.units.allunits import pampere


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.iteration_19_tapered_dendrites.data import to_SI
from src.iteration_19_tapered_dendrites.ImprovedSimulationSave import (
    BalancedSimulationMetadata,
    _generate_uniform_events,
    run_and_save_balanced_conical_simulation,
    run_and_save_balanced_cylindrical_simulation,
)
from src.iteration_19_tapered_dendrites.TaperredDendritesPDE import default_params


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=("conical", "cylindrical"), default="conical")
    parser.add_argument("--output-root", type=Path, default=Path("profiling_runs"))
    parser.add_argument("--profile-name", default="balanced_simulation")
    parser.add_argument("--t-max-ms", type=float, default=20.0)
    parser.add_argument("--dt-us", type=float, default=10.0)
    parser.add_argument("--length-um", type=float, default=2_000.0)
    parser.add_argument("--spatial-points", type=int, default=101)
    parser.add_argument("--saved-frames", type=int, default=200)
    parser.add_argument("--e-left-um", type=float, default=500.0)
    parser.add_argument("--e-right-um", type=float, default=1_000.0)
    parser.add_argument("--i-left-um", type=float, default=1_000.0)
    parser.add_argument("--i-right-um", type=float, default=2_000.0)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--g", type=float, default=5.0)
    parser.add_argument("--base-rate-hz-per-um", type=float, default=0.4)
    parser.add_argument("--base-i-e-pa", type=float, default=150.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sort", default="cumtime")
    parser.add_argument("--limit", type=int, default=40)
    return parser.parse_args()


def run_profiled_simulation(args):
    t_max = to_SI(args.t_max_ms / 1000.0 * second)
    dt = to_SI(args.dt_us * 1e-6 * second)
    length = to_SI(args.length_um * um)
    e_limits = (to_SI(args.e_left_um * um), to_SI(args.e_right_um * um))
    i_limits = (to_SI(args.i_left_um * um), to_SI(args.i_right_um * um))
    base_rate = args.base_rate_hz_per_um * Hz / um
    base_I_e = to_SI(args.base_i_e_pa * pampere)

    metadata = BalancedSimulationMetadata.from_brunel_params(
        simulation_label=args.profile_name,
        e_limits=e_limits,
        i_limits=i_limits,
        t_max=t_max,
        gamma=args.gamma,
        g=args.g,
        base_rate=base_rate,
        base_I_e_strength=base_I_e,
        seed=args.seed,
    )

    conical_p = default_params.with_SI_properties(
        dt=dt,
        N=args.spatial_points,
        L=length,
        I_e=base_I_e,
        I_i=args.g * base_I_e,
    ).to_numerical()

    common_kwargs = {
        "t_max": t_max,
        "output_root": args.output_root / args.geometry,
        "simulation_label": args.profile_name,
        "e_limits": e_limits,
        "i_limits": i_limits,
        "r_e_density": metadata.r_e_density,
        "r_i_density": metadata.r_i_density,
        "g": args.g,
        "experiment_metadata": metadata,
        "seed": args.seed,
        "saved_frames": args.saved_frames,
        "verbose": False,
        "plot": False,
        "show_plot": False,
    }

    if args.geometry == "conical":
        return run_and_save_balanced_conical_simulation(p=conical_p, **common_kwargs)

    excitatory_events, inhibitory_events = _generate_uniform_events(
        t_max=t_max,
        e_limits=e_limits,
        i_limits=i_limits,
        seed=args.seed,
        r_e_density=metadata.r_e_density,
        r_i_density=metadata.r_i_density,
    )
    return run_and_save_balanced_cylindrical_simulation(
        p=conical_p.to_numerical_cylindrical_params(),
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        **common_kwargs,
    )


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    profile_file = args.output_root / f"{args.profile_name}_{args.geometry}.prof"
    stats_file = args.output_root / f"{args.profile_name}_{args.geometry}_pstats.txt"

    profiler = cProfile.Profile()
    result = profiler.runcall(run_profiled_simulation, args)
    profiler.dump_stats(profile_file)

    with stats_file.open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(profiler, stream=handle).strip_dirs().sort_stats(args.sort)
        stats.print_stats(args.limit)

    print(f"saved simulation: {result.save_dir}")
    print(f"profile: {profile_file}")
    print(f"stats: {stats_file}")


if __name__ == "__main__":
    main()
