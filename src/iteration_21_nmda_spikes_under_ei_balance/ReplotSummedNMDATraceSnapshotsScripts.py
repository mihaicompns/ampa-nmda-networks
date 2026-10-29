"""Recreate moving-nullcline MP4 and snapshots from a saved unit trace."""

import unittest
from pathlib import Path

from iteration_21_nmda_spikes_under_ei_balance.SimulateSummedNMDASpikesUnderEIBalanceScripts import (
    SUMMED_OUTPUT_DIR,
    generate_summed_nmda_moving_nullcline_animation_from_saved_trace,
    load_summed_nmda_unit_trace_csv,
    plot_summed_nmda_trace_snapshot,
    summed_nmda_trace_filename,
)


# Configure one previously saved scan run here.  Times are in milliseconds and
# are matched to the nearest recorded simulation sample.

RATE_SCAN_OUTPUT_DIR = SUMMED_OUTPUT_DIR / "nmda_only" / "rate_scan"


def _number_label(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def generate_configured_summed_nmda_trace_snapshots(
    *,
    rate_hz,
    unit_index,
    random_seed,
    snapshot_times_ms=(),
    movie_t_min_ms=None,
    movie_t_max_ms=None,
    movie_frame_count=1_000,
    movie_fps=40,
    rate_scan_output_dir=RATE_SCAN_OUTPUT_DIR,
):
    """Read a saved unit CSV and recreate its MP4 plus requested PNGs."""
    rate_scan_output_dir = Path(rate_scan_output_dir)
    trace_path = (
        rate_scan_output_dir
        / "trace_data"
        / summed_nmda_trace_filename(rate_hz, unit_index, random_seed)
    )
    saved_trace = load_summed_nmda_unit_trace_csv(trace_path)
    movie_path = rate_scan_output_dir / "animations_from_saved_csv" / (
        f"summed_nmda_re_{_number_label(rate_hz)}_Hz_unit_{unit_index}_"
        f"seed_{random_seed}_{movie_frame_count}_frames_{movie_fps}_fps.mp4"
    )
    generate_summed_nmda_moving_nullcline_animation_from_saved_trace(
        saved_trace,
        movie_path,
        t_min_ms=movie_t_min_ms,
        t_max_ms=movie_t_max_ms,
        frame_count=movie_frame_count,
        fps=movie_fps,
        output_format="mp4",
    )
    snapshot_dir = rate_scan_output_dir / "snapshots"
    output_paths = []
    for snapshot_time_ms in snapshot_times_ms:
        output_path = snapshot_dir / (
            f"summed_nmda_re_{_number_label(rate_hz)}_Hz_unit_{unit_index}_"
            f"t_{_number_label(snapshot_time_ms)}_ms_seed_{random_seed}.png"
        )
        output_paths.append(
            plot_summed_nmda_trace_snapshot(
                saved_trace,
                snapshot_time_ms=snapshot_time_ms,
                output_path=output_path,
            )
        )
    return trace_path, movie_path, output_paths


class ReplotSummedNMDATraceSnapshotsScripts(unittest.TestCase):
    """Manual runnable that recreates a movie and PNGs without simulation."""

    def test_generate_configured_snapshots_from_saved_csv(self):
        replot_times_ms = (550.0, 600.0, 650.0)
        movie_frame_count = 1_000
        movie_fps = 40

        trace_path, movie_path, output_paths = (
            generate_configured_summed_nmda_trace_snapshots(
                rate_hz=25,
                unit_index=0,
                random_seed=22,
                snapshot_times_ms=replot_times_ms,
                movie_frame_count=movie_frame_count,
                movie_fps=movie_fps,
            )
        )

        self.assertTrue(trace_path.exists(), trace_path)
        self.assertTrue(movie_path.exists(), movie_path)
        self.assertGreater(movie_path.stat().st_size, 0, movie_path)
        self.assertEqual(len(output_paths), len(replot_times_ms))
        for output_path in output_paths:
            self.assertTrue(output_path.exists(), output_path)
            self.assertGreater(output_path.stat().st_size, 0, output_path)


if __name__ == "__main__":
    unittest.main()
