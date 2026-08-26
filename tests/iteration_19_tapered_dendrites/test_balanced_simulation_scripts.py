import shutil
import sys
import unittest
from pathlib import Path

from brian2 import ms, um


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
ITERATION_SRC = SRC_ROOT / "iteration_19_tapered_dendrites"
for path in (REPO_ROOT, SRC_ROOT, ITERATION_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.iteration_19_tapered_dendrites.data import to_SI
import src.iteration_19_tapered_dendrites.TaperedDendritesBalancedSimulationScripts as scripts


class TestBalancedSimulationScripts(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("./test_balanced_simulation_scripts")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_run_balanced_limit_sweep_1000um_waits_and_generates_files(self):
        L = to_SI(1000 * um)

        results = scripts.test_run_balanced_limit_sweep_1000um(
            t_max=to_SI(0.001 * ms),
            limit_sets=[(0.75 * L, L)],
            output_root=self.test_dir,
            saved_frames=2,
            verbose=False,
            plot=False,
            show_plot=False,
            max_workers=1,
        )

        self.assertEqual(len(results), 1)
        comparison = results[0]

        self.assertTrue((comparison["save_dir"] / "conical").is_dir())
        self.assertTrue((comparison["save_dir"] / "cylindrical").is_dir())
        for result in (comparison["conical"], comparison["cylindrical"]):
            self.assertTrue((result.save_dir / "inputs").is_dir())
            self.assertTrue((result.save_dir / "outputs").is_dir())
            self.assertTrue((result.save_dir / "graphs").is_dir())
            self.assertTrue((result.save_dir / "metadata").is_dir())
            self.assertTrue((result.save_dir / "statistics").is_dir())
            self.assertTrue(result.inputs_file.exists())
            self.assertTrue(result.outputs_file.exists())
            self.assertTrue(result.metadata_file.exists())
            self.assertTrue(result.statistics_file.exists())

def test_run_balanced_limit_sweep_1000um_uses_one_bounded_runner(monkeypatch, tmp_path):
    L = to_SI(1000 * um)
    captured = {}

    def fake_bounded_runner(comparison_kwargs, max_workers):
        captured["comparison_kwargs"] = comparison_kwargs
        captured["max_workers"] = max_workers
        return [{"save_dir": tmp_path / str(index)} for index, _ in enumerate(comparison_kwargs)]

    monkeypatch.setattr(scripts, "run_balanced_comparison_sweep", fake_bounded_runner)

    results = scripts.test_run_balanced_limit_sweep_1000um(
        t_max=to_SI(0.001 * ms),
        limit_sets=[(0.0, L), (0.75 * L, L)],
        output_root=tmp_path,
        saved_frames=2,
        verbose=False,
        plot=False,
        show_plot=False,
        max_workers=12,
    )

    assert captured["max_workers"] == 12
    assert len(captured["comparison_kwargs"]) == 4
    assert len(results) == 4


if __name__ == "__main__":
    unittest.main()
