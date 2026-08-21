from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
ITERATION_SRC = SRC_ROOT / "iteration_19_tapered_dendrites"
for path in (REPO_ROOT, SRC_ROOT, ITERATION_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from TaperredDendritesBalancedCrankNicolson import _consume_event_stream


class FakeParameters:
    x = np.array([0.0, 0.5, 1.0])


def test_consume_event_stream_does_not_allocate_zero_vector_for_future_events():
    events = np.array([[0.2], [0.5]])

    contribution, next_idx = _consume_event_stream(
        t=0.0,
        dt=0.1,
        events=events,
        next_idx=0,
        p=FakeParameters(),
        current_amplitude=1.0,
        sign=1.0,
    )

    assert contribution is None
    assert next_idx == 0
