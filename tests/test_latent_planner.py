from __future__ import annotations

import numpy as np
from core.autograd_numpy.tensor import Tensor
from core.latent_planner_block import LatentPlannerBlock


def test_latent_planner_shapes_and_stability() -> None:
    planner = LatentPlannerBlock(
        n_in=3,
        n_hidden=6,
        planner_hidden=10,
        max_steps=3,
        detach_each_step=True,
        retain_plan=True,
    )

    x = Tensor(np.random.randn(1, 3).astype(np.float32))
    y = planner.forward(x)

    assert planner.get_last_plan() is not None
    assert y.data.shape == (1, 1)

    planner.zero_state()

    x2 = Tensor((x.data + 1e-3).astype(np.float32))
    y2 = planner.forward(x2)
    diff = abs(float(y.data.squeeze()) - float(y2.data.squeeze()))
    assert diff < 0.75
