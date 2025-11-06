"""Demo del LatentPlannerBlock planificando antes de razonar."""

from __future__ import annotations

import numpy as np

from core.autograd_numpy.tensor import Tensor
from core.latent_planner_block import LatentPlannerBlock


def demo() -> None:
    model = LatentPlannerBlock(
        n_in=2,
        n_hidden=8,
        max_steps=4,
        planner_hidden=12,
        detach_each_step=True,
        retain_plan=True,
    )

    dataset = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

    print("=== Latent Planner demo ===")
    for sample in dataset:
        x = Tensor(sample[None, :])
        y = model.forward(x)
        plan = model.get_last_plan()
        plan_mean = float(np.mean(plan)) if plan is not None else float("nan")
        print(f"in={sample} -> out={float(y.data.squeeze()):.4f} | plan_mean={plan_mean:.4f}")


if __name__ == "__main__":
    demo()
