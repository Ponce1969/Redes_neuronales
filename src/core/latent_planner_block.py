"""Latent planner block that extends TRM_ACT_Block with an initial plan."""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.autograd_numpy.tensor import Tensor
from core.autograd_numpy.loss import mse_loss
from core.trm_act_block import TRM_ACT_Block


class LatentPlannerBlock(TRM_ACT_Block):
    """TRM_ACT_Block + latent planning stage before recursive reasoning."""

    def __init__(
        self,
        n_in: int,
        n_hidden: int = 8,
        max_steps: int = 6,
        planner_hidden: int = 16,
        detach_each_step: bool = True,
        retain_plan: bool = False,
    ) -> None:
        super().__init__(n_in=n_in, n_hidden=n_hidden, max_steps=max_steps)
        z_dim = n_hidden
        self.planner_W1 = Tensor(np.random.randn(n_in, planner_hidden).astype(np.float32) * 0.08)
        self.planner_b1 = Tensor(np.zeros((1, planner_hidden), dtype=np.float32))
        self.planner_W2 = Tensor(np.random.randn(planner_hidden, z_dim).astype(np.float32) * 0.08)
        self.planner_b2 = Tensor(np.zeros((1, z_dim), dtype=np.float32))

        self.detach_each_step = detach_each_step
        self.retain_plan = retain_plan
        self.last_z_plan: Optional[np.ndarray] = None

    def plan(self, x: Tensor) -> Tensor:
        hidden = (x.matmul(self.planner_W1) + self.planner_b1).tanh()
        z_plan = (hidden.matmul(self.planner_W2) + self.planner_b2).tanh()
        self.last_z_plan = z_plan.data.copy()
        return z_plan

    def forward(self, x: Tensor) -> Tensor:
        z_plan = self.plan(x)
        self.z = z_plan.detach() if self.detach_each_step else z_plan

        total_output = Tensor(np.zeros((1, 1), dtype=np.float32))
        remaining_prob = 1.0

        for _ in range(self.max_steps):
            z_pre = x.matmul(self.W_in) + self.z.matmul(self.W_z) + self.b
            z_new = z_pre.tanh()
            y = z_new.matmul(self.W_out).tanh()
            h = z_new.matmul(self.W_halt).sigmoid()
            h_val = float(h.data.squeeze())

            p_t = remaining_prob * h_val
            total_output = total_output + y * p_t
            remaining_prob *= (1.0 - h_val)

            self.z = z_new.detach() if self.detach_each_step else z_new

            if h_val > 0.5:
                break

        if not self.retain_plan:
            self.last_z_plan = None

        return total_output

    def deep_supervision_loss(self, x: Tensor, y_true: Tensor) -> Tensor:
        z_plan = self.plan(x)
        backup_z = self.z
        previous_plan = None if self.last_z_plan is None else self.last_z_plan.copy()
        self.z = z_plan.detach() if self.detach_each_step else z_plan

        total_loss = 0.0
        for _ in range(self.max_steps):
            z_pre = x.matmul(self.W_in) + self.z.matmul(self.W_z) + self.b
            z_new = z_pre.tanh()
            y_pred = z_new.matmul(self.W_out).tanh()
            loss = mse_loss(y_pred, y_true)
            total_loss += float(loss.data)
            self.z = z_new.detach() if self.detach_each_step else z_new

        self.z = backup_z
        if not self.retain_plan:
            self.last_z_plan = previous_plan
        return Tensor(total_loss / self.max_steps)

    def zero_state(self) -> None:
        super().zero_state()
        self.last_z_plan = None

    def get_last_plan(self) -> Optional[np.ndarray]:
        return self.last_z_plan
