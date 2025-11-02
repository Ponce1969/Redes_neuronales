from __future__ import annotations

import numpy as np

from core.autograd_numpy.tensor import Tensor
from core.trm_act_block import TRM_ACT_Block


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

model = TRM_ACT_Block(n_in=2, n_hidden=8, max_steps=6)
lr = 0.05

for epoch in range(2000):
    total_loss = 0.0
    for i in range(len(X)):
        x = Tensor(X[i : i + 1])
        y_true = Tensor(Y[i : i + 1])
        loss = model.deep_supervision_loss(x, y_true)
        total_loss += loss.data
        for w in [model.W_in, model.W_z, model.W_out, model.W_halt, model.b]:
            grad = np.sign(np.random.randn(*w.data.shape)) * 0.001
            w.data -= lr * grad
    if epoch % 400 == 0:
        print(f"Epoch {epoch:4d} | Loss={total_loss:.5f}")

print("\nRazonamiento adaptativo XOR:")
for x in X:
    y_pred = model.forward(Tensor(x[None, :]))
    print(f"Input {x} -> Pred: {y_pred.data.squeeze():.3f}")
