from __future__ import annotations

import numpy as np

from core.autograd_numpy.tensor import Tensor
from core.autograd_numpy.loss import mse_loss
from core.trm_block import TRMBlock


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

model = TRMBlock(n_in=2, n_hidden=8, n_steps=4)
lr = 0.05

for epoch in range(2000):
    total_loss = 0.0
    for i in range(len(X)):
        x = Tensor(X[i : i + 1])
        y_true = Tensor(Y[i : i + 1])
        y_pred = model.forward(x)
        loss = mse_loss(y_pred, y_true)
        total_loss += loss.data

        for w in [model.W_in, model.W_z, model.W_out, model.b]:
            grad = (y_pred.data - y_true.data) * 2 * 0.1
            w.data -= lr * np.mean(grad)
    if epoch % 400 == 0:
        print(f"Epoch {epoch:4d} | Loss={total_loss:.6f}")

print("\nPredicciones XOR:")
for x in X:
    pred = model.forward(Tensor(x[None, :]))
    print(f"{x} -> {pred.data.squeeze():.3f}")
