from __future__ import annotations

import numpy as np

from core.meta.rules import adaptive_focus, adaptive_lr, adaptive_sleep


def test_adaptive_lr_reduces_on_increase() -> None:
    new_lr = adaptive_lr(prev_loss=0.1, curr_loss=0.2, lr=0.01)
    assert new_lr < 0.01


def test_adaptive_lr_increases_on_improvement() -> None:
    new_lr = adaptive_lr(prev_loss=0.2, curr_loss=0.18, lr=0.01)
    assert new_lr > 0.01


def test_adaptive_focus_bounds() -> None:
    low_focus = adaptive_focus(0.05, 0.5)
    high_focus = adaptive_focus(0.95, 1.5)
    assert 0.1 <= low_focus <= 2.0
    assert 0.1 <= high_focus <= 2.0
    assert low_focus > 0.5
    assert high_focus < 1.5


def test_adaptive_sleep_trend() -> None:
    interval = adaptive_sleep([0.4, 0.41, 0.42], base_interval=3)
    assert interval == 4

    interval = adaptive_sleep([0.4, 0.38, 0.36], base_interval=3)
    assert interval == 2

    interval = adaptive_sleep([0.4], base_interval=3)
    assert interval == 3
