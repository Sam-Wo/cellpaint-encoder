import math

import numpy as np
import torch

from cellpaint_encoder.models.dino_loss import DINOLoss, collapse_monitors
from cellpaint_encoder.train.loop import ema_update
from cellpaint_encoder.train.schedules import cosine_schedule, teacher_temp_schedule


def test_loss_finite_and_pairs():
    torch.manual_seed(0)
    K, B, n = 32, 4, 4
    loss = DINOLoss(K, n)
    s, t = torch.randn(n * B, K), torch.randn(2 * B, K)
    v = loss(s, t, 0.04)
    assert torch.isfinite(v) and v > 0
    # a student whose softmax equals the sharpened teacher reaches the minimum (teacher entropy)
    x = torch.randn(B, K)
    t2 = torch.cat([x, x])
    s_good = torch.cat([x] * n) * (0.1 / 0.04)
    q = torch.softmax(x / 0.04, -1)
    ent = float(-(q * torch.log_softmax(x / 0.04, -1)).sum(-1).mean())
    assert math.isclose(float(loss(s_good, t2, 0.04)), ent, rel_tol=1e-4)
    assert loss(s_good, t2, 0.04) < loss(torch.randn(n * B, K), t2, 0.04)


def test_uniform_outputs_give_log_k():
    K = 16
    loss = DINOLoss(K, 3)
    v = loss(torch.zeros(6, K), torch.zeros(4, K), 0.04)
    assert math.isclose(float(v), math.log(K), rel_tol=1e-5)


def test_center_update_ema():
    loss = DINOLoss(4, 2, center_momentum=0.9)
    batch = torch.tensor([[1.0, 2, 3, 4], [3.0, 2, 1, 0]])
    loss.update_center(batch.sum(0), 2)
    assert torch.allclose(loss.center, 0.1 * batch.mean(0, keepdim=True))
    loss.update_center(batch.sum(0), 2)
    assert torch.allclose(loss.center, (0.9 * 0.1 + 0.1) * batch.mean(0, keepdim=True))


def test_centering_removes_constant_bias():
    loss = DINOLoss(8, 2)
    t = torch.randn(4, 8)
    s = torch.randn(4, 8)
    loss.center.fill_(3.0)
    a = loss(s, t + 3.0, 0.04)
    loss.center.zero_()
    b = loss(s, t, 0.04)
    assert torch.allclose(a, b, atol=1e-6)


def test_sharpening_lowers_teacher_entropy():
    t = torch.randn(8, 32)
    e_sharp = collapse_monitors(torch.randn(16, 32), t, torch.zeros(1, 32), 0.04, 0.1, torch.randn(8, 4))
    e_soft = collapse_monitors(torch.randn(16, 32), t, torch.zeros(1, 32), 0.5, 0.1, torch.randn(8, 4))
    assert e_sharp["teacher_entropy"] < e_soft["teacher_entropy"] <= e_soft["max_entropy"] + 1e-6


def test_temperature_and_momentum_schedules():
    t = teacher_temp_schedule(0.04, 0.07, 30, 100, 10)
    assert math.isclose(t[0], 0.04) and math.isclose(t[300], 0.07) and np.all(np.diff(t[:300]) > 0)
    m = cosine_schedule(0.996, 1.0, 100, 10)
    assert math.isclose(m[0], 0.996) and m[-1] > 0.99999 and np.all(np.diff(m) >= 0)
    wd = cosine_schedule(0.04, 0.4, 100, 10)
    assert math.isclose(wd[0], 0.04) and abs(wd[-1] - 0.4) < 1e-4


def test_ema_update():
    s, t = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    t0 = [p.detach().clone() for p in t.parameters()]
    ema_update(t, s, 0.9)
    for p_t, p0, p_s in zip(t.parameters(), t0, s.parameters()):
        assert torch.allclose(p_t, 0.9 * p0 + 0.1 * p_s)
    ema_update(t, s, 1.0)  # momentum 1 freezes the teacher
    snap = [p.detach().clone() for p in t.parameters()]
    ema_update(t, s, 1.0)
    assert all(torch.equal(a, b) for a, b in zip(snap, t.parameters()))
