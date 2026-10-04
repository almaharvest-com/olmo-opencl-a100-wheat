#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Compare olmo_cl's AdamW and ReduceLROnPlateau with PyTorch.

Usage: test_optim.py PATH_TO_test_optim_BINARY
"""

import math
import subprocess
import sys

import torch

N, STEPS, EPOCHS = 8, 300, 60

out = [float(x) for x in subprocess.check_output([sys.argv[1]]).split()]

# float64 torch: olmo_cl keeps the Adam moments in double, so only the
# algorithm is compared, not float32 rounding. The parameters themselves
# are float32 in both, hence the 1e-6 tolerance.
p = torch.tensor([0.1 * (i - 3) for i in range(N)], dtype=torch.float64,
                 requires_grad=True)
opt = torch.optim.AdamW([p], lr=1e-3)
for t in range(STEPS):
    with torch.no_grad():
        g = torch.tensor([math.sin(0.37 * t + i) * (1.0 + 0.1 * i)
                          for i in range(N)], dtype=torch.float32)
        p.grad = (g + p.float()).double()
    opt.step()
    with torch.no_grad():
        p.copy_(p.float().double())
err = max(abs(a - b) for a, b in zip(out[:N], p.tolist()))

q = torch.zeros(1, requires_grad=True)
opt2 = torch.optim.AdamW([q], lr=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt2, factor=0.2, patience=2, cooldown=10, min_lr=0)
lrs = []
for e in range(EPOCHS):
    m = (1.0 - 0.05 * e if e < 10 else
         0.5 + 0.00001 * math.sin(e) - (0.01 * (e - 40) if e > 40 else 0))
    sched.step(m)
    lrs.append(opt2.param_groups[0]["lr"])
lr_err = max(abs(a - b) / b for a, b in zip(out[N:], lrs))
changes = sorted({(i, lr) for i, lr in enumerate(lrs)
                  if i == 0 or lr != lrs[i - 1]})
print(f"AdamW max |param diff| after {STEPS} steps: {err:.2e}")
print(f"Plateau lr max rel diff over {EPOCHS} epochs: {lr_err:.2e}; "
      f"torch lr changes at {changes}")
sys.exit(0 if err < 1e-6 and lr_err < 1e-9 else 1)
