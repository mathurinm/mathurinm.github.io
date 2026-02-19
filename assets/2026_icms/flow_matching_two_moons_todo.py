# %%
import numpy as np
import torch
from sklearn.datasets import make_moons
from torch import nn, Tensor
import matplotlib.pyplot as plt

# %%


class Flow(nn.Module):
    def __init__(self, dim=2, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim)
        )

    def forward(self, x_t, t):
        return self.net(torch.cat((t, x_t), -1))

    def euler_step(self, x_t, t, stepsize):
        """One step of Euler method at time `t`, with stepsize `stepsize`.
        This is used at generation time only.
        """
        t = t.view(1, 1).expand(x_t.shape[0], 1)
        return ...  # TODO


# %%
flow = Flow()
optimizer = torch.optim.Adam(flow.parameters(), 1e-2)
loss_fn = nn.MSELoss()
batch_size = 256

for it in range(10_000):
    if it % 500 == 0:
        print(it)
    x_1 = ...  # TODO
    x_0 = ...  # TODO
    t = ...  # TODO (must be of shape (batch_size, 1))

    x_t = ...  # TODO
    u_true = ...  # TODO
    optimizer.zero_grad()
    loss = ...  # TODO
    loss.backward()
    optimizer.step()

# %% data generation
x = torch.randn(300, 2)
n_steps = 100
time_steps = torch.linspace(0, 1, n_steps)
xt = torch.zeros((n_steps, 300, 2))
xt[0] = x
for i in range(n_steps-1):
    x = ...  # TODO
    xt[i + 1] = x

# %% plots
