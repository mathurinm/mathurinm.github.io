# %%
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

    def step(self, x_t, t_start, t_end):
        """Euler ODE solver"""
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        delta = t_end - t_start
        return ...  # TODO


# %%
flow = Flow()
optimizer = torch.optim.Adam(flow.parameters(), 1e-2)
loss_fn = nn.MSELoss()

for it in range(10_000):
    if it % 500 == 0:
        print(it)
    x_1 = ...
    x_0 = ...
    t = ...  # (must be of shape (256, 1))

    x_t = ...
    u_true = ...
    optimizer.zero_grad()
    loss = ...
    loss.backward()
    optimizer.step()

# %%
plt.close('all')
x = torch.randn(300, 2)
n_steps = 8
time_steps = torch.linspace(0, 1, n_steps+1)
fig, ax = plt.subplots(1, n_steps + 1, sharex=True, sharey=True, figsize=(15, 3))
ax[0].scatter(*x.detach().T, s=10, alpha=0.5)
ax[0].set_xlim(-3, 3)
ax[0].set_ylim(-3, 3)
ax[0].set_title(f't = 0.00', fontsize=18)


for i in range(n_steps):
    x = flow.step(x, time_steps[i], time_steps[i+1])
    ax[i + 1].scatter(*x.detach().T, s=10, alpha=0.5)
    ax[i + 1].set_title(f't = {time_steps[i + 1]:.2f}', fontsize=18)

for a in ax:
    a.set_aspect("equal")
plt.tight_layout()
plt.show(block=False)

plt.savefig("lab.pdf")

# %%
