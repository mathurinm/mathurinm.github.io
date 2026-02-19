# %%
import torch
import numpy as np
from torch import nn, Tensor
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

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
        return x_t + stepsize * self(x_t, t)


# %% network training
flow = Flow()
optimizer = torch.optim.Adam(flow.parameters(), 1e-2)
loss_fn = nn.MSELoss()  # squared L2 loss for network training
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
x = torch.randn(300, 2)  # we will generate 300 points
n_steps = 100
time_steps = torch.linspace(0, 1, n_steps)
xt = torch.zeros((n_steps, x.shape[0], x.shape[1]))
xt[0] = x
for i in range(n_steps-1):
    x = flow.step(x, time_steps[i], time_steps[i+1])
    xt[i + 1] = x

# %% plots
plt.close('all')

tab10 = plt.colormaps["tab10"]
fig, axarr = plt.subplots(1, 2, sharex=True, sharey=True)
x1 = make_moons(256, noise=0.05)[0]
axarr[0].scatter(*x1.T, color=tab10.colors[1])
axarr[0].set_title("True data")
axarr[1].scatter(*x.T.detach(), color=tab10.colors[1])
axarr[1].set_title("Generated data")
plt.show(block=False)
plt.savefig("2moons_goal.pdf")

# %%
n_plots = 8
fig, ax = plt.subplots(1, n_plots, sharex=True, sharey=True, figsize=(15, 3))

for idx, i in enumerate(np.linspace(0, n_steps - 1, n_plots, endpoint=True)):
    x = xt[int(i)]
    ax[idx].scatter(*x.detach().T, s=10, alpha=0.5)
    ax[idx].set_title(f't = {time_steps[int(i) + 1]:.2f}', fontsize=18)

for a in ax:
    a.set_aspect("equal")
    a.set_xlim(-3, 3)
    a.set_ylim(-3, 3)

plt.tight_layout()
plt.show(block=False)

plt.savefig("lab.pdf")
