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
        return x_t + delta * self(x_t, t_start)


# %% network training
flow = Flow()
optimizer = torch.optim.Adam(flow.parameters(), 1e-2)
loss_fn = nn.MSELoss()  # squared L2 loss for network training

for it in range(10_000):
    if it % 500 == 0:
        print(it)
    x_1 = Tensor(make_moons(256, noise=0.05)[0])
    x_0 = torch.randn_like(x_1)
    t = torch.rand(len(x_1), 1)
    x_t = (1 - t) * x_0 + t * x_1
    optimizer.zero_grad()
    ut_xt = flow(x_t, t)
    loss = loss_fn(ut_xt, x_1 - x_0)
    loss.backward()
    optimizer.step()

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
