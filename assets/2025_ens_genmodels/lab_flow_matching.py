"""
1. Plot the training data.
2. In FM init, define self.net to be a MLP with Relu activation function.
3. Define the forward method of `FM` to call self.net on (xt, t)
4. Define the `step` method to perform one step of Euler method for solving the
    ODE with step size `delta`.
5. Complete the training loop.
6. Generate samples with the trained model; plot them.
7. Plot the distribution $p_t$ of the solution of the ODE at time $t$ for a few
   values of $t$.
8. Retrain the model by using only a small number of training samples. What do
   you observe?
"""
# %%
import torch
from sklearn.datasets import make_moons
from torch import nn, Tensor
import matplotlib.pyplot as plt


target = make_moons(256, noise=0.05)[0]
# %%


class FM(nn.Module):
    def __init__(self, dim=2, h=64):
        super().__init__()
        self.net = nn.Sequential(
            ...
        )
        # TODO

    def forward(self, x_t, t):
        input = ...
        # TODO

    def step(self, x_t, delta):
        """1 step of Euler ODE solver"""
        x_tp1 = ...
        # TODO


# %%
model = FM()
optimizer = torch.optim.Adam(model.parameters(), 1e-2)
loss_fn = nn.MSELoss()

n_epochs = 1000
for _ in range(n_epochs):
    pass
    # TODO
