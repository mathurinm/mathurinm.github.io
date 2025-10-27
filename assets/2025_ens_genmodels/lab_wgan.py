# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.autograd as autograd
# %%
nb_samples = 10000
radius = 1
nz = .1
# generate the data
X_train = torch.zeros((nb_samples, 2))
r = radius + nz*torch.randn(nb_samples)
theta = torch.rand(nb_samples)*2*torch.pi
X_train[:, 0] = r*torch.cos(theta)
X_train[:, 1] = r*torch.sin(theta)

plt.figure(figsize=(6, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], s=20, alpha=0.8, edgecolor='k', marker='o', label='original samples')
plt.grid(alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()
plt.show()
# %%


def generate_images(generator_model, noise_dim, num_samples=1000):
    with torch.no_grad():
        z = torch.Tensor(np.random.normal(0, 1, (num_samples, noise_dim))).type(torch.float32)
        predicted_samples = generator_model(z)
    plt.figure(figsize=(6, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], s=40, alpha=0.2, edgecolor='k', marker='+', label='original samples')
    plt.scatter(predicted_samples[:, 0], predicted_samples[:, 1], s=10,
                alpha=0.9, c='r', edgecolor='k', marker='o', label='predicted')
    plt.grid(alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


class MyNeuralNet(nn.Module):

    # neede for pytorch
    def forward(self, x):
        pass


class Generator(nn.Module):
    def __init__(self, noise_dim=10):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

    def forward(self, z):
        output = nn.Sequential(
            nn.Linear(self.noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)


# %%
# Initialize generator and discriminator
noise_dim = 2
# neural net
generator = Generator(noise_dim=noise_dim)
discriminator = Discriminator()

# Optimizers
lr_G = 0.001
lr_D = 0.001
n_epochs = 500  # 500
clip_value = 0.3
n_critic = 5
batch_size = 128
# optimizer for G
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.9))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.9))
dataloader = DataLoader(X_train, batch_size, shuffle=True)
batches_done = 0
for epoch in range(n_epochs):
    for i, x in enumerate(dataloader):
        # x is a batch of data:
        # x.shape = (batch_size, data_dim)
        # Configure input
        x = x.type(torch.float32)
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Sample noise as generator input
        # batch_size = x.shape[0]
        z =  #TODO

        # Generate a batch of images
        fake_x =  #TODO
        # Adversarial loss
        loss_D =  #TODO

        loss_D.backward()  # computes the gradient
        optimizer_D.step() # do gradient step

        # Clip weights of discriminator
        for p in discriminator.parameters():
            #TODO

        # Train the generator every n_critic iterations
        if i % n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            fake_x = #TODO
            # Adversarial loss
            loss_G = #TODO

            loss_G.backward()
            optimizer_G.step()

        batches_done += 1

    # Visualization of intermediate results
    if epoch % 10 == 0:
        print("Epoch: ", epoch)
        generate_images(generator, noise_dim)

# %% With gradient penalty


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = #TODO
    d_interpolates = #TODO
    # Get gradient w.r.t. interpolates
    gradients = #TODO
    gradient_penalty = #TODO
    return gradient_penalty
# %%


# Initialize generator and discriminator
noise_dim = 2
generator = Generator(noise_dim=noise_dim)
discriminator = Discriminator()

# Optimizers
lr_G = 0.0001
lr_D = 0.001
n_epochs = 500  # 500
lambda_gp = 1.0
n_critic = 5
batch_size = 128
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.9))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.9))
dataloader = DataLoader(X_train, batch_size, shuffle=True)
batches_done = 0
for epoch in range(n_epochs):
    for i, x in enumerate(dataloader):
        # Configure input
        x = x.type(torch.float32)
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = #TODO

        # Generate a batch of images
        fake_x = #TODO
        # Adversarial loss
        gradient_penalty = compute_gradient_penalty(discriminator, x, fake_x)
        loss_D = #TODO

        loss_D.backward()
        optimizer_D.step()


        # Train the generator every n_critic iterations
        if i % n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            fake_x = #TODO
            # Adversarial loss
            loss_G = #TODO

            loss_G.backward()
            optimizer_G.step()

        batches_done += 1

    # Visualization of intermediate results
    if epoch % 10 == 0:
        print("Epoch: ", epoch)
        generate_images(generator, noise_dim)

# %%
