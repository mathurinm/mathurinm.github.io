# %%
import matplotlib.pyplot as plt
import deepinv
import torch

from tqdm import tqdm
from deepinv.utils.plotting import plot


device = "cuda" if torch.cuda.is_available() else "cpu"

denoiser = deepinv.models.DiffUNet(large_model=False).to(device)
denoiser.eval()

alphas_bar = (denoiser.get_alpha_prod()[-1]**2).to(device)
betas = torch.linspace(1e-4, 2e-2, 1000).to(device)  # todo get from denoiser
# %%
res = []
ts = [900, 700, 500, 300, 100, 0]

# how to use the network:
t = ts[2]
xt = torch.randn([1, 3, 256, 256], device=device)  # replace by a real xt
noise_pred = denoiser.forward_diffusion(
    xt, torch.tensor([t]).to(device))[:, :3]
# the network does noise prediction, it returns an estimation of the noise in xt
# it requires t to be passed (between 1 and 1000)
# beware that for diffusion we got from large t to small t to generate
# (p0 is the data distribution)

# TODO implement DDIM and DDPM, compare their results
rho = 0  # ddim scaling factor, 1 means DDPM
torch.manual_seed(0)
with torch.no_grad():
    xt = torch.randn([1, 3, 256, 256], device=device)
    for t in tqdm(reversed(range(0, 1000, skip))):
        at = alphas_bar[t]  # shorthand notation but THIS IS ALPHA BAR
        aprev = alphas_bar[t - 1] if t - 1 >= 0 else torch.tensor(1)
        bt = 1 - at / aprev
        noise_pred = denoiser.forward_diffusion(
            xt, torch.tensor([t]).to(device))[:, :3]
        x0 = ...
        xt = ...
        noise = torch.randn_like(xt)
        sigma_t = ...
        xt += sigma_t * noise
        if t in ts:
            res.append(xt.clone().detach())

# TODO: plot a few trajectories
