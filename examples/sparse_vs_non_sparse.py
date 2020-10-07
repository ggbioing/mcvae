#!/usr/bin/env python
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from mcvae.datasets import SyntheticDataset
from mcvae.utilities import preprocess_and_add_noise
from mcvae.models import Mcvae
from mcvae.models.utils import DEVICE, load_or_fit
from mcvae.diagnostics import plot_loss

# from mcvae import pytorch_modules, utilities, preprocessing, plot, diagnostics


print(f"Running on {DEVICE}")

Nobs = 500
n_channels = 3
n_feats = 4
true_lat_dims = 2
fit_lat_dims = 5
snr = 5

ds = SyntheticDataset(
    n=Nobs,
    lat_dim=true_lat_dims,
    n_feats=n_feats,
    n_channels=n_channels,
)
x_ = ds.x

x, x_noisy = preprocess_and_add_noise(x_, snr=snr)

# Send to GPU (if possible)
X = [c.to(DEVICE) for c in x] if torch.cuda.is_available() else x

###################
## Model Fitting ##
###################
adam_lr = 1e-2
epochs = 20000

models = {}

# Multi-Channel VAE
torch.manual_seed(24)
models['mcvae'] = Mcvae(data=X, lat_dim=fit_lat_dims)

# Sparse Multi-Channel VAE
torch.manual_seed(24)
models['smcvae'] = Mcvae(data=X, lat_dim=fit_lat_dims, sparse=True)

for model_name, model in models.items():

    model.optimizer = torch.optim.Adam(params=model.parameters(), lr=adam_lr)
    model.to(DEVICE)

    ptfile = Path(model_name + '.pt')

    load_or_fit(model, model.data, epochs, ptfile, force_fit=True)

# Output of the models
pred = {}  # Prediction
z = {}     # Latent Space
g = {}     # Generative Parameters
x_hat = {}  # reconstructed channels

for model_name, model in models.items():
    m = model_name
    # plot_loss(model)
    q = model.encode(X)  # encoded distribution q(z|x)
    z[m] = [q[i].mean.detach().numpy() for i in range(n_channels)]
    z[m] = np.array(z[m]).reshape(-1)  # flatten
    x_hat[m] = model.reconstruct(X, dropout_threshold=0.2)
    g[m] = [model.vae[i].W_out.weight.detach().numpy() for i in range(n_channels)]
    g[m] = np.array(g[m]).reshape(-1)  #flatten

# plot.lsplom(utilities.ltonumpy(x), title=f'Ground truth')
# plot.lsplom(utilities.ltonumpy(x_noisy), title=f'Noisy data fitted by the models (snr={snr})')
# for m in model.keys():
#     plot.lsplom(utilities.ltonumpy(x_hat[m]), title=f'Reconstructed with {m} model')

plt.figure()
plt.subplot(1,2,1)
plt.hist([z['smcvae'], z['mcvae']], bins=50, color=['k', 'gray'])
plt.legend(['Sarse', 'Non sparse'])
plt.title(r'Latent dimensions distribution')
plt.ylabel('Count')
plt.xlabel('Value')
plt.subplot(1,2,2)
plt.hist([g['smcvae'], g['mcvae']], bins=50, color=['k', 'gray'])
plt.legend(['Sparse', 'Non sparse'])
plt.title(r'Generative parameters $\mathbf{\theta} = \{\mathbf{\theta}_1 \ldots \mathbf{\theta}_C\}$')
plt.xlabel('Value')


# Show dropout effect
do = np.sort(models['smcvae'].dropout.detach().numpy().reshape(-1))
plt.figure()
plt.bar(range(len(do)), do)
plt.suptitle(f'Dropout probability of {fit_lat_dims} fitted latent dimensions in Sparse Model')
plt.title(f'({true_lat_dims} true latent dimensions)')

plt.show()
print("See you!")
