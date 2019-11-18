#!/usr/bin/env python
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
sys.path.append(os.getcwd() + '/../src')
from mcvae import pytorch_modules, utilities, preprocessing, plot, diagnostics


DEVICE = pytorch_modules.DEVICE
print(f"Running on {DEVICE}")

Nobs = 500
n_channels = 3
n_feats = 4
true_lat_dims = 2
fit_lat_dims = 5
snr=10

np.random.seed(7)
z = np.random.randn(Nobs, true_lat_dims)

generator = pytorch_modules.ScenarioGenerator(
    lat_dim=true_lat_dims,
    n_channels=n_channels,
    n_feats=n_feats,
)

x_ = generator(z)
x, x_noisy = utilities.preprocess_and_add_noise(x_, snr=snr)

# Send to GPU (if possible)
X = [c.to(DEVICE) for c in x] if torch.cuda.is_available() else x

###################
## Model Fitting ##
###################
init_dict = {
    'n_channels': len(x),
    'lat_dim': fit_lat_dims,
    'n_feats': tuple([i.shape[1] for i in X]),
}

adam_lr = 1e-3
n_epochs = 20000

model = {}

# Multi-Channel VAE
torch.manual_seed(24)
model['mcvae'] = pytorch_modules.MultiChannelBase(
    **init_dict,
    model_name_dict={**init_dict, 'adam_lr': adam_lr, 'snr': snr},
)

# Sparse Multi-Channel VAE
torch.manual_seed(24)
model['smcvae'] = pytorch_modules.MultiChannelSparseVAE(
    **init_dict,
    model_name_dict={**init_dict, 'adam_lr': adam_lr, 'snr': snr},
)

for current_model in model.keys():

    model[current_model].to(DEVICE)

    modelpath = model[current_model].model_name + '.pt'
    if os.path.exists(modelpath):
        print(f"Loading {modelpath}")
        mdict = torch.load(modelpath, map_location=DEVICE)
        model[current_model].load_state_dict(mdict['state_dict'])
        model[current_model].optimizer = torch.optim.Adam(model[current_model].parameters())
        model[current_model].optimizer.load_state_dict(mdict['optimizer'])
        model[current_model].loss = mdict['loss']
        model[current_model].eval()
        del mdict
    else:
        print(f"Fitting {modelpath}")
        model[current_model].init_loss()
        model[current_model].optimizer = torch.optim.Adam(model[current_model].parameters(), lr=adam_lr)
        model[current_model].optimize(epochs=n_epochs, data=X)
        print("Refine optimization...")
        for pg in model[current_model].optimizer.param_groups:
            pg['lr'] *= 0.1
        model[current_model].optimize(epochs=n_epochs, data=X)
        utilities.save_model(model[current_model])

# Output of the models
pred = {}  # Prediction
z = {}     # Latent Space
g = {}     # Generative Parameters
x_hat = {}  # reconstructed channels

for m in model.keys():
    diagnostics.plot_loss(model[m])
    pred[m] = model[m](X)
    x_hat[m] = model[m].reconstruct(X)
    z[m] = np.array([pred[m]['qzx'][i].loc.detach().numpy() for i in range(n_channels)]).reshape(-1)
    g[m] = np.array([model[m].W_out[i].weight.detach().numpy() for i in range(n_channels)]).reshape(-1)

plot.lsplom(utilities.ltonumpy(x), title=f'Ground truth')
plot.lsplom(utilities.ltonumpy(x_noisy), title=f'Noisy data fitted by the models (snr={snr})')
for m in model.keys():
    plot.lsplom(utilities.ltonumpy(x_hat[m]), title=f'Reconstructed with {m} model')

plt.figure()
plt.subplot(1,2,1)
plt.hist([z['smcvae'], z['mcvae']], bins=20, color=['k', 'gray'])
plt.legend(['Sparse', 'Non sparse'])
plt.title(r'Latent dimensions distribution')
plt.ylabel('Count')
plt.xlabel('Value')
plt.subplot(1,2,2)
plt.hist([g['smcvae'], g['mcvae']], bins=20, color=['k', 'gray'])
plt.legend(['Sparse', 'Non sparse'])
plt.title(r'Generative parameters $\mathbf{\theta} = \{\mathbf{\theta}_1 \ldots \mathbf{\theta}_C\}$')
plt.xlabel('Value')


# Show dropout effect
do = np.sort(model['smcvae'].dropout.detach().numpy().reshape(-1))
plt.figure()
plt.bar(range(len(do)), do)
plt.suptitle(f'Dropout probability of {fit_lat_dims} fitted latent dimensions in Sparse Model')
plt.title(f'({true_lat_dims} true latent dimensions)')

plt.show()
print("See you!")
