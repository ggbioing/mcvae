#!/usr/bin/env python
import os
import numpy as np
import torch
import ggpy.pytorch_modules
import ggpy.utilities
import ggpy.preprocessing
import matplotlib.pyplot as plt


DEVICE = ggpy.pytorch_modules.DEVICE
print(f"Running on {DEVICE}")

Nobs = 500
n_channels = 3
n_feats = 4
true_lat_dims = 2
fit_lat_dims = 5


np.random.seed(7)
z = np.random.randn(Nobs, true_lat_dims)
z_test = np.random.randn(Nobs, true_lat_dims)

generator = ggpy.pytorch_modules.ScenarioGenerator(
    lat_dim=true_lat_dims,
    n_channels=n_channels,
    n_feats=n_feats,
)

preprocpars = {'remove_mean': True, 'normalize': True, 'whitening': False}

x_ = generator(z)
x = ggpy.utilities.ltotensor(
    ggpy.preprocessing.preprocess(x_, **preprocpars)
)
# Send to GPU (if possible)
X = [c.to(DEVICE) for c in x] if torch.cuda.is_available() else x

x_test_ = generator(z_test)
x_test = ggpy.utilities.ltotensor(
    ggpy.preprocessing.preprocess(x_test_, **preprocpars)
)
X_test = [c.to(DEVICE) for c in x_test] if torch.cuda.is_available() else x_test

###################
## Model Fitting ##
###################
init_dict = {
    'n_channels': len(x),
    'lat_dim': fit_lat_dims,
    'n_feats': tuple([i.shape[1] for i in X])
}

adam_lr = 1e-3
n_epochs = 20000


# Multi-Channel VAE
torch.manual_seed(24)
mcvae = ggpy.pytorch_modules.MultiChannelBase(
    **init_dict,
    model_name_dict={**init_dict, 'adam_lr': adam_lr},
)
mcvae.to(DEVICE)

modelpath = mcvae.model_name + '.pt'
if os.path.exists(modelpath):
    print(f"Loading {modelpath}")
    mdict = torch.load(modelpath, map_location=DEVICE)
    mcvae.load_state_dict(mdict['state_dict'])
    mcvae.optimizer = torch.optim.Adam(mcvae.parameters())
    mcvae.optimizer.load_state_dict(mdict['optimizer'])
    mcvae.loss = mdict['loss']
    mcvae.eval()
    del mdict
else:
    print(f"Fitting {modelpath}")
    mcvae.init_loss()
    mcvae.optimizer = torch.optim.Adam(mcvae.parameters(), lr=adam_lr)
    mcvae.optimize(epochs=n_epochs, data=X)
    print("Refine optimization...")
    for pg in mcvae.optimizer.param_groups:
        pg['lr'] *= 0.1
    mcvae.optimize(epochs=n_epochs, data=X)
    ggpy.utilities.save_model(mcvae)

# Sparse Multi-Channel VAE
torch.manual_seed(24)
smcvae = ggpy.pytorch_modules.MultiChannelSparseVAE(
    **init_dict,
    model_name_dict={**init_dict, 'adam_lr': adam_lr},
)
smcvae.to(DEVICE)

modelpath = smcvae.model_name + '.pt'
if os.path.exists(modelpath):
    print(f"Loading {modelpath}")
    mdict = torch.load(modelpath, map_location=DEVICE)
    smcvae.load_state_dict(mdict['state_dict'])
    smcvae.optimizer = torch.optim.Adam(smcvae.parameters())
    smcvae.optimizer.load_state_dict(mdict['optimizer'])
    smcvae.loss = mdict['loss']
    smcvae.eval()
    del mdict
else:
    print(f"Fitting {modelpath}")
    smcvae.init_loss()
    smcvae.optimizer = torch.optim.Adam(smcvae.parameters(), lr=adam_lr)
    smcvae.optimize(epochs=n_epochs, data=X)
    print("Refine optimization...")
    for pg in smcvae.optimizer.param_groups:
        pg['lr'] *= 0.1
    smcvae.optimize(epochs=n_epochs, data=X)
    ggpy.utilities.save_model(smcvae)


# Output of the models
pred = mcvae(X)
pred_s = smcvae(X)

# Latent spaces
z = np.array([pred['qzx'][i]['mu'].detach().numpy() for i in range(n_channels)]).reshape(-1)
zs = np.array([pred_s['qzx'][i]['mu'].detach().numpy() for i in range(n_channels)]).reshape(-1)

# Generative parameters
g = np.array([mcvae.W_out[i].weight.detach().numpy() for i in range(n_channels)]).reshape(-1)
gs = np.array([smcvae.W_out[i].weight.detach().numpy() for i in range(n_channels)]).reshape(-1)

plt.figure()
plt.subplot(1,2,1)
plt.hist([zs, z], bins=20, color=['k', 'gray'])
plt.legend(['Sparse', 'Non sparse'])
plt.title(r'Latent dimensions distribution')
plt.ylabel('Count')
plt.xlabel('Value')
plt.subplot(1,2,2)
plt.hist([gs, g], bins=20, color=['k', 'gray'])
plt.legend(['Sparse', 'Non sparse'])
plt.title(r'Generative parameters $\mathbf{\theta} = \{\mathbf{\theta}_1 \ldots \mathbf{\theta}_4\}$')
plt.xlabel('Value')


# Show dropout effect
do = np.sort(smcvae.dropout.detach().numpy().reshape(-1))
plt.figure()
plt.bar(range(len(do)), do)
plt.suptitle(f'Dropout probability of {fit_lat_dims} fitted latent dimensions in Sparse Model')
plt.title(f'({true_lat_dims} true latent dimensions)')

print("See you!")