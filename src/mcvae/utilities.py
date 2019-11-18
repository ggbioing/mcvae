import os
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from sklearn.preprocessing import StandardScaler

DEVICEID = 0
DEVICE = torch.device('cuda:'+str(DEVICEID) if torch.cuda.is_available() else 'cpu')


def experiment_to_namespace(experiment):
	pars = {}
	pars['experiment'] = experiment
	pars['generator_path'] = experiment.split('generator_')[0] + 'generator'
	pars['n_channels'] = int(experiment.split('/')[0].split('_')[-1])
	pars['n_feats'] = int(experiment.split('/')[1].split('_')[-1])
	pars['gen_lat_dim'] = int(experiment.split('/')[2].split('_')[-1])
	pars['gen_idx'] = int(experiment.split('/')[-2].split('_')[1])
	pars['N_obs'] = int(experiment.split('/')[-1].split('__')[1].split('_')[-1])
	snr_string = experiment.split('/')[-1].split('__')[2].split('_')[-1]
	try:
		snr = int(snr_string)
	except ValueError:
		snr = float(snr_string)
	pars['snr'] = snr
	depth = experiment.split('/')[-1].split('__')[3].split('_')[-1]
	if not depth == 'symmetric':
		depth = int(depth)
	pars['depth'] = depth
	pars['fit_lat_dim'] = int(experiment.split('/')[-1].split('__')[4].split('_')[-1])
	pars['current_gen'] = pars['generator_path'] + '_' + str(pars['gen_idx'])
	pars['train_dir'] = pars['current_gen'] + '_train'
	pars['generator'] = torch.load(pars['current_gen'] + '.pt')
	pars['generator_state_dict'] = torch.load(pars['current_gen'] + '.state_dict.pt')

	if not os.path.exists(pars['train_dir']):
		os.makedirs(pars['train_dir'])

	z_train = pars['train_dir'] + '/z_train_N_obs_' + str(pars['N_obs']) + '.txt'
	if os.path.exists(z_train):
		pars['z_train'] = np.loadtxt(z_train).reshape(pars['N_obs'], pars['generator'].lat_dim)
	else:
		np.random.seed(42)  # for the replicability of train data generation
		pars['z_train'] = np.random.randn(pars['N_obs'], pars['generator'].lat_dim)
		np.savetxt(z_train, pars['z_train'])

	#z_test = pars['train_dir'] + '/z_test_N_obs_' + str(pars['N_obs']) + '.txt'
	pars['N_test'] = 10000
	z_test = pars['train_dir'] + '/z_test_N_obs_' + str(pars['N_test']) + '.txt'
	if os.path.exists(z_test):
		#pars['z_test'] = np.loadtxt(z_test).reshape(pars['N_obs'], pars['generator'].lat_dim)
		pars['z_test'] = np.loadtxt(z_test).reshape(pars['N_test'], pars['generator'].lat_dim)
	else:
		np.random.seed(24)  # for the replicability of test data generation
		#pars['z_test'] = np.random.randn(pars['N_obs'], pars['generator'].lat_dim)
		pars['z_test'] = np.random.randn(pars['N_test'], pars['generator'].lat_dim)
		np.savetxt(z_test, pars['z_test'])

	return argparse.Namespace(**pars)


def gpu_info():
	if torch.cuda.is_available():
		info = {}
		gpuquery = 'nvidia-smi -q -d Utilization | grep Gpu'
		memquery = 'nvidia-smi -q -d Memory | grep -A3 FB | grep '

		ret = os.popen(gpuquery).read()
		info['gpu'] = [int(s) for s in ret.split() if s.isdigit()]

		ret = os.popen(memquery+'Total').read()
		info['mtotal'] = [int(s) for s in ret.split() if s.isdigit()]

		ret = os.popen(memquery+'Used').read()
		info['mused'] = [int(s) for s in ret.split() if s.isdigit()]

		ret = os.popen(memquery+'Free').read()
		info['mfree'] = [int(s) for s in ret.split() if s.isdigit()]

		return info
	else:
		print('No GPU found!')


def load_model(modeldir):
	#print("Loading " + modeldir)
	files = os.listdir(modeldir)
	files = [f for f in files if not f.endswith('dict.pt')]
	files.sort()
	epoch_file = files[-1]
	epoch_path = modeldir + '/' + epoch_file
	return torch.load(epoch_path, map_location=lambda storage, loc: storage)


def save_model(model, filename=None):
	"""
	Adapted from:
	https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3
	http://archive.is/2pnMw
	"""
	fn = model.model_name if filename is None else filename
	fn += '.pt'
	loss = model.loss['total']
	if str(loss[-1]) == 'nan':
		print("Loss is nan! Not saving.")
	elif loss[-1] > loss[0]:
		print("Loss diverged! Not saving.")
	else:
		state = {
			'state_dict': model.state_dict(),
			'optimizer': model.optimizer.state_dict(),
			'loss': model.loss,
			'epochs': model.epochs
		}
		print(f"Saving on {fn}")
		torch.save(state, fn)


def model2excel(model, data=None, filename=None):
	fn = model.model_name if filename is None else filename
	with pd.ExcelWriter(f'{fn}.xlsx') as writer:
		if data is not None:
			data.to_excel(writer, sheet_name='Data')
		for i, c in enumerate(model.ch_name):
			w = model.W_out[i].weight.detach().numpy().reshape(-1)
			w_names = model.varname[i].copy()
			pd.DataFrame(
				{'Variables': w_names, 'Weights': w}
			).to_excel(writer, sheet_name=c, index=None)


def ltonumpy(X):
	"""
	ltonumpy: short for "list_to_numpy"
	:param X: list of pytorch tensors
	:return: list of numpy arrays
	"""
	assert isinstance(X, list)
	assert len(X) > 0
	if isinstance(X[0], torch.Tensor):
		return [x.detach().numpy() for x in X]
	else:
		return [x.numpy() for x in X]


def ltotensor(X, device=DEVICE):
	"""

	:param X: list of numpy array or pytorch variables
	:return:
	"""
	assert isinstance(X, list)
	assert len(X) > 0
	ret = []
	for x in X:
		if isinstance(x, np.ndarray):
			ret.append(torch.FloatTensor(x).to(device))
		elif x is None:
			ret.append(None)
		else:
			raise ValueError('Cannot transform elemnt list to tensor')
	return ret


def modulate_aal(coefs, rois, modality='test', comp='', saveimg=False):
	aal = nib.load('/user/lantelmi/home/Software/MATLAB/spm12/templates/aal/ROI_MNI_V4.nii')
	img = np.zeros(aal.get_data().shape)
	for i, r in enumerate(rois):
		roi_name = r[1]
		roi_val = int(r[2])
		mask = np.where(aal.get_data() == roi_val)
		img[mask] = coefs[i]
	if saveimg:
		saveNifti(f'aal_{modality}_comp_{comp}.nii.gz', img, aal)
	return nib.Nifti1Image(img, aal.affine, aal.header)


def preprocess_and_add_noise(X, snr, seed=0, device=DEVICE):
	"""

	:param X: list of pytorch variables
	:param snr:
	:param seed:
	:return:
	"""
	if not isinstance(snr, list):
		SNR = [snr for _ in X]
	else:
		SNR = snr

	X_ = ltonumpy(X)
	FIT = [StandardScaler().fit(x) for x in X_]
	X_std_ = [FIT[i].transform(X_[i]) for i in range(len(X_))]
	X_std = ltotensor(X_std_)

	# seed for reproducibility in training/testing based on prime number basis
	seed = seed + 3 * int(SNR[0] + 1) + 5 * len(X_) + 7 * X_[0].shape[0] + 11 * X_[0].shape[1]
	np.random.seed(seed)

	X_std_noisy_ = []
	for c, x in enumerate(X_std_):
		sigma_noise = np.sqrt(1.0/SNR[c])
		X_std_noisy_.append(x + sigma_noise * np.random.randn(*x.shape))

	X_std_noisy = ltotensor(X_std_noisy_, device=device)
	return X_std, X_std_noisy


def rotation_matrix(theta):
	cost = np.cos(theta)
	sint = np.sin(theta)
	return np.array([[cost, -sint], [sint, cost]])


def saveNifti(filename, data, ref):
	img = nib.Nifti1Image(data, ref.affine, ref.header)
	nib.loadsave.save(img, filename)
	print(filename + " saved!")


def show_nifti_weights(model, ref, ch=0):
	w = model.W_out[ch].weight.detach().numpy()
	for d in range(w.shape[1]):
		img = nib.Nifti1Image(w[:,d].reshape(ref.shape), ref.affine, ref.header)
		nib.loadsave.save(img, f"{model.ch_name[ch]}_d{d}.nii.gz")


__all__ = [
	'experiment_to_namespace',
	'load_model',
	'ltonumpy',
	'ltotensor',
	'modulate_aal',
	'preprocess_and_add_noise',
	'saveNifti',
]