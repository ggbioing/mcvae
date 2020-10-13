import numpy as np
import matplotlib.pylab as plt


def plot_loss(model, stop_at_convergence=True, fig_path=None, skip=0):
	true_epochs = len(model.loss['total']) - 1
	if skip	> 0:
		print(f'skipping first {skip} epochs where losses might be very high')
	losses = np.array([model.loss[key][skip:] for key in model.loss.keys()]).T
	fig = plt.figure()
	try:
		plt.suptitle('Model ' + str(model.model_name) + '\n')
	except:
		pass
	plt.subplot(1, 2, 1)
	plt.title('Loss (common scale)')
	plt.xlabel('epoch')
	plt.plot(losses), plt.legend(model.loss.keys())
	if not stop_at_convergence:
		plt.xlim([0, model.epochs])
	plt.subplot(1, 2, 2)
	plt.title('loss (relative scale)')
	plt.xlabel('epoch')
	max_losses = 1e-8 + np.max(np.abs(losses), axis=0)
	plt.plot(losses / max_losses), plt.legend(model.loss.keys())
	if not stop_at_convergence:
		plt.xlim([0, model.epochs])

	if fig_path is not None:
		# pickle.dump(fig, open(fig_path + '.pkl', 'wb'))
		plt.rcParams['figure.figsize'] = (8, 5)
		plt.savefig(f'{fig_path}.png', bbox_inches='tight')
		plt.close()


def plot_weights(model, side='decoder', title = '', xlabel='', comp=None, rotation=15, fig_path=None):
	try:
		model.n_channels = model.n_datasets
	except:
		pass
	fig, axs = plt.subplots(model.n_channels, 1)
	if comp is None:
		suptitle = 'Model Weights\n({})'.format(side)
	else:
		suptitle = 'Model Weights\n({}, comp. {})'.format(side, comp)
	plt.suptitle(title + suptitle)
	for ch in range(model.n_channels):
		ax = axs if model.n_channels == 1 else axs[ch]  # 'AxesSubplot' object does not support indexing
		x = np.arange(model.n_feats[ch])
		if side == 'encoder':
			y = model.vae[ch].W_mu.weight.detach().numpy().T.copy()
		else:
			y = model.vae[ch].W_out.weight.detach().numpy().copy()
			try:  # In case of bernoulli features
				if model.bern_feats is not None:
					bidx = model.bern_feats[ch]
					y[bidx, :] = sigmoid(y[bidx, :])
			except:
				pass
		if y.shape[0] > 200:
			pass
		else:
			if comp is not None:
				y = y[:, comp]
			# axs[ch].plot(y)
			if model.lat_dim == 1 or comp is not None:
				ax.bar(x, y.reshape(-1), width=0.25)
			else:
				ax.plot(x, y)
			ax.set_ylabel(model.ch_name[ch], rotation=45, fontsize=14)
			if comp is None:
				ax.legend(['comp. '+str(c) for c in range(model.lat_dim)])
			ax.axhline(y=0, ls="--", c=".3")
			try:
				tick_marks = np.arange(len(model.varname[ch]))
				ax.set_xticks(tick_marks)
				ax.set_xticklabels(model.varname[ch], rotation=rotation, fontsize=20)
			except:
				pass
			plt.xlabel(xlabel)

	if fig_path is not None:
		pickle.dump(fig, open(fig_path+'.pkl', 'wb'))
		fsch = 4 * model.n_channels
		fsfeat = 3 * max(model.n_feats)
		plt.rcParams['figure.figsize'] = (fsfeat, fsch)
		plt.tight_layout()
		plt.savefig(fig_path, bbox_inches='tight')
		plt.close()


__all__ = [
	'plot_loss',
]