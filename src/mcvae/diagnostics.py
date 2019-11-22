import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
import itertools
import copy


def sigmoid(x, derivative=False):
	sigm = 1. / (1. + np.exp(-x))
	if derivative:
		return sigm * (1. - sigm)
	return sigm


def plot_hist(data, colnames=None):
	cols = data.shape[1]
	for i in range(cols):
		plt.subplot(1, cols, i + 1)
		plt.hist(data[:, i])
		if colnames is not None:
			plt.xlabel(colnames[i])


def plot_confusion_matrix(
		cm, classes,
		normalize=False,
		title='Confusion matrix',
		cmap=plt.cm.Blues,
		colorbar=False
):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	if colorbar:
		plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


def z_distrib(model, data):
	pred = model(data)

	def mu(ch, fwdreturn=pred):
		return fwdreturn['qzx'][ch]['mu'].detach().numpy()

	def sigma(ch, fwdreturn=pred):
		return np.exp(0.5*fwdreturn['qzx'][ch]['logvar'].detach().numpy())

	return mu, sigma


def plot_ellipses(model, data):
	channels = len(data)
	mu, sigma = z_distrib(model, data)
	fig, axs = plt.subplots(1)
	ells = [Ellipse(xy=[mu(0)[i][0], mu(1)[i][0]], width=2*sigma(0)[i][0], height=2*sigma(1)[i][0]) for i in range(len(mu(0)))]
	for e in ells:
		axs.add_artist(e)
		e.set_clip_box(axs.bbox)
		e.set_alpha(0.5)
		e.set_facecolor(np.random.rand(3))
	plt.show()


def plot_latent_space(model, data=None, classificator=None, text=None, all_plots=False, uncertainty=True, comp=None):
	if data is None:
		data = model.data
	channels = len(data)
	comps = model.lat_dim
	if comp is None:
		try:
			comp = model.kept_components
			print(f'Dropout threshold: {model.dropout_threshold}')
			print(f'Components kept: {comp}')
		except AttributeError:
			pass

	output = model(data)
	#zx = [output['qzx'][c]['mu'] for c in range(channels)]
	zx = output['zx']
	qzx = output['qzx']

	if classificator is not None:
		groups = np.unique(classificator)
		if not groups.dtype == np.dtype('O'):
			# remove nans if groups are not objects (strings)
			groups = groups[~np.isnan(groups)]

	# One figure per latent component
	#  Linear relationships expected between channels
	if comp is not None:
		itercomps = comp if isinstance(comp, list) else [comp]
	else:
		itercomps = range(comps)
	for comp in itercomps:
		fig, axs = plt.subplots(channels, channels)
		fig.suptitle(r'$z_{' + str(comp) + '}$', fontsize=30)
		for i, j in itertools.product(range(channels), range(channels)):
			ax = axs if channels == 1 else axs[j, i]
			if i == j:
				ax.text(
					0.5, 0.5, 'z|{}'.format(model.ch_name[i]),
					horizontalalignment='center', verticalalignment='center',
					fontsize=20
				)
				ax.axis('off')
			elif i > j:
				xi = qzx[i].loc.detach().numpy()[:, comp]
				xj = qzx[j].loc.detach().numpy()[:, comp]
				si = qzx[i].scale.detach().numpy()[:, comp]
				sj = qzx[j].scale.detach().numpy()[:, comp]
				ells = [Ellipse(xy=[xi[p], xj[p]], width=2 * si[p], height=2 * sj[p]) for p in range(len(xi))]
				if classificator is not None:
					for g in groups:
						g_idx = classificator == g
						ax.plot(xi[g_idx], xj[g_idx], '.', alpha=0.75, markersize=15)
						if uncertainty:
							color = ax.get_lines()[-1].get_color()
							for idx in np.where(g_idx)[0]:
								ax.add_artist(ells[idx])
								ells[idx].set_alpha(0.1)
								ells[idx].set_facecolor(color)
				else:
					ax.plot(xi, xj, '.')
					if uncertainty:
						for e in ells:
							ax.add_artist(e)
							e.set_alpha(0.1)
				if text is not None:
					[ax.text(*item) for item in zip(xi, xj, text)]
				# Bisettrice
				lox, hix = ax.get_xlim()
				loy, hiy = ax.get_ylim()
				lo, hi = np.min([lox, loy]), np.max([hix, hiy])
				ax.plot([lo, hi], [lo, hi], ls="--", c=".3")
			else:
				ax.axis('off')
		if classificator is not None:
			[axs[-1, 0].plot(0,0) for g in groups]
			legend = ['{} (n={})'.format(g, len(classificator[classificator==g])) for g in groups]
			axs[-1,0].legend(legend)
			try:
				axs[-1, 0].set_title(classificator.name)
			except AttributeError:
				axs[-1, 0].set_title('Groups')

	if all_plots:  # comps > 1:
		# TODO: remove based on components
		# One figure per channel
		#  Uncorrelated relationsips expected between latent components
		for ch in range(channels):
			fig, axs = plt.subplots(comps, comps)
			fig.suptitle(model.ch_name[ch], fontsize=30)
			for i, j in itertools.product(range(comps), range(comps)):
				if i == j:
					axs[j, i].text(
						0.5, 0.5, r'$z_{' + str(i) + '}$',
						horizontalalignment='center', verticalalignment='center',
						fontsize=20
					)
					axs[j, i].axis('off')
				elif i > j:
					xi = qzx[ch].loc.detach().numpy()[:, i]
					xj = qzx[ch].loc.detach().numpy()[:, j]
					if classificator is not None:
						for g in groups:
							g_idx = classificator == g
							axs[j, i].plot(xi[g_idx], xj[g_idx], '.')
					else:
						axs[j, i].plot(xi, xj, '.')
					# zero axis
					axs[j, i].axhline(y=0, ls="--", c=".3")
					axs[j, i].axvline(x=0, ls="--", c=".3")
				else:
					axs[j, i].axis('off')
			if classificator is not None:
				axs[0, -1].legend(groups)


def plot_noise_model(model, save_fig=False, ylim=None):
	epochs = len(model.loss['total'])
	channels = model.n_channels
	fig, axs = plt.subplots(channels, 1)
	fig.suptitle('Learned Variance (logvar units)\n{} epochs'.format(epochs))
	for ch in range(channels):
		noise = model.W_out_logvar[ch].data.numpy().T
		if len(noise) > 200:
			pass
		else:
			ax = axs if channels == 1 else axs[ch]
			ax.plot(noise, '.')
			ax.set_ylabel(model.ch_name[ch], rotation=0, fontsize=14)
			if model.varname is not None:
				tick_marks = np.arange(len(model.varname[ch]))
				ax.set_xticks(tick_marks)
				ax.set_xticklabels(model.varname[ch], rotation=70)
			if ylim is not None:
				ax.set_ylim(ylim)
	if save_fig:
		plt.savefig('noise_{}_epochs'.format(epochs))


def plot_loss(model, stop_at_convergence=True, save_fig=False):
	true_epochs = len(model.loss['total']) - 1
	losses = np.array([model.loss[key] for key in model.loss.keys()]).T
	plt.figure()
	try:
		plt.suptitle('Model ' + model.model_name + '\n')
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
	plt.plot(losses / (1e-8 + np.max(np.abs(losses), axis=0))), plt.legend(model.loss.keys())
	if not stop_at_convergence:
		plt.xlim([0, model.epochs])
	if save_fig:
		plt.savefig('loss_{}_epochs'.format(true_epochs))


def plot_loss_decimate(model, save_fig=False):
	true_epochs = len(model.loss['total']) - 1
	losses = [model.loss[key] for key in model.loss.keys()]
	losses = np.array(list(zip(*losses)))
	# Decimation (millification)
	dfactor = 1000
	losses = losses[::dfactor, :]
	plt.figure()
	try:
		plt.suptitle('Model ' + model.model_name + '\n')
	except:
		pass
	plt.subplot(1, 2, 1)
	plt.title('Loss (common scale)')
	plt.xlabel('epoch (x{})'.format(dfactor))
	plt.plot(losses), plt.legend(model.loss.keys())

	plt.subplot(1, 2, 2)
	plt.title('loss (relative scale)')
	plt.xlabel('epoch (x{})'.format(dfactor))
	plt.plot(losses / np.max(np.abs(losses), axis=0)), plt.legend(model.loss.keys())

	if save_fig:
		plt.savefig('loss_{}_epochs'.format(true_epochs))


def plot_weights(model, side='decoder', title='', xlabel='', comp=None, save_fig=False, rotation=15):

	if comp is None:
		try:
			comp = model.kept_components
			print(f'Dropout threshold: {model.dropout_threshold}')
			print(f'Components kept: {comp}')
		except AttributeError:
			pass

	if comp is None:
		suptitle = 'Model Weights\n({})'.format(side)
		comp = list(range(model.lat_dim))
	else:
		suptitle = 'Model Weights\n({}, comp. {})'.format(side, comp)
		comp = comp if isinstance(comp, list) else [comp]

	fig, axs = plt.subplots(model.n_channels, 1)

	plt.suptitle(title + suptitle)
	for ch in range(model.n_channels):
		ax = axs if model.n_channels == 1 else axs[ch]  # 'AxesSubplot' object does not support indexing
		x = np.arange(model.n_feats[ch])
		if side == 'encoder':
			y = model.W_mu[ch].weight.detach().numpy().T.copy()
		else:
			y = model.W_out[ch].weight.detach().numpy().copy()
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
			if model.lat_dim == 1 or len(comp) == 1:
				ax.bar(x, y.reshape(-1), width=0.25)
			else:
				ax.plot(x, y)
			ax.set_ylabel(model.ch_name[ch], rotation=45, fontsize=14)
			if comp is None:
				ax.legend(['comp. '+str(c) for c in range(model.lat_dim)])
			else:
				ax.legend(['comp. ' + str(c) for c in comp])
			ax.axhline(y=0, ls="--", c=".3")
			if model.varname is not None:
				tick_marks = np.arange(len(model.varname[ch]))
				ax.set_xticks(tick_marks)
				ax.set_xticklabels(model.varname[ch], rotation=rotation, fontsize=20)
			plt.xlabel(xlabel)
	if save_fig:
		plt.savefig('model_weights')