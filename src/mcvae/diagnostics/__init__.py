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


__all__ = [
	'plot_loss',
]