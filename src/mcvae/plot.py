import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from matplotlib.ticker import NullFormatter
from matplotlib.gridspec import GridSpec
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import itertools

# Plotting functions
# set the colormap and centre the colorbar
class MidpointNormalize(matplotlib.colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	see: http://chris35wills.github.io/matplotlib_diverging_colorbar/
	"""

	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def plot_img(img, data=None, anat=False):
	"""
	Plot slices at cut_coords = (0,0,0)
	"""
	if isinstance(img, dict):
		plot_img([v for k, v in img['imaging'].items()])
	elif isinstance(img, list):
		[plot_img(img[i]) for i in range(len(img))],
	else:
		assert isinstance(img, nib.nifti1.Nifti1Image)
		if data is not None:
			img = nib.Nifti1Image(data.reshape(img.shape), img.affine, img.header)
		myplot = plotting.plot_anat if anat else plotting.plot_img
		myplot(img, title=img.get_filename(), cut_coords=(0,0,0), threshold=None, colorbar=True)



def show(image, title=' ', show_now=False, save_fig=False):
	# assert len(image.shape)==2
	f, ax = plt.subplots()
	im = ax.imshow(
		image,
		plt.get_cmap('coolwarm'),
		norm=MidpointNormalize(midpoint=0, vmin=-np.abs(image).max(), vmax=np.abs(image).max())
	)
	ax.set_title(title)
	f.colorbar(im) if image.shape[0] >= image.shape[1] else f.colorbar(im, orientation="horizontal")
	if save_fig:
		f.savefig(title.replace(' ', '_'))
	if show_now:
		plt.show()


def plot1(x, title=' ', show_now=False, save_fig=False):
	plt.figure()
	plt.plot(x, '.')
	plt.title(title)
	if save_fig:
		plt.savefig(title.replace(' ', '_'))
	if show_now:
		plt.show()


def plot2(x, y, title=' ', classificator=None, show_now=False, save_fig=False):
	plt.figure()
	if classificator is not None:
		groups = np.unique(classificator)
		groups = groups[~np.isnan(groups)]
		for g in groups:
			g_idx = classificator == g
			plt.plot(x[g_idx], y[g_idx], '.', alpha=0.55, markersize=10)
		legend = ['{} (n={})'.format(g, len(classificator[classificator == g])) for g in groups]
		plt.legend(legend, title=classificator.name)
	else:
		plt.plot(x.reshape(-1), y.reshape(-1), '.', markersize=10)
	plt.title(title)
	#plt.axis('equal')
	plt.axhline(y=0, color='k')
	plt.axvline(x=0, color='k')
	if save_fig:
		plt.savefig(title.replace(' ', '_'))
	if show_now:
		plt.show()


def side_distrib(x, y, bins=None, classificator=None, xlabel=None, ylabel=None, show_now=False):

	if classificator is not None:
		groups = np.unique(classificator)
		if not groups.dtype == np.dtype('O'):
			# remove nans if groups are not objects (strings)
			groups = groups[~np.isnan(groups)]
		X = [x[classificator == g] for g in groups]
		Y = [y[classificator == g] for g in groups]
		legend = ['{} (n={})'.format(g, len(classificator[classificator == g])) for g in groups]
	else:
		X = [x]
		Y = [y]

	nullfmt = NullFormatter()  # no labels

	# definitions for the axes
	left, width = 0.1, 0.65
	bottom, height = 0.1, 0.65
	bottom_h = left_h = left + width + 0.02

	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom_h, width, 0.2]
	rect_histy = [left_h, bottom, 0.2, height]
	rect_legend = [left_h, bottom_h, 0.2, 0.2]

	# start with a rectangular Figure
	plt.figure(figsize=(8, 8))

	axScatter = plt.axes(rect_scatter)
	axHistx = plt.axes(rect_histx, sharex=axScatter)
	axHisty = plt.axes(rect_histy, sharey=axScatter)
	axLegend = plt.axes(rect_legend)

	# no labels
	#axHistx.xaxis.set_major_formatter(nullfmt)
	#axHisty.yaxis.set_major_formatter(nullfmt)

	# now determine nice limits by hand:
	binwidth = 0.25
	xymax = np.max([np.max(np.fabs(np.concatenate(X))), np.max(np.fabs(np.concatenate(Y)))])
	lim = (int(xymax / binwidth) + 1) * binwidth
	if bins is None:
		bins = np.arange(-lim, lim + binwidth, binwidth)

	for x, y in zip(X, Y):

		# the scatter plot:
		axScatter.scatter(x, y, alpha=0.5)
		axHistx.hist(x, alpha=0.5, bins=bins)
		axHisty.hist(y, alpha=0.5, bins=bins, orientation='horizontal')

	axLegend.axis('off')
	if classificator is not None:
		[axLegend.plot(0, 0) for g in groups]
		axLegend.legend(legend, title=classificator.name)
	if xlabel is not None:
		axScatter.set_xlabel(xlabel)
	if ylabel is not None:
		axScatter.set_ylabel(ylabel)

	axScatter.set_xlim((-lim, lim))
	axScatter.set_ylim((-lim, lim))

	axHistx.set_xlim(axScatter.get_xlim())
	axHisty.set_ylim(axScatter.get_ylim())

	axScatter._shared_x_axes

	if show_now:
		plt.show()


def splom(X, title=' ', colprefix='col.', show_now=False, save_fig=False):
	if type(X) is list:
		for i, x in enumerate(X):
			splom(x, title=title + str(i), colprefix=colprefix)
	else:
		features = X.shape[1]
		f, ax = plt.subplots(features, features, sharex=True, sharey=True)
		f.suptitle(title)
		for i, j in itertools.product(range(features), range(features)):
			ax_ = ax[i, j]
			if i == j:
				if True:
					ax_.text(
						0.5, 0.5, colprefix + str(i),
						horizontalalignment='center', verticalalignment='center',
						fontsize=20
					)
					ax_.axis('off')
				else:
					x = X[:, i]
					nbins = np.int(np.floor(len(x) / 10))
					ax_.hist(x, nbins)
			if i > j:
				ax_.plot(X[:, j], X[:, i], '.')
				# ax_.plot(X[0:2, j], X[0:2, i], 'r-', markersize=2)
				# ax_.plot(X[1:3, j], X[1:3, i], 'g-', markersize=2)
				# ax_.plot(X[2:4, j], X[2:4, i], 'k-', markersize=2)
				# zero axis
				ax_.axhline(y=0, ls="--", c=".3")
				ax_.axvline(x=0, ls="--", c=".3")
				ax_.axis('equal')
				if j == 0:
					pass
				# ax_.set_ylabel("{}  ".format(i), rotation='0')
				if i == features - 1:
					pass
				# ax_.set_xlabel("{}".format(j))
			else:
				ax_.axis('off')
		# f.subplots_adjust(hspace=0)
		# hide ticks
		# plt.setp([a.get_xticklabels() for a in f.axes], visible=False)
		# plt.setp([a.get_yticklabels() for a in f.axes], visible=False)
		if save_fig:
			f.savefig(title.replace(' ', '_'))
		if show_now:
			plt.show()


def lsplom(X, ax_ref=None, title='', names=None):
	assert type(X) == list

	if ax_ref is None:
		ax_ref = X
	if names is None:
		names = ['Ch.{}'.format(i) for i in range(len(X))]

	def fix_axes(fig, ax_ref):
		min = np.min([x.min() for x in ax_ref])
		max = np.max([x.max() for x in ax_ref])
		for i, ax in enumerate(fig.axes):
			# ax.text(0, 0, "ax%d" % (i+1), va="center", ha="center")
			ax.set_xlim([min, max])
			ax.set_ylim([min, max])
			# Hide the right and top spines
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.spines['left'].set_visible(False)
			ax.spines['bottom'].set_visible(False)
			# Only show ticks on the left and bottom spines
			ax.yaxis.set_ticks_position('left')
			ax.xaxis.set_ticks_position('bottom')
			for tl in ax.get_xticklabels() + ax.get_yticklabels():
				# tl.set_visible(False)
				pass

	N = len(X)
	f = plt.figure()
	plt.suptitle(title, fontsize=18)
	for n, x in enumerate(X):
		features = x.shape[1]
		gs = GridSpec(features, features)
		gs.update(left=n / N + 0.05, right=(n + 1) / N, wspace=0.05)
		for i, j in itertools.product(range(features), range(features)):
			plt.subplot(gs[i, j])
			if i > j:
				plt.plot(x[:, j], x[:, i], '.')
				if False:
					plt.plot(x[0:2, j], x[0:2, i], 'r-', markersize=2)
					plt.plot(x[1:3, j], x[1:3, i], 'g-', markersize=2)
					plt.plot(x[2:4, j], x[2:4, i], 'k-', markersize=2)
				# zero axis
				plt.axhline(y=0, ls="--", c=".3")
				plt.axvline(x=0, ls="--", c=".3")
			elif i == j:
				if i == 0:
					plt.title(names[n], fontsize=22)
				plt.text(
					0, 0, 'd.' + str(i),
					horizontalalignment='center', verticalalignment='center',
					fontsize=18
				)
				plt.axis('off')
			else:
				plt.axis('off')
	fix_axes(f, ax_ref)


def test():
	def make_ticklabels_invisible(fig):
		for i, ax in enumerate(fig.axes):
			ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
			for tl in ax.get_xticklabels() + ax.get_yticklabels():
				tl.set_visible(False)

	# gridspec with subplotpars set.

	f = plt.figure()

	plt.suptitle("GridSpec w/ different subplotpars")

	gs1 = GridSpec(3, 3)
	gs1.update(left=0.05, right=0.48, wspace=0.05)
	plt.subplot(gs1[:-1, :])
	plt.subplot(gs1[-1, :-1])
	plt.subplot(gs1[-1, -1])

	gs2 = GridSpec(3, 3)
	gs2.update(left=0.55, right=0.98, hspace=0.05)
	plt.subplot(gs2[:, :-1])
	plt.subplot(gs2[:-1, -1])
	plt.subplot(gs2[-1, -1])

	make_ticklabels_invisible(f)


def plot_scenario(scenario):
	model = scenario['generator']
	z = scenario['z']
	gt = [x.numpy() for x in scenario['ground_truth']]
	obs = [x.numpy() for x in scenario['observations']]
	for ch in range(len(obs)):
		print('Ch. {} Weights:\n{}\n'.format(ch, model.W[ch].weight))

	lsplom(gt, ax_ref=obs, title='Ground truth')
	lsplom(obs, title='Observations')
	if z.shape[1] > 1:
		splom(z, title='z lat.dim.', colprefix='l.d.')
	else:
		plt.figure()
		plt.hist(z, 50)


def plot3d(x, y, z, classificator=None):
	if classificator is not None:
		groups = np.unique(classificator)
		if not groups.dtype == np.dtype('O'):
			# remove nans if groups are not objects (strings)
			groups = groups[~np.isnan(groups)]
		X = [x[classificator == g] for g in groups]
		Y = [y[classificator == g] for g in groups]
		Z = [z[classificator == g] for g in groups]
		legend = ['{} (n={})'.format(g, len(classificator[classificator == g])) for g in groups]
	else:
		X = [x]
		Y = [y]
		Z = [z]

	fig = plt.figure()
	ax = plt.axes(projection='3d')

	for x, y, z in zip(X, Y, Z):
		ax.scatter3D(x, y, z)


def plotclass(x, y, classificator=None):
	if classificator is not None:
		groups = np.unique(classificator)
		if not groups.dtype == np.dtype('O'):
			# remove nans if groups are not objects (strings)
			groups = groups[~np.isnan(groups)]
		X = [x[classificator == g] for g in groups]
		Y = [y[classificator == g] for g in groups]
		legend = ['{} (n={})'.format(g, len(classificator[classificator == g])) for g in groups]
	else:
		X = [x]
		Y = [y]

	for x, y in zip(X, Y):
		# the scatter plot:
		plt.scatter(x, y, alpha=0.5)
		plt.legend(legend)