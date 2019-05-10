import warnings
import numpy as np


def whiten(X):
	"""
	Return a matrix W s.t.:
	1) W.shape = X.shape
	2) cov(W) = I
	"""
	N = X.shape[0]
	U, S, Vt = np.linalg.svd(X, full_matrices=False)
	W = np.dot(np.dot(X, Vt.T), np.diag(S ** -1))
	if not np.allclose(W, U):  # False in presence of null Singular Values
		warnings.warn('Presence of null Singular Values')
	# W = U 
	return W * (N - 1) ** 0.5  # standardized scores


def preprocess(x_, remove_mean=True, normalize=True, whitening=False, dimscale=False, dim=0, *args, **kwargs):
	if isinstance(x_, list):
		return [preprocess(i, remove_mean, normalize, whitening, dimscale, dim, *args, **kwargs) for i in x_]
	x = np.array(x_, copy=True)
	print("Data:", x.shape)
	if remove_mean or normalize or whitening:
		print("\tRemoving mean")
		x -= np.mean(x, dim)
		if normalize or whitening:
			print("\tNormalizing")
			x /= np.std(x, dim, ddof=1)
			if whitening:
				print("\tWhitening")
				x = whiten(x)
	print("\tSetting +/- inf and nans to zero!")
	x[np.abs(x) == np.inf] = 0  # put +/- inf to zero
	x[x!=x] = 0  # put nans to zero
	if dimscale:
		scale = x.shape[1]
		print(f"\tScaling by sqrt({scale})")
		x /= np.sqrt(scale)
	return x
