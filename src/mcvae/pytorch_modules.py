import numpy as np
import torch
import torch.utils.data
from torch.distributions import Normal, kl_divergence

# TODO: write a better initialization for DEVICE, considering the available resources.
DEVICE_ID = 0
DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(DEVICE_ID)

pi = torch.FloatTensor([np.pi]).to(DEVICE)  # torch.Size([1])
log_2pi = torch.log(2 * pi)

BCELoss = torch.nn.BCELoss(reduction='none')


def reconstruction_error(predicted, target):
	# sum over data dimensions (n_feats); average over observations (N_obs)
	return ((target - predicted) ** 2).sum(1).mean(0)  # torch.Size([1])


def mae(predicted, target):
	"""
	Mean Absolute Error
	"""
	# sum over data dimensions (n_feats); average over observations (N_obs)
	return torch.abs(target - predicted).sum(1).mean(0)  # torch.Size([1])


def LL(predicted, target, logvar, index_dict=None):
	'''
	Computes Log-Likelihood.

	If X represents N observations of a scalar variable x, then
	ln p(X|mu,var) = -1/(2*var) sum_{n=1}^N( (x_n - mu)^2 -N/2 logvar - N/2 log(2*pi) )

	Remember: the maximum likelihood approach systematically underestimates the variance of the distribution.

	See Bishop PRML pag.27

	:param predicted: double tensor (Nob_s x N_features)
	:param target: double tensor (same size as 'predicted')
	:param logvar: single logvariance associated to each feature (size = N_features
	:return: scalar value containing total log-likelihood
	'''
	if index_dict is None:
		ll = -0.5 * ((predicted - target) ** 2 / logvar.exp() + logvar + log_2pi)
	else:
		ll = -0.5 * ((predicted[index_dict['left'], :] - target[index_dict['right'], :]) ** 2 / logvar.exp() + logvar + log_2pi)
	# sum over data dimensions (n_feats); average over observations (N_obs)
	return ll.sum(1).mean(0)  # mean ll per observation


def KL(mu, logvar):
	'''
	Solution of KL(qφ(z)||pθ(z)), Gaussian case:

	Sum over the dimensionality of z
	KL = - 0.5 * sum(1 + log(σ^2) - µ^2 - σ^2)

	When using a recognition model qφ(z|x) then µ and σ are simply functions of x and the variational parameters φ

	See Appendix B from VAE paper:
	Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	https://arxiv.org/abs/1312.6114
	'''
	kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
	return kl


def compute_log_alpha(mu, logvar):
	# clamp because dropout rate p in 0-99%, where p = alpha/(alpha+1)
	return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(min=-8, max=8)


def compute_logvar(mu, log_alpha):
	return log_alpha + 2 * torch.log(torch.abs(mu) + 1e-8)


def compute_clip_mask(mu, logvar, thresh=3):
	# thresh < 3 means p < 0.95
	# clamp dropout rate p in 0-99%, where p = alpha/(alpha+1)
	log_alpha = compute_log_alpha(mu, logvar)
	return (log_alpha < thresh).float()


def KL_log_uniform(p, *args, **kwargs):
	"""
	Arguments other than 'p' are ignored.

	Formula from Paragraph 4.2 in:
	Variational Dropout Sparsifies Deep Neural Networks
	Molchanov, Dmitry; Ashukha, Arsenii; Vetrov, Dmitry
	https://arxiv.org/abs/1701.05369
	"""
	log_alpha = compute_log_alpha(p.loc, p.scale.pow(2).log())
	k1, k2, k3 = 0.63576, 1.8732, 1.48695
	neg_KL = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) - k1
	return -neg_KL


def loss_has_converged(x, N=500, stop_slope=-0.01):
	"""
	True if loss function started to oscillate or small incrementation
	True also if oscillations are big; that is why you need "loss_has_diverged()"
	"""
	if stop_slope is None:
		return False
	L = len(x)
	if L < 1002:
		return False
	else:
		# slope of the loss function, moving averaged on a window of N epochs
		rate = [np.polyfit(np.arange(N), x[i:i + N], deg=1)[0] for i in range(L - N)]
		oscillations = (np.array(rate) > 0).sum()
		if oscillations > 1:
			pass
		# print("{} oscillations (> 1)".format(str((np.array(rate) > 0).sum())))
		if np.mean(rate[-100:-1]) > stop_slope:
			pass
		# print("Sl. {}. Slope criteria reached (sl > {})".format(np.mean(rate[-100:-1]), stop_slope))
		return oscillations > 1 or np.mean(rate[-100:-1]) > stop_slope


def loss_has_diverged(x):
	return x[-1] > x[0]


def loss_is_nan(x):
	return str(x[-1]) == 'nan'


def moving_average(x, n=1):
	return [np.mean(x[i - n:i]) for i in range(n, len(x))]


class MultiChannelBase(torch.nn.Module):

	def __init__(
			self,
			lat_dim=1,
			n_channels=1,
			n_feats=(1,),
			noise_init_logvar=3,
			model_name_dict=None,
	):
		super().__init__()

		assert n_channels == len(n_feats)

		self.lat_dim = lat_dim
		self.n_channels = n_channels
		self.n_feats = n_feats
		self.noise_init_logvar = noise_init_logvar
		self.model_name_dict = model_name_dict

		self.init_names()

		self.init_encoder()
		self.init_decoder()

		self.init_KL()

	def init_names(self):
		self.model_name = self._get_name()
		if not self.model_name_dict == None:
			for key in sorted(self.model_name_dict):
				val = self.model_name_dict[key]
				if type(val) == list or type(val) == tuple:
					# val = '_'.join([str(i) for i in val])
					val = str(np.sum(val))
				self.model_name += '__' + key + '_' + str(val)

		self.ch_name = ['Ch.' + str(i) for i in range(self.n_channels)]

		self.varname = []
		for ch in range(self.n_channels):
			self.varname.append(['feat.' + str(j) for j in range(self.n_feats[ch])])

	def init_encoder(self):
		# Encoders: random initialization of weights
		W_mu = []
		W_logvar = []

		for ch in range(self.n_channels):
			bias = False
			W_mu.append(torch.nn.Linear(self.n_feats[ch], self.lat_dim, bias=bias))
			W_logvar.append(torch.nn.Linear(self.n_feats[ch], self.lat_dim, bias=bias))

		self.W_mu = torch.nn.ModuleList(W_mu)
		self.W_logvar = torch.nn.ModuleList(W_logvar)

	def init_decoder(self):
		# Decoders: random initialization of weights
		W_out = []
		W_out_logvar = []

		for ch in range(self.n_channels):
			bias = False
			W_out.append(torch.nn.Linear(self.lat_dim, self.n_feats[ch], bias=bias))
			W_out_logvar.append(torch.nn.Parameter(torch.FloatTensor(1, self.n_feats[ch]).fill_(self.noise_init_logvar)))

		self.W_out = torch.nn.ModuleList(W_out)
		self.W_out_logvar = torch.nn.ParameterList(W_out_logvar)

	def init_KL(self):
		self.KL_fn = kl_divergence

	def encode(self, x):

		qzx = []
		for i, xi in enumerate(x):
			q = Normal(
				loc=self.W_mu[i](xi),
				scale=self.W_logvar[i](xi).exp().pow(0.5)
			)
			qzx.append(q)
			del q

		return qzx

	def sample_from(self, qzx):

		'''
		sampling by leveraging on the reparametrization trick
		'''

		zx = []

		for ch in range(self.n_channels):
			if self.training:
				zx.append(
					qzx[ch].rsample()
				)
			else:
				zx.append(
					qzx[ch].loc
				)

		return zx

	def decode(self, zx):

		pxz = []
		for i in range(self.n_channels):
			pxz.append([])
			for j in range(self.n_channels):
				p_out = Normal(
					loc=self.W_out[j](zx[i]),
					scale=self.W_out_logvar[j].exp().pow(0.5)
				)
				pxz[i].append(p_out)
				del p_out
		return pxz

	def forward(self, x):

		qzx = self.encode(x)
		zx = self.sample_from(qzx)
		pxz = self.decode(zx)
		return {
			'x': x,
			'qzx': qzx,
			'zx': zx,
			'pxz': pxz
		}

	def reconstruct(self, x, reconstruct_from=None):
		available_channels = range(self.n_channels) if reconstruct_from is None else reconstruct_from
		fwd_return = self.forward(x)
		pxz = fwd_return['pxz']

		Xhat = []
		for c in range(self.n_channels):
			# mean along the stacking direction
			xhat = torch.stack([pxz[e][c].loc.detach() for e in available_channels]).mean(0)
			Xhat.append(xhat)
			del xhat

		return Xhat

	def optimize_batch(self, local_batch):

		pred = self.forward(local_batch)
		loss = self.loss_function(pred)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.detach().item()

	def optimize(self, epochs, data, *args, **kwargs):

		self.train()  # Inherited method which sets self.training = True

		try:  # list of epochs to allow more training
			so_far = len(self.loss['total'])
			self.epochs[-1] = so_far
			self.epochs.append(so_far + epochs)
		except AttributeError:
			self.epochs = [0, epochs]

		for epoch in range(self.epochs[-2], self.epochs[-1]):
			if type(data) is torch.utils.data.DataLoader:
				current_batch = 0
				for local_batch in data:
					print("Batch # {} / {}".format(current_batch, len(data) - 1), end='\t')
					loss = self.optimize_batch(local_batch)
					current_batch += 1
			else:
				loss = self.optimize_batch(data)

			if np.isnan(loss):
				print('Loss is nan!')
				break

			if epoch % 100 == 0:
				self.print_loss(epoch)
				if loss_has_diverged(self.loss['total']):
					print('Loss diverged!')
					break

		self.eval()  # Inherited method which sets self.training = False

	def loss_function(self, fwd_return):
		x = fwd_return['x']
		qzx = fwd_return['qzx']
		pxz = fwd_return['pxz']

		kl = 0
		ll = 0

		for i in range(self.n_channels):

			# KL Divergence
			kl += self.KL_fn(qzx[i], Normal(0, 1)).sum(1).mean(0)  # torch.Size([1])

			for j in range(self.n_channels):
				# i = latent comp; j = decoder
				# Direct (i=j) and Crossed (i!=j) Log-Likelihood
				ll += pxz[i][j].log_prob(x[j]).sum(1).mean(0)

		total = kl - ll

		losses = {
			'total': total,
			'kl': kl,
			'll': ll
		}

		if self.training:
			self.save_loss(losses)
			return total
		else:
			return losses

	def recon_loss(self, fwd_return, target=None):
		x = fwd_return['x'] if target is None else target
		qzx = fwd_return['qzx']
		pxz = fwd_return['pxz']

		kl = 0
		ll = 0
		rec_loss = 0
		mae_loss = 0

		for i in range(self.n_channels):

			# KL Divergence
			kl += self.KL_fn(qzx[i], Normal(0, 1))

			for j in range(self.n_channels):
				# i = latent comp; j = decoder
				# Direct (i=j) and Crossed (i!=j) Log-Likelihood
				ll += pxz[i][j].log_prob(x[j]).sum(1).mean(0)
				rec_loss += reconstruction_error(target=x[j], predicted=pxz[i][j].loc)
				mae_loss += mae(target=x[j], predicted=pxz[i][j].loc)

		total = kl - ll

		losses = {
			'total': total,
			'kl': kl,
			'll': ll,
			'rec_loss': rec_loss,
			'mae': mae_loss
		}

		return losses

	def init_loss(self):
		empty_loss = {
			'total': [],
			'kl': [],
			'll': []
		}
		self.loss = empty_loss

	def print_loss(self, epoch):
		print('====> Epoch: {:4d}/{} ({:.0f}%)\tLoss: {:.4f}\tLL: {:.4f}\tKL: {:.4f}\tLL/KL: {:.4f}'.format(
			epoch,
			self.epochs[-1],
			100. * (epoch) / self.epochs[-1],
			self.loss['total'][-1],
			self.loss['ll'][-1],
			self.loss['kl'][-1],
			self.loss['ll'][-1] / (1e-8 + self.loss['kl'][-1])
		), end='\n')

	def save_loss(self, losses):
		for key in self.loss.keys():
			self.loss[key].append(float(losses[key].detach().item()))


class MultiChannelSparseVAE(MultiChannelBase):

	def __init__(
			self,
			dropout_threshold=0.2,
			*args, **kwargs,
	):
		super().__init__(*args, **kwargs)
		self.dropout_threshold = dropout_threshold

	def init_encoder(self):
		# Encoders: random initialization of weights
		W_mu = []

		for ch in range(self.n_channels):
			bias = False
			W_mu.append(torch.nn.Linear(self.n_feats[ch], self.lat_dim, bias=bias))

		self.W_mu = torch.nn.ModuleList(W_mu)
		self.log_alpha = torch.nn.Parameter(torch.FloatTensor(1, self.lat_dim).normal_(0, 0.01))

	def init_KL(self):
		self.KL_fn = KL_log_uniform

	def encode(self, x):
		"""
		z ~ N( z | mu, alpha * mu^2 )
		"""
		qzx = []
		for i, xi in enumerate(x):
			mu = self.W_mu[i](xi)
			logvar = compute_logvar(mu, self.log_alpha)
			q = Normal(
				loc=mu,
				scale=logvar.exp().pow(0.5)
			)
			qzx.append(q)
			del mu, logvar, q

		return qzx

	def dropout_fn(self, lv):
		alpha = torch.exp(self.log_alpha.detach())
		do = alpha / (alpha + 1)
		lv_out = []
		for ch in range(self.n_channels):
			lv_out.append(lv[ch] * (do < self.dropout_threshold).float())
		return lv_out

	def forward(self, x):

		qzx = self.encode(x)
		zx = self.sample_from(qzx)
		if self.training:
			pxz = self.decode(zx)
		else:
			pxz = self.decode(self.dropout_fn(zx))
		return {
			'x': x,
			'qzx': qzx,
			'zx': zx,
			'pxz': pxz
		}

	@property
	def dropout(self):
		alpha = torch.exp(self.log_alpha.detach())
		return alpha / (alpha + 1)

	@property
	def kept_components(self):
		keep = (self.dropout.reshape(-1) < self.dropout_threshold).tolist()
		components = [i for i, kept in enumerate(keep) if kept]
		return components

	def logvar(self, x):
		qzx = self.encode(x)
		logvar = []
		for ch in range(self.n_channels):
			mu = qzx[ch].loc
			lv = compute_logvar(mu, self.log_alpha)
			logvar.append(lv.detach())
		return logvar


class ScenarioGenerator(torch.nn.Module):
	def __init__(
			self,
			lat_dim=1,
			n_channels=1,
			n_feats=1,
			seed=100
	):
		"""
		Generate multiple sources (channels) of data through a linear generative model:

		z ~ N(0,I)

		for ch in N_channels:
			x_ch = W_ch(z)

		where:

			"W_ch" is an arbitrary linear mapping z -> x_ch

		:param lat_dim:
		:param n_channels:
		:param n_feats:
		"""
		super().__init__()

		self.lat_dim = lat_dim
		self.n_channels = n_channels
		self.n_feats = n_feats

		#  Save random state (http://archive.is/9glXO)
		np.random.seed(seed)  # or None
		self.random_state = np.random.get_state()  # get the initial state of the RNG

		W = []

		for ch in range(n_channels):
			w_ = np.random.uniform(-1, 1, (n_feats, lat_dim))
			u, s, vt = np.linalg.svd(w_, full_matrices=False)
			w = u if n_feats >= lat_dim else vt
			W.append(torch.nn.Linear(lat_dim, n_feats, bias=False))
			W[ch].weight.data = torch.FloatTensor(w)

		self.W = torch.nn.ModuleList(W)

	def forward(self, z):
		if type(z) == np.ndarray:
			z = torch.FloatTensor(z)

		assert z.size(1) == self.lat_dim

		obs = []
		for ch in range(self.n_channels):
			x = self.W[ch](z)
			obs.append(x.detach())

		return obs