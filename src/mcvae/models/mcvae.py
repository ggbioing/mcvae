import copy
from functools import reduce
from operator import add
from warnings import warn
import torch
from torch.utils.data._utils.collate import default_collate  # for imputation
from .utils import Utilities
from .vae import VAE
from ..distributions import Normal
from ..imputation import process_ids


class Mcvae(torch.nn.Module, Utilities):
	"""
	Multi-Channel VAE
	"""
	def __init__(
			self,
			data=None,
			lat_dim=1,
			n_channels=1,
			n_feats=(1,),
			beta=1.0,  # for beta-VAE
			enc_channels=None,  # encode only these channels (for kl computation)
			dec_channels=None,  # decode only these channels (for ll computation)
			noise_init_logvar=-3,
			noise_fixed=False,
			sparse=False,
			vaeclass=VAE,
			vaeclass_kwargs={},
			n_labels=None,  # for classifier, eventually
			*args, **kwargs,
	):
		super().__init__()

		assert n_channels == len(n_feats)
		self.data = data
		if data is not None:
			self.n_channels = len(data)
			self.n_feats = tuple(_.shape[1] for _ in data)
		else:
			self.n_channels = n_channels
			self.n_feats = n_feats

		self.beta = beta
		self.lat_dim = lat_dim

		if enc_channels is None:
			self.enc_channels = list(range(self.n_channels))
		else:
			assert len(enc_channels) <= self.n_channels
			self.enc_channels = enc_channels
		if dec_channels is None:
			self.dec_channels = list(range(self.n_channels))
		else:
			assert len(dec_channels) <= self.n_channels
			self.dec_channels = dec_channels
		self.n_enc_channels = len(self.enc_channels)
		self.n_dec_channels = len(self.dec_channels)

		self.noise_init_logvar = noise_init_logvar
		self.noise_fixed = noise_fixed

		self.sparse = sparse
		self.vaeclass = vaeclass
		self.vaeclass_kwargs = vaeclass_kwargs
		self.n_labels = n_labels
		self.init_vae()

	def init_vae(self):
		if self.sparse:
			self.log_alpha = torch.nn.Parameter(torch.FloatTensor(1, self.lat_dim).normal_(0, 0.01))
		else:
			self.log_alpha = None
		vae = []
		for ch in range(self.n_channels):
			vae.append(
				self.vaeclass(
					lat_dim=self.lat_dim,
					n_feats=self.n_feats[ch],
					noise_init_logvar=self.noise_init_logvar,
					noise_fixed=self.noise_fixed,
					sparse=self.sparse,
					log_alpha=self.log_alpha,
					**self.vaeclass_kwargs,
				)
			)
		self.vae = torch.nn.ModuleList(vae)

	def encode(self, x):
		return [self.vae[i].encode(x[i]) for i in range(self.n_channels)]

	def decode(self, z):
		p = []
		for i in range(self.n_channels):
			pi = [self.vae[i].decode(z[j]) for j in range(self.n_channels)]
			p.append(pi)
			del pi
		return p  # p[x][z]: p(x|z)

	def decode_in_reconstruction(self, z, *args, **kwargs):
		return self.decode(z)

	def forward(self, x):
		q = self.encode(x)
		z = [_.rsample() for _ in q]
		p = self.decode(z)

		return {
			'x': x,
			'q': q,
			'p': p
		}

	def compute_kl(self, q):
		kl = 0
		if not self.sparse:
			for i, qi in enumerate(q):
				if i in self.enc_channels:
					kl += qi.kl_divergence(Normal(0, 1)).sum(1, keepdims=True).mean(0)
		else:
			for i, qi in enumerate(q):
				if i in self.enc_channels:
					kl += qi.kl_from_log_uniform().sum(1, keepdims=True).mean(0)

		return kl

	def compute_ll(self, p, x):
		# p[x][z]: p(x|z)
		ll = 0
		for i in range(self.n_channels):
			for j in range(self.n_channels):
				# xi = reconstructed; zj = encoding
				if i in self.dec_channels and j in self.enc_channels:
					ll += self.vae[i].compute_ll(p=p[i][j], x=x[i]).mean(0)  # average ll per observation

		return ll

	def loss_function(self, fwd_ret):
		x = fwd_ret['x']
		q = fwd_ret['q']
		p = fwd_ret['p']

		kl = self.compute_kl(q)
		kl *= self.beta
		ll = self.compute_ll(p=p, x=x)

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

	def reconstruct(self, x, reconstruct_from=None, sample=False, dropout_threshold=None, *args, **kwargs):

		with torch.no_grad():

			available_channels = self.enc_channels if reconstruct_from is None else reconstruct_from
			for _ in available_channels:
				assert _ in self.enc_channels

			q = self.encode(x)

			if sample:
				z = [_.sample() for _ in q]
			else:
				z = [_.loc for _ in q]

			if dropout_threshold is not None:
				try:
					z = self.apply_threshold(z, dropout_threshold)
				except NotImplementedError:
					warn('\tTrying to apply dropout to a non-sparse model.'
						 '\tIn this case we use all the latent dimensions.')

			p = self.decode_in_reconstruction(z, *args, **kwargs)

			x_hat = []
			for c in range(self.n_channels):
				if c in self.dec_channels:
					# mean along the stacking direction
					x_tmp = torch.stack([p[c][e].loc.detach() for e in available_channels]).mean(0)
					x_hat.append(x_tmp)
					del x_tmp
				else:
					x_hat.append(None)

			return x_hat

	def impute(self, x, ids, dropout_threshold=None):

		assert len(x) == len(ids)
		assert len(x) == self.n_channels

		union = sorted(set(reduce(add, ids)))
		if len(union) is not 0:
			x_hat = []
			for s in union:
				# print(s, union.index(s))
				# available_channels_for_s = [i if s in _ and i in self.enc_channels else None for i, _ in enumerate(ids)]
				available_channels_for_s = [i if s in _ else None for i, _ in enumerate(ids)]
				xs = [x[ch][ids[ch].index(s)].unsqueeze(0) if ch is not None else None for ch in available_channels_for_s]
				qs = [self.vae[ch].encode(_) if _ is not None else None for ch, _ in enumerate(xs)]
				zs_ = [_.loc if _ is not None else None for _ in qs]
				if dropout_threshold is not None:
					zs = [self.apply_threshold(_, dropout_threshold) if _ is not None else None for _ in zs_]
				else:
					zs = zs_
				xs_hat = []
				for i in range(self.n_channels):
					if i in self.dec_channels:
						ps_hat_i = [self.vae[i].decode(z) for z in zs if z is not None]
						ev_hat_i = [self.vae[i].p_to_expected_value(_) for _ in ps_hat_i]
						xs_hat_i = torch.stack(ev_hat_i, 0).mean(0)
						if isinstance(ps_hat_i[0], torch.distributions.Categorical):
							xs_hat_i = xs_hat_i.argmax(1, keepdim=True)
						xs_hat.append(xs_hat_i)
						del ps_hat_i, ev_hat_i, xs_hat_i
					else:
						xs_hat.append([])
				x_hat.append(xs_hat)
				del xs_hat

			ret = [_.squeeze() if not isinstance(_, list) else [] for _ in default_collate(x_hat)]
		else:
			ret = [[] for _ in x]

		return ret

	def impute_from_other(self, x, ids, *args, **kwargs):

		with torch.no_grad():

			x_rec = []
			new_ids = []

			for i in range(len(x)):
				tmp_x = copy.deepcopy(x)
				tmp_ids = copy.deepcopy(ids)
				# Remove i-th channel and impute
				tmp_x[i] = list()
				tmp_ids[i] = list()
				new_ids.append(sorted(set(reduce(add, tmp_ids))))
				if i in self.dec_channels:
					tmp_rec = self.impute(x=tmp_x, ids=tmp_ids, *args, **kwargs)
					x_rec.append(tmp_rec[i])
					del tmp_rec
				else:
					x_rec.append([])

				del tmp_x, tmp_ids

			return x_rec, new_ids

	@property
	def dropout(self):
		if self.sparse:
			alpha = torch.exp(self.log_alpha.detach())
			return alpha / (alpha + 1)
		else:
			raise NotImplementedError

	def apply_threshold(self, z, threshold):
		# print(f'Applying dropout threshold of {threshold}')
		assert threshold <= 1.0
		keep = (self.dropout < threshold).squeeze()
		z_keep = []
		for _ in z:
			_[:, ~keep] = 0
			z_keep.append(_)
			del _
		return z


class MtMcvae(Mcvae):
	"""
	Multi-Task Mcvae
	"""
	def __init__(
			self,
			ids=None,
			*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self.ids = ids
		self.init_ids()

	def init_ids(self, verbose=True):
		"""
		This function is used if a list of ids is provided
		Originally thought to allow the training of channels with missing subjects/observations
		"""
		if self.ids is not None:
			print('Optimizing with missing data') if verbose else None
			# {'uniq': sorted_ids, 'KL_index': index_list, 'LL_index': ids_to_selection(ids)}
			selection = process_ids(self.ids)
			self.z_index = selection['z_index']
			self.LL_index = selection['LL_index']

	def decode(self, z):
		p = []
		for i in range(self.n_channels):  # output channels
			pi = []
			for j in range(self.n_channels):  # input channels
				if i is j:
					z_tmp = z[i]
					pij = self.vae[i].decode(z_tmp)
				else:
					if self.LL_index[j][i] is None:
						z_tmp = None
						pij = None
					else:
						selection = self.LL_index[j][i]['input']
						z_tmp = z[j][selection, ]
						pij = self.vae[i].decode(z_tmp)
				pi.append(pij)
				del z_tmp, pij
			p.append(pi)
			del pi
		return p  # p[x][z]: p(x|z)

	def decode_in_reconstruction(self, z, *args, **kwargs):
		try:
			ids = kwargs['ids']
			# save training ids in a temp variable
			train_ids = self.ids
			self.ids = ids
			self.init_ids()

			ret = self.decode(z)
			# put things back together before return
			self.ids = train_ids
			self.init_ids()
			return ret
		except KeyError:
			return super().decode(z)

	def compute_ll(self, p, x):
		# p[x][z]: p(x|z)
		ll = 0
		for i in range(self.n_channels):  # output channel
			lli = 0
			lli_items = 0
			for j in range(self.n_channels):  # input channel
				# xi = reconstructed; zj = encoding
				if i in self.dec_channels and j in self.enc_channels:  # for mc-regression
					if i is j:
						x_tmp = x[i]
					elif self.LL_index[j][i] is not None:
						x_tmp = x[i][self.LL_index[j][i]['output'], ]

					if p[i][j] is not None:  # there may be empty intersections
						lli += self.vae[i].compute_ll(p=p[i][j], x=x_tmp).mean(0)  # average ll per observation (already sum for features)
						lli_items += 1
						del x_tmp

			try:
				lli /= lli_items  # average lli per available reconstruction
				ll += lli
				del lli
			except ZeroDivisionError:
				pass

		return ll


__all__ = [
	'Mcvae',
	'MtMcvae',
]