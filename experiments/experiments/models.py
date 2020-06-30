import numpy as np
import pickle
import traceback
import sys
import os
from django.contrib.postgres.fields import ArrayField, JSONField
from django.db import models
from itertools import chain
from django.db.models.signals import post_init
from django.utils.functional import cached_property
from polymorphic.models import PolymorphicModel
from privatespectrallda import DPSLatentDirichletAllocation
from privatespectrallda.helpers import generate_document_word_counts
from autodp import rdp_bank, rdp_acct, dp_acct,privacy_calibrator
from .hungarian import Hungarian

from .helpers import *


# np.random.seed(42)
def calibrate_epsilon(params, delta):#lemma_8
	# We use approximate-CDP for the composition, and then calculate the \epsilon parameters as a function of \delta

	# Input 'params' should contain the following fields
	# params['config'] keeps the integer denoting which configuration it is
	# params['eps_sigma']  keeps the epsilon parameter used by the Laplace mechanism when releasing M2's eigenvalue
	# params['delta_sigma'] denotes the failure probability for the high-probability upper bound of LS
	# params['eps_gamma'] and params['delta_gamma']  are similarly for M3's eigenvalue
	# params['Gaussian'] contains a list of tuples each containing (sensitivity,  variance)
	# this is because each config often release more than one quantities
	config = params['config']
	eps_edge_dist = params['eps_dist']
	acct = rdp_acct.anaRDPacct()

	if not config:
		return 0

	delta0 = 0

	if config == 'config4':
		eps_e9 = eps_edge_dist['e9']
		eps_sigma = eps_e9 / 4
		eps_gamma = eps_e9 / 4
		delta_sigma = delta / 4
		delta_gamma = delta / 4
		delta0 = delta_sigma + delta_gamma
		acct.compose_mechanism(lambda x: rdp_bank.RDP_pureDP({'eps': eps_sigma}, x))
		acct.compose_mechanism(lambda x: rdp_bank.RDP_pureDP({'eps': eps_gamma}, x))

	if config == 'config3':
		eps_e7 = eps_edge_dist['e7']
		eps_sigma = eps_e7 / 3
		eps_gamma = eps_e7 / 3
		delta_sigma = delta / 3
		delta_gamma = delta / 3
		delta0 = delta_sigma + delta_gamma
		acct.compose_mechanism(lambda x: rdp_bank.RDP_pureDP({'eps': eps_sigma}, x))
		acct.compose_mechanism(lambda x: rdp_bank.RDP_pureDP({'eps': eps_gamma}, x))

	if config == 'config2':
		eps_e6 = eps_edge_dist['e6']
		eps_sigma = eps_e6 / 2
		delta_sigma = delta / 2
		delta0 = delta_sigma
		acct.compose_mechanism(lambda x: rdp_bank.RDP_pureDP({'eps': eps_sigma}, x))

	print('delta0:', delta0)

	if delta0 >= delta:
		return np.inf

	for sensitivity, variance in params['gaussian']:
		## often we pre-emptively calculate sensitivities, 
		## so they might not be zero in places where we aren;t adding noise. 
		## variance provides a better check for this.
		if sensitivity == 0 or variance == 0:
			continue

		std = np.sqrt(variance)
		# CDP of gaussian mechanism conditioning on the event is the same as its RDP.
		acct.compose_mechanism(lambda x: rdp_bank.RDP_gaussian({'sigma': std / max(sensitivity,np.finfo(np.float32).eps)}, x))

	# This privacy calcluation follows from Lemma 8.8 of Bun et al. (2016) https://arxiv.org/pdf/1605.02065.pdf
	return acct.get_eps((delta - delta0) / (1 - delta0))


class Corpus(models.Model):
	seed = models.IntegerField(default=42)
	n_topics = models.IntegerField(default=3, null=True, blank=True)
	n_docs = models.IntegerField(default=10_000, null=True, blank=True)
	n_words = models.IntegerField(default=10, null=True, blank=True)
	pkl = models.CharField(max_length=200, null=True, blank=True)

	_document_word_counts = ArrayField(ArrayField(models.IntegerField(default=0), default=list), default=list)

	def __repr__(self):
		return f'{self.__class__.__name__}(n_docs: {self.n_docs}, n_words: {self.n_words})'

	@property
	def document_word_counts(self):
		if self.pkl:
			with open(self.pkl, 'rb') as f:
				return pickle.load(f)
		else:
			return np.array(self._document_word_counts)

	def generate_document_word_counts(self, alpha, beta):
		self._document_word_counts = generate_document_word_counts(alpha, beta, self.n_docs, seed=self.seed).tolist()
		self.save()

	@cached_property
	def actual_documents(self):
		return [' '.join(chain.from_iterable([[str(i)] * int(word) for i, word in enumerate(doc)])) for doc in self._document_word_counts]

	@cached_property
	def vocabulary(self):
		return [str(i) for i in range(self.document_word_counts.shape[1])]


class Experiment(PolymorphicModel):
	seed = models.IntegerField(default=42)

	@property
	def n_docs(self):
		return self.document.n_docs

	@property
	def n_words(self):
		return self.document.n_words

	@property
	def best_run(self):
		return self.runs.all().order_by('-utility_loss').first()

	@property
	def alpha(self):
		if not self._alpha:
			return None
		_alpha = np.array(self._alpha)
		_alpha = (_alpha / _alpha.sum()) * self.alpha0
		return _alpha

	@property
	def beta(self):
		return np.array(self._beta)


class ParentExperiment(Experiment):#unnoised experiments
	alpha0 = models.FloatField(default=1)
	n_topics = models.IntegerField(null=True, blank=True)
	delta = models.FloatField(default=0.1, null=True, blank=True)
	composite_epsilon = models.FloatField(default=0.1)
	document = models.ForeignKey(Corpus, blank=True, null=True, on_delete=models.CASCADE, related_name='experiments')
	_alpha = ArrayField(models.FloatField(default=0), null=True, blank=True)
	_beta = ArrayField(ArrayField(models.FloatField(default=0)), null=True, blank=True)

	@property
	def run(self):
		return self.runs.first()


class ChildExperiment(Experiment):# noised experiments config1,2,3 and 4
	configuration = models.CharField(max_length=255, null=True, blank=True, choices=[
		(c, c) for c in ['config1', 'config2', 'config3', 'config4']
	])
	parent = models.ForeignKey(ParentExperiment, null=True, blank=True, related_name='children',
	                           on_delete=models.CASCADE)

	@property
	def n_topics(self):
		return self.parent.n_topics

	@property
	def document(self):
		return self.parent.document

	@property
	def alpha0(self):
		return self.parent.alpha0

	@property
	def composite_epsilon(self):
		return self.parent.composite_epsilon

	@property
	def delta(self):
		return self.parent.delta

	@property
	def _alpha(self):
		return self.parent._alpha

	@property
	def _beta(self):
		return self.parent._beta

	def sensitivity(self, edge,  edge_epsilon, edge_delta):
		result = SENSITIVITIES[edge](
			self.parent.run.lda.document_word_counts,
			self.parent.run.lda.n_topics,
			self.parent.run.alpha0,
			self.parent.run.lda.m2_eigenvalues,
			self.parent.run.lda.m3_eigenvalues[::-1],
			edge_epsilon,
			edge_delta,
		)
		return result


class Run(models.Model):
	experiment = models.ForeignKey(Experiment, null=True, blank=True, on_delete=models.CASCADE, related_name='runs')
	epsilon_edge_distribution = JSONField(default=zeroed_edges, null=True, blank=True)#a dictionary(key is the edge 'e9', value is the budget epsilon), absolute value of epsilon budget for each edge
	utility_loss = models.FloatField(null=True, blank=True)
	failed = models.BooleanField(default=False)
	calibrated_epsilon = models.FloatField(default=0)

	def compute(self):
		np.random.seed(int(self.experiment.seed))
		lda = None
		try:
			lda = DPSLatentDirichletAllocation(
				n_topics=self.n_topics,
				doc_topic_prior=self.alpha,
				_alpha0 = self.experiment.alpha0,
				variances=self.variances,
				l1_simplex_projection=True,
			)

			## If variances for e3 and e4 are 0, and this is not a parent experiment, we are not using config1.
			## Then, we can just use the cached value of whitener, unwhitener, and whitened m3.
			if self.variances['e3'] == 0 and self.variances['e4'] == 0 and hasattr(self.experiment, 'parent'):
				print('not using config 1. Caching attributes.')
				whitener = self.experiment.parent.run.lda.whitener
				unwhitener = self.experiment.parent.run.lda.unwhitener
				whitened_moment3 = self.experiment.parent.run.lda.whitened_moment3
				setattr(lda, 'whitener', whitener)
				setattr(lda, 'unwhitener', unwhitener)
				setattr(lda, 'whitened_moment3', whitened_moment3)

			lda.fit(self.document.document_word_counts)

			beta = lda.beta
			synthetic_beta = np.array(self.experiment._beta) if self.experiment._beta else None

			# for synthetic experiments: this calculates the utility loss between the true beta and the estimated.
			# for real-data experiments, this calculates the utility loss between the unnoised beta and the noised.
			if synthetic_beta is not None:
				cost = np.zeros((beta.T.shape[0], synthetic_beta.T.shape[0]))
				for ii, i in enumerate(beta.T):
					for jj, j in enumerate(synthetic_beta.T):
						cost[ii, jj] = np.linalg.norm(i - j, ord=1)
				#print('cost matrix:',cost)
				hungarian = Hungarian(cost)
				hungarian.calculate()
				# Hungarian
				self.utility_loss = sum(cost[result] for result in hungarian.get_results())
				print(f'Utility Loss: {round(self.utility_loss, 2)}')

			# Input 'params' should contain the following fields
			# params['config'] keeps the integer denoting which configuration it is
			# params['eps_sigma']  keeps the epsilon parameter used by the Laplace mechanism when releasing M2's
			# eigenvalue
			# params['delta_sigma'] denotes the failure probability for the high-probability upper bound of LS
			# params['eps_gamma'] and params['delta_gamma']  are similarly for M3's eigenvalue
			# params['Gaussian'] contains a list of tuples each containing (sensitivity,  variance)
			# this is because each config often release more than one quantities
			self.calibrated_epsilon = calibrate_epsilon({
				'config': self.experiment.configuration if not isinstance(self.experiment, ParentExperiment) else None,
				'eps_dist': self.epsilon_edge_distribution,
				'gaussian': zip(self.sensitivities.values(), self.variances.values()),
			}, self.delta)
			print(f'calibrated epsilon: {self.calibrated_epsilon}')

			self.failed = False
			self.save()
			self.save_lda(lda)
			self.save_misc()

			#print('kth M2 Eigenvalue:', self.lda.m2_eigenvalues[self.n_topics-1], self.lda.m2_eigenvalues, sorted(self.lda.m2_eigenvalues)[self.n_topics-1])
		except Exception as e:
			print(e)
			exc_info = sys.exc_info()
			traceback.print_exception(*exc_info)
			#breakpoint()
			self.failed = True
			self.save()
			self.save_lda(lda)
			self.save_misc()

	@property
	def n_topics(self):
		return self.experiment.n_topics

	@property
	def document(self):
		return self.experiment.document

	@property
	def alpha(self):
		return self.experiment.alpha

	@property
	def alpha0(self):
		return self.experiment.alpha0

	@property
	def beta(self):
		return self.experiment.beta

	@cached_property
	def taus(self): # gaussian mechnism's coefficient w.r.t the epsilon and delta
		if isinstance(self.experiment, ParentExperiment):
			return zeroed_edges()

		taus_ = {}
		for edge, edge_epsilon in self.epsilon_edge_distribution.items():
			taus_[edge] = tau(edge, edge_epsilon, self.delta)

		return taus_

	@cached_property
	def sensitivities(self):#dictionary, maps an edge ('e9') to its sensitivity
		if isinstance(self.experiment, ParentExperiment):
			return zeroed_edges()
		sensitivities_ = {}
		for edge, edge_epsilon in self.epsilon_edge_distribution.items():
			sensitivities_[edge] = self.sensitivity(edge, edge_epsilon, self.delta)

		return sensitivities_

	@cached_property
	def variances(self): # variance of the Gaussian noise
		if isinstance(self.experiment, ParentExperiment):
			return zeroed_edges()

		variances_ = {}
		for edge, edge_epsilon in self.epsilon_edge_distribution.items():

			if edge in ['e6']:
				epsilon = edge_epsilon / 2
				delta = self.delta / 2
			elif edge in ['e7']:
				epsilon = edge_epsilon / 3
				delta = self.delta / 3
			elif edge in ['e9']:
				epsilon = edge_epsilon / 4
				delta = self.delta / 4
			elif edge in ['e3', 'e4', 'e8']:
				epsilon = edge_epsilon
				delta = self.delta
			else:
				raise Exception(f'{edge} not supported.')

			if epsilon == 0:
				variances_[edge] = 0
				continue

			term1 = (self.sensitivity(edge, edge_epsilon, self.delta) ** 2) / (2 * epsilon ** 2) # lemma 8's first multiplicative factor
			term2 = (np.sqrt(epsilon + np.log(1 / delta)) + np.sqrt(np.log(1 / delta))) ** 2 # lemma 8's second multiplicative factor

			variances_[edge] = term1 * term2

		return variances_

	@property
	def sensitivity(self):
		return self.experiment.sensitivity

	@property
	def delta(self):
		return self.experiment.delta

	def save_lda(self, lda):
		with open(f'outputs/lda/{self.experiment.id}__{self.id}.pickle', 'wb') as file:
			pickle.dump(lda, file)

		with open(f'outputs/lda/{self.experiment.id}__{self.id}_as_dict.pickle', 'wb') as file:
			pickle.dump(lda.__dict__, file)

	@cached_property
	def edge_params(self):
		edge_params_ = {}

		if isinstance(self.experiment, ChildExperiment):
			for edge, edge_epsilon in self.epsilon_edge_distribution.items():
				edge_params_[edge] = {
					'lambda_sigma_k': None,
					'lambda_gamma_s': None,
				}

				if edge_epsilon == 0:
					continue

				if self.experiment.configuration == 'config2':
					edge_params_[edge]['lambda_sigma_k'] = 1 / (self.experiment.n_docs * edge_epsilon / 2)
				elif self.experiment.configuration in ['config3', 'config4']:
					edge_params_[edge]['lambda_sigma_k'] = 1 / (self.experiment.n_docs * edge_epsilon / 2)
					edge_params_[edge]['lambda_gamma_s'] = 1 / (self.experiment.n_docs * edge_epsilon / 2)
				else:
					pass

		return edge_params_

	def save_misc(self):
		with open(f'outputs/misc/{self.experiment.id}__{self.id}.pickle', 'wb') as file:
			pickle.dump({
				'variances': self.variances,
				'epsilon_edge_distribution': self.epsilon_edge_distribution,
				'edge_params': self.edge_params,
				'calibrated_epsilon': calibrate_epsilon({
					'config': self.experiment.configuration if not isinstance(self.experiment, ParentExperiment) else None,
					'eps_dist': self.epsilon_edge_distribution,
					'gaussian': zip(self.sensitivities.values(), self.variances.values()),
				}, self.delta), 
			}, file)

	@cached_property
	def lda(self):
		with open(f'outputs/lda/{self.experiment.id}__{self.id}.pickle', 'rb') as file:
			return pickle.load(file)
