from typing import Dict

import numpy as np
import opt_einsum as oe
import tensorly
from scipy import sparse
from tensorly.decomposition import parafac

from .helpers import noiser, proj_l1_simplex, euclidean_proj_simplex
from .misc import *


class DPSLatentDirichletAllocation:
	def __init__(
			self,
			n_topics: int,
			doc_topic_prior,
			_alpha0 = None,
			variances: Dict = {},
			l1_simplex_projection: bool = True,
	):
		"""
		:param n_topics: Number of topics.
		:param doc_topic_prior: Prior of document topic distribution. Also called alpha.
		:param variances: Noise variances configured to a given set of edges.
		:param l1_simplex_projection: Boolean condition illustrating projection of beta to simplex.
		"""
		self.n_topics = int(n_topics)
		self.doc_topic_prior = doc_topic_prior
		self._alpha0 = _alpha0
		self.variances = variances
		self.l1_simplex_projection = l1_simplex_projection

	def __repr__(self):
		return f'{self.__class__.__name__}(n_topics={self.n_topics}, doc_topic_prior={self.doc_topic_prior})'

	def calculate_moment2(self):
		pass

	def fit(self, document_word_counts):
		"""
		:param document_word_counts: Document Word Count Matrix
		:return:
		"""
		self.document_word_counts = document_word_counts
		if not hasattr(self, 'whitener') or not hasattr(self, 'unwhitener'):
			print('Whitener and Unwhitener not cached. Calculating from scratch...')
			self.m2_eigenvalues, self.m2_eigenvectors = self.decompose_moment2()# TODO: Compute actual M2
			self.whitener = self.create_whitener()
			self.unwhitener = self.create_unwhitener()
		if self.n_words > self.n_words_threshold:
			print('vocabulary size too large, skip generating raw M3, generate whitened M3 directly')
			self.whitened_moment3 = self.create_whitened_moment3()
		else:
			print('vocabulary size small, generating raw M3, then whiten it')
			self.moment3 = self.create_moment3()
			self.whitened_moment3 = self.whiten_moment3()

		self.factors = self.decompose_moment3()
		self.m3_eigenvalues = np.linalg.norm(self.factors[0], axis=0)
		self.factors[0] /= self.m3_eigenvalues
		assert np.allclose(np.linalg.norm(self.factors[0], axis=0), 1)

		self.unique_factor = self.factor_correct_sign()

		self.doc_topic_posterior = (1 / (self.m3_eigenvalues ** 2))[::-1]# problematic
		self.topic_word_distribution = self.create_topic_word_distribution() # how does this work?
		#breakpoint()
		#print('self.factors', self.factors[0])
		#print('self.unique_factor', self.unique_factor)
		#print('self.m3_eigenvalues',self.m3_eigenvalues)
		# print('norm(T-T_hat)', np.linalg.norm(,ord=1))
		# print('self.topic_word_distribution before projection',self.topic_word_distribution)

		if self.l1_simplex_projection:
			for i in range(self.n_topics):
				try:
					#self.topic_word_distribution[:, i] = proj_l1_simplex(self.topic_word_distribution[:, i])
					self.topic_word_distribution[:, i] = euclidean_proj_simplex(self.topic_word_distribution[:, i])
				except Exception as e:
					print(e)

		self.topic_word_distribution = self.topic_word_distribution[:, ::-1]
		# print('l1_projection..')
		# print('self.topic_word_distribution after projection',self.topic_word_distribution)

	@property
	def moment1(self):
		"""
		First moment of document word count matrix.
		:return:
		"""
		m1 = self.document_word_proportions.sum(axis=0) / self.n_docs
		## m1 should be a vector, but if a sparse representation was used
		## to create it, it would have 2 dims instead of 1. We use this to convert
		## from matrix to vector
		if len(m1.shape) > 1:
			m1 = m1.A1
		return m1

	@property
	def n_words_threshold(self):
		return 200
	
	@property
	def n_partitions(self):
		return 1

	# @seed(44)
	def decompose_moment2(self):
		"""
		This method approximates the second moment by using power iteration
		and computes a truncated SVD on it.
		:return:
		"""
		if self.n_words > self.n_words_threshold:
			print('vocabulary size too large, using rand_svd')
			return rand_svd(self.document_word_counts, self.alpha0, self.k, docs_m1=self.moment1, n_iter=3, n_partitions=self.n_partitions)
		else:
			print('vocabulary size small, using compute_raw_M2')
			return compute_raw_M2(self.document_word_counts, self.alpha0)

	def create_whitened_moment3(self):
		if not hasattr(self, 'whitened_moment3'):
			print('moment 3 not cached, whitening moment3')
			the_whitened_m3 = whiten_m3(self.document_word_counts, self.whitener, self._alpha0, docs_m1=self.moment1, n_partitions=self.n_partitions)
			the_whitened_m3 *= self.alpha0 * (self.alpha0 + 1) * (self.alpha0 + 2) / 2
			the_whitened_m3 = the_whitened_m3.reshape((self.k, self.k, self.k))
			return the_whitened_m3
		else:
			print('moment3 cached, returning cached moment3.')
			return self.whitened_moment3

	# @seed(43)
	@noiser(edge='e3')
	def create_moment3(self):
		"""
		Creates the third moment.
		:return:
		"""

		def term1():

			# clip zero counts to 1 to avoid divide by zero errors.
			l_ns_min_1 = self.l_ns - 1
			l_ns_min_1[l_ns_min_1 == 0] = 1
			l_ns_min_2 = self.l_ns - 2
			l_ns_min_2[l_ns_min_2 == 0] = 1

			scaling_factor = 1 / (self.l_ns * (l_ns_min_1) * (l_ns_min_2))
			scaled_document_word_counts = scaling_factor * self.document_word_counts

			part1 = oe.contract(
				'ij,ik,il->jkl',
				scaled_document_word_counts,
				self.document_word_counts,
				self.document_word_counts,
			)

			rho = oe.contract('ij,ik->jk', scaled_document_word_counts,
			                  self.document_word_counts)
			part2 = np.zeros((self.n_words, self.n_words, self.n_words))
			diagonal_indices = np.diag_indices(self.n_words, ndim=2)
			for i, item in enumerate(rho):
				part2[i][diagonal_indices] = item
			part2 += oe.contract('ijk->jik', part2) + oe.contract('ijk->kji', part2)

			part3 = np.zeros((self.n_words, self.n_words, self.n_words))
			part3[np.diag_indices(self.n_words, ndim=3)] = 2 * scaled_document_word_counts.sum(axis=0)

			return (part1 - part2 + part3) / self.n_docs

		def term2():

			# clip zero counts to 1 to avoid divide by zero errors.
			l_ns_min_1 = self.l_ns - 1
			l_ns_min_1[l_ns_min_1 == 0] = 1
			
			scaling_factor = 1 / (self.l_ns * (l_ns_min_1))
			scaled_document_word_counts = scaling_factor * self.document_word_counts

			part1 = oe.contract('i,jk,jl->ikl', self.moment1,
			                    scaled_document_word_counts,
			                    self.document_word_counts)
			part1 += oe.contract('ijk->kij', part1) + oe.contract('ijk->jki', part1)

			rho = oe.contract('i,j->ij', scaled_document_word_counts.sum(axis=0),
			                  self.moment1)
			part2 = np.zeros((self.n_words, self.n_words, self.n_words))
			diagonal_indices = np.diag_indices(self.n_words, ndim=2)
			for i, item in enumerate(rho):
				part2[i][diagonal_indices] = item
			part2 += oe.contract('ijk->jik', part2) + oe.contract('ijk->kji', part2)

			return (self.alpha0 / (self.alpha0 + 2)) * (
					part1 - part2) / self.n_docs

		def term3():
			moment1 = self.moment1
			return 2 * ((self.alpha0 ** 2) / (
					(self.alpha0 + 1) * (self.alpha0 + 2))) * oe.contract(
				'i,j,k->ijk', moment1,
				moment1, moment1)

		return (self.alpha0 * (self.alpha0 + 1) * (self.alpha0 + 2) / 2) * (
			term1() - term2() + term3())

	# @seed(48)
	@noiser(edge='e6', symmetric=True)
	def whiten_moment3(self):
		"""
		Whitens the third moment.
		:return: k by k by k tensor
		"""
		return tensorly.tenalg.multi_mode_dot(self.moment3, np.array(
			[self.whitener.T for _ in range(3)]))

	# @seed(49)
	@noiser(edge='e7')
	def decompose_moment3(self):
		"""
		Performs CP decomposition on third moment.
		:return:
		"""
		#if self.n_words > self.n_words_threshold:
		#	factors = parafac(self.whitened_moment3, self.n_topics)
		#else:
		factors = np.array(parafac(self.whitened_moment3, self.n_topics).factors)
		return factors #np.sort(factors)[::-1] # what does the sort do?

	def factor_correct_sign(self):
		"""
		Magic
		:return:
		"""
		factor = np.zeros((self.n_topics, self.n_topics))
		for i in range(self.n_topics):
			diff = [
				np.linalg.norm(self.factors[1][:, i] - self.factors[2][:, i]),
				np.linalg.norm(self.factors[0][:, i] - self.factors[2][:, i]),
				np.linalg.norm(self.factors[0][:, i] - self.factors[1][:, i]),
			]
			factor[:, i] = self.factors[np.argmin(diff)][:, i]
		return factor

	# @seed(44)
	@noiser(edge='e4')
	def create_whitener(self):
		"""
		Creates whitener.
		:return:
		"""
		return oe.contract('ij,ik->jk', self.m2_eigenvectors_partial,
		                   np.diag(1 / np.sqrt(self.m2_eigenvalues_partial)))

	# @seed(45)
	@noiser(edge='e8')
	def create_unwhitener(self):
		"""
		Creates unwhitener.
		:return:
		"""
		return oe.contract('ij,ik->jk', self.m2_eigenvectors_partial,
		                   np.diag(np.sqrt(self.m2_eigenvalues_partial)))

	# @seed(50)
	@noiser(edge='e9')
	def create_topic_word_distribution(self):
		"""
		Creates topic word distribution.
		:return:
		"""
		# this is not true
		return self.unwhitener.dot(self.unique_factor).dot(
			np.diag(self.m3_eigenvalues))
# 		return self.unwhitener.dot(self.unique_factor).dot(
# 			np.diag(np.sqrt(self.m2_eigenvalues_partial)))

	@property
	def n_docs(self):
		return self.document_word_counts.shape[0]

	@property
	def n_words(self):
		return self.document_word_counts.shape[1]

	@property
	def alpha0(self):
		if self._alpha0:
			return self._alpha0
		else:
			return np.sum(self.alpha)

	@property
	def alpha(self):
		"""
		Alias of doc_topic_prior
		:return:
		"""
		return self.doc_topic_prior

	@property
	def beta(self):
		"""
		Alias of topic_word_distribution
		:return:
		"""
		return self.topic_word_distribution

	@property
	def k(self):
		"""
		Alias of n_topics
		:return:
		"""
		return self.n_topics

	@property
	def vocab_size(self):
		"""
		Alias of n_words
		:return:
		"""
		return self.n_words

	@property
	def l_ns(self):
		"""
		Total number of words in each document.
		:return:
		"""
		if sparse.issparse(self.document_word_counts):
			return self.document_word_counts.sum(axis=1)
		else:
			return self.document_word_counts.sum(axis=1, keepdims=True)

	@property
	def document_word_proportions(self):
		if sparse.issparse(self.document_word_counts):
			return self.document_word_counts / self.document_word_counts.sum(axis=1)
		else:
			return self.document_word_counts / self.document_word_counts.sum(axis=1, 
																			 keepdims=True)


	@property
	def m2_eigenvalues_partial(self):
		return self.m2_eigenvalues[:self.n_topics]

	@property
	def m2_eigenvectors_partial(self):
		return self.m2_eigenvectors[:self.n_topics]
