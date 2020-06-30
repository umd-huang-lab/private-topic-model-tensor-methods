# onlineldavb.py: Package of functions for fitting Latent Dirichlet
# Allocation (LDA) with online variational Bayes (VB).
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import numpy as np
import re
import string
import sys
import time
from scipy.special import gammaln, psi
from autodp import rdp_bank, rdp_acct, dp_acct,privacy_calibrator

np.random.seed(100000001)
# meanchangethresh = 0.001
meanchangethresh = 1
maxLen = 500
clippingProportion = 0.1  # 1 for non-private #JF: Improve sensitivity by clipping
# (actually projecting) the norm of a document's sufficient statistics
# to this fraction of maxLen.  To do no clipping, set to 1.
# Set this to 1 when performing non-private LDA, where clipping is not required.
print("clipping proportion %f" % (clippingProportion))


def dirichlet_expectation(alpha):
	"""
	For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
	"""
	if len(alpha.shape) == 1:
		return psi(alpha) - psi(np.sum(alpha))
	return psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]


def parse_doc_list(docs, vocab):
	"""
	Parse a document into a list of word ids and a list of counts,
	or parse a set of documents into two lists of lists of word ids
	and counts.

	Arguments:
	docs:  List of D documents. Each document must be represented as
		   a single string. (Word order is unimportant.) Any
		   words not in the vocabulary will be ignored.
	vocab: Dictionary mapping from words to integer ids.

	Returns a pair of lists of lists.

	The first, wordids, says what vocabulary tokens are present in
	each document. wordids[i][j] gives the jth unique token present in
	document i. (Don't count on these tokens being in any particular
	order.)

	The second, wordcts, says how many times each vocabulary token is
	present. wordcts[i][j] is the number of times that the token given
	by wordids[i][j] appears in document i.
	"""

	# if type(docs).__name__ == 'str':
	# 	temp = list()
	# 	temp.append(docs)
	# 	docs = temp
	#
	# D = len(docs)
	#
	# wordids = list()
	# wordcts = list()
	# for d in range(0, D):
	# 	docs[d] = docs[d].lower()
	# 	words = docs[d].split()
	# 	ddict = dict()
	# 	for word in words:
	# 		if (word in vocab):
	# 			wordtoken = vocab[word]
	# 			if (not wordtoken in ddict):
	# 				ddict[wordtoken] = 0
	# 			ddict[wordtoken] += 1
	# 	wordids.append(list(ddict.keys()))
	# 	wordcts.append(list(ddict.values()))
	#
	# return ((wordids, wordcts))

	return tuple(zip(*(list(zip(*[(i, c) for i, c in enumerate(doc) if c != 0])) for doc in docs)))


class OnlineLDA:
	"""
	Implements online VB for LDA as described in (Hoffman et al. 2010).
	"""

	def __init__(self, vocab, K, D, alpha, eta, tau0, kappa, priv, budget, gamma_noise, mech):
		"""
		Arguments:
		K: Number of topics
		vocab: A set of words to recognize. When analyzing documents, any word
		   not in this set will be ignored.
		D: Total number of documents in the population. For a fixed corpus,
		   this is the size of the corpus. In the truly online setting, this
		   can be an estimate of the maximum number of documents that
		   could ever be seen.
		alpha: Hyperparameter for prior on weight vectors theta
		eta: Hyperparameter for prior on topics beta
		tau0: A (positive) learning parameter that downweights early iterations
		kappa: Learning rate: exponential decay rate---should be between
			 (0.5, 1.0] to guarantee asymptotic convergence.

		Note that if you pass the same set of D documents in every time and
		set kappa=0 this class can also be used to do batch VB.
		"""
		print('Creating object')
		self._vocab = {word.lower(): i for i, word in enumerate(vocab)}

		self._K = K
		self._W = len(self._vocab)
		self._D = D
		self._alpha = alpha
		self._eta = eta
		self._tau0 = tau0 + 1
		self._kappa = kappa
		self._updatect = 0
		self.priv = priv
		self.budget = budget
		self.gamma_noise = gamma_noise
		self.mech = mech

		# Initialize the variational distribution q(beta|lambda)
		self._lambda = 1 * np.random.gamma(100., 1. / 100., (self._K, self._W))
		self._Elogbeta = dirichlet_expectation(self._lambda)
		self._expElogbeta = np.exp(self._Elogbeta)
		self.convergence_iterations = []

	def do_e_step(self, wordids, wordcts):
		batchD = len(wordids)
		print('batchD:', batchD)

		# Initialize the variational distribution q(theta|gamma) for
		# the mini-batch
		gamma = 1 * np.random.gamma(100., 1. / 100., (batchD, self._K))
		Elogtheta = dirichlet_expectation(gamma)
		expElogtheta = np.exp(Elogtheta)

		sstats = np.zeros(self._lambda.shape)
		# Now, for each document d update that document's gamma and phi
		it = 0
		meanchange = 0
		numClipped = 0  # JF: how many documents in the minibatch were subject to clipping
		for d in range(0, batchD):
			# print sum(wordcts[d])
			# These are mostly just shorthand (but might help cache locality)
			ids = wordids[d]
			cts = wordcts[d]
			gammad = gamma[d, :]
			Elogthetad = Elogtheta[d, :]
			expElogthetad = expElogtheta[d, :]
			expElogbetad = self._expElogbeta[:, ids]
			# The optimal phi_{dwk} is proportional to
			# expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
			phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
			# Iterate between gamma and phi until convergence
			for it in range(0, 100):
				lastgamma = gammad
				# We represent phi implicitly to save memory and time.
				# Substituting the value of the optimal phi back into
				# the update for gamma gives this update. Cf. Lee&Seung 2001.
				gammad = self._alpha + expElogthetad * \
				         np.dot(cts / phinorm, expElogbetad.T)
				# print gammad[:, n.newaxis]
				Elogthetad = dirichlet_expectation(gammad)
				expElogthetad = np.exp(Elogthetad)
				phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
				# If gamma hasn't changed much, we're done.
				meanchange = np.mean(abs(gammad - lastgamma))
				if it % 80 == 0:
					print(it, 'Mean Change:', meanchange)
				if meanchange < meanchangethresh:
					print(f'Converged at {it}')
					self.convergence_iterations.append(it)
					break
			gamma[d, :] = gammad
			# Contribution of document d to the expected sufficient
			# statistics for the M step.
			# sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)

			# sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm) #JF: commented, to add clipping code
			if clippingProportion == 1:
				sstats[:, ids] += np.outer(expElogthetad.T, cts / phinorm)
			else:  # JF: implement norm clipping
				temp = np.outer(expElogthetad.T, cts / phinorm)
				norm = np.linalg.norm(temp * self._expElogbeta[:, ids])  # TODO: can I calculate the norm without
				# doing this multiplication?
				# print "SS norm: %f" %(norm) #JF: testing
				# print "SS sum: %f" %(n.sum(temp * self._expElogbeta[:, ids] )) #JF: testing
				clippedSensitivity = maxLen * clippingProportion  # JF: per-document sensitivity, after clipping
				print(clippedSensitivity)
				if norm > clippedSensitivity:
					temp = temp / (norm / clippedSensitivity)  # clip (really, project) sufficient statistics for the
					# document to satisfy an L2 norm constraint
					# normAfterClipping = n.linalg.norm(temp * self._expElogbeta[:, ids])
					numClipped += 1
				sstats[:, ids] += temp

		if clippingProportion != 1:
			print("%d of %d documents clipped" % (numClipped, batchD))

		# This step finishes computing the sufficient statistics for the
		# M step, so that
		# sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
		# = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.

		#######################################################
		""" this is what I need to perturb for privacy """
		sstats = sstats * self._expElogbeta
		#######################################################
		# size of sstats: # topics times # of vocab
		# this sstats is sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
		# so before noise addition, multiply 1/S.
		# then, after noise addition, multiply S, because
		# this is how Matt updates lambda: self._lambda = self._lambda * (1-rhot) + rhot * (self._eta + self._D *
		# sstats / len(docs))

		# how to draw gamma random variables (numpy intro)
		# shape, scale = 2., 2. # mean and dispersion
		# s = np.random.gamma(shape, scale, 1000)

		if self.priv:
			epsilon = self.budget[0]

			if self.gamma_noise:
				gam_shape = 1
				gam_rate = self._D * epsilon / float(maxLen)  # get good results if batchD >> maxLen
				gam_scale = 1 / float(gam_rate)

				noise = np.random.gamma(gam_shape, gam_scale, sstats.shape)

			else:
				if self.mech == 0:  # mech ==0, Gaussian
					delta_iter = self.budget[1]
					c2 = 2 * np.log(1.25 / delta_iter)
					# sensitivity = maxLen/float(self._D)
					sensitivity = maxLen / float(batchD)
					# sensitivity = 0.0013/float(batchD) #JF: TESTING!!!!!!!!!! BEST CASE
					# print "best case sensitivity is %f" %(sensitivity)
					sensitivity *= clippingProportion  # JF
					print("clipped sensitivity is %f" % (sensitivity))  # JF
					nsv = c2 * (sensitivity ** 2) / (epsilon ** 2)
					noise = np.random.normal(0, np.sqrt(nsv), sstats.shape)
				else:  # mech == 1, Laplace
					laplace_b = maxLen / float(self._D * epsilon)
					noise = np.random.laplace(0, laplace_b, sstats.shape)

			# maxSstat = n.max(sstats/float(batchD))
			print(noise)
			print('Before', sstats)
			before = sstats
			sstats = sstats / float(batchD) + noise
			after = sstats
			print('After', sstats)

			print(noise)
			print(after - before)

			""" plotting """
			# s = sstats/float(batchD)
			# plt.hist(n.reshape(s, n.prod(s.shape)), 100)
			# plt.hist(n.reshape(noise, n.prod(noise.shape)), 100)
			# plt.xlim([-0.005, 0.005])
			# plt.show()

			# count, bins, ignored = plt.hist(s, 30)

			# map back to non-negative sstats
			neg_idx = np.nonzero(sstats < 0)
			sstats[neg_idx] = 0
			# large_idx = n.nonzero(sstats>maxSstat)
			# sstats[large_idx] = n.max(sstats)

			sstats = float(batchD) * sstats

		return ((gamma, sstats))

	def do_e_step_docs(self, docs):
		"""
		Given a mini-batch of documents, estimates the parameters
		gamma controlling the variational distribution over the topic
		weights for each document in the mini-batch.

		Arguments:
		docs:  List of D documents. Each document must be represented
			   as a string. (Word order is unimportant.) Any
			   words not in the vocabulary will be ignored.

		Returns a tuple containing the estimated values of gamma,
		as well as sufficient statistics needed to update lambda.
		"""
		# This is to handle the case where someone just hands us a single
		# document, not in a list.
		if isinstance(docs, str):
			temp = list()
			temp.append(docs)
			docs = temp
		wordids, wordcts = parse_doc_list(docs, self._vocab)

		return self.do_e_step(wordids, wordcts)

	def update_lambda_docs(self, docs):
		"""
		First does an E step on the mini-batch given in wordids and
		wordcts, then uses the result of that E step to update the
		variational parameter matrix lambda.

		Arguments:
		docs:  List of D documents. Each document must be represented
			   as a string. (Word order is unimportant.) Any
			   words not in the vocabulary will be ignored.

		Returns gamma, the parameters to the variational distribution
		over the topic weights theta for the documents analyzed in this
		update.

		Also returns an estimate of the variational bound for the
		entire corpus for the OLD setting of lambda based on the
		documents passed in. This can be used as a (possibly very
		noisy) estimate of held-out likelihood.
		"""

		# rhot will be between 0 and 1, and says how much to weight
		# the information we got from this mini-batch.
		rhot = pow(self._tau0 + self._updatect, -self._kappa)
		self._rhot = rhot
		# Do an E step to update gamma, phi | lambda for this
		# mini-batch. This also returns the information about phi that
		# we need to update lambda.
		gamma, sstats = self.do_e_step_docs(docs)
		# Estimate held-out likelihood for current values of lambda.
		bound = self.approx_bound_docs(docs, gamma)
		# Update lambda based on documents.
		self._lambda = self._lambda * (1 - rhot) + \
		               rhot * (self._eta + self._D * sstats / len(docs))
		self._Elogbeta = dirichlet_expectation(self._lambda)
		self._expElogbeta = np.exp(self._Elogbeta)
		self._updatect += 1

		self._gamma = gamma

		return (gamma, bound)

	def update_lambda(self, wordids, wordcts):
		"""
		First does an E step on the mini-batch given in wordids and
		wordcts, then uses the result of that E step to update the
		variational parameter matrix lambda.

		Arguments:
		docs:  List of D documents. Each document must be represented
			   as a string. (Word order is unimportant.) Any
			   words not in the vocabulary will be ignored.

		Returns gamma, the parameters to the variational distribution
		over the topic weights theta for the documents analyzed in this
		update.

		Also returns an estimate of the variational bound for the
		entire corpus for the OLD setting of lambda based on the
		documents passed in. This can be used as a (possibly very
		noisy) estimate of held-out likelihood.
		"""

		# rhot will be between 0 and 1, and says how much to weight
		# the information we got from this mini-batch.
		rhot = pow(self._tau0 + self._updatect, -self._kappa)
		self._rhot = rhot
		# Do an E step to update gamma, phi | lambda for this
		# mini-batch. This also returns the information about phi that
		# we need to update lambda.
		(gamma, sstats) = self.do_e_step(wordids, wordcts)
		# Estimate held-out likelihood for current values of lambda.
		bound = self.approx_bound(wordids, wordcts, gamma)
		# Update lambda based on documents.
		self._lambda = self._lambda * (1 - rhot) + \
		               rhot * (self._eta + self._D * sstats / len(wordids))
		self._Elogbeta = dirichlet_expectation(self._lambda)
		self._expElogbeta = np.exp(self._Elogbeta)
		self._updatect += 1

		return (gamma, bound)

	def approx_bound(self, wordids, wordcts, gamma):
		"""
		Estimates the variational bound over *all documents* using only
		the documents passed in as "docs." gamma is the set of parameters
		to the variational distribution q(theta) corresponding to the
		set of documents passed in.

		The output of this function is going to be noisy, but can be
		useful for assessing convergence.
		"""
		# This is to handle the case where someone just hands us a single
		# document, not in a list.
		batchD = len(wordids)

		score = 0
		Elogtheta = dirichlet_expectation(gamma)
		expElogtheta = n.exp(Elogtheta)

		# E[log p(docs | theta, beta)]
		for d in range(0, batchD):
			gammad = gamma[d, :]
			ids = wordids[d]
			cts = np.array(wordcts[d])
			phinorm = np.zeros(len(ids))
			for i in range(0, len(ids)):
				temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
				tmax = max(temp)
				phinorm[i] = np.log(sum(np.exp(temp - tmax))) + tmax
			score += np.sum(cts * phinorm)
		#             oldphinorm = phinorm
		#             phinorm = n.dot(expElogtheta[d, :], self._expElogbeta[:, ids])
		#             print oldphinorm
		#             print n.log(phinorm)
		#             score += n.sum(cts * n.log(phinorm))

		# E[log p(theta | alpha) - log q(theta | gamma)]
		score += np.sum((self._alpha - gamma) * Elogtheta)
		score += np.sum(gammaln(gamma) - gammaln(self._alpha))
		score += sum(gammaln(self._alpha * self._K) - gammaln(np.sum(gamma, 1)))

		# Compensate for the subsampling of the population of documents
		score = score * self._D / len(wordids)

		# E[log p(beta | eta) - log q (beta | lambda)]
		score = score + np.sum((self._eta - self._lambda) * self._Elogbeta)
		score = score + np.sum(gammaln(self._lambda) - gammaln(self._eta))
		score = score + np.sum(gammaln(self._eta * self._W) -
		                      gammaln(np.sum(self._lambda, 1)))

		return score

	def approx_bound_docs(self, docs, gamma):
		"""
		Estimates the variational bound over *all documents* using only
		the documents passed in as "docs." gamma is the set of parameters
		to the variational distribution q(theta) corresponding to the
		set of documents passed in.

		The output of this function is going to be noisy, but can be
		useful for assessing convergence.
		"""

		# This is to handle the case where someone just hands us a single
		# document, not in a list.
		if (type(docs).__name__ == 'string'):
			temp = list()
			temp.append(docs)
			docs = temp

		(wordids, wordcts) = parse_doc_list(docs, self._vocab)
		batchD = len(docs)

		score = 0
		Elogtheta = dirichlet_expectation(gamma)
		expElogtheta = np.exp(Elogtheta)
		self._expElogtheta = expElogtheta

		# E[log p(docs | theta, beta)]
		for d in range(0, batchD):
			gammad = gamma[d, :]
			ids = wordids[d]
			cts = np.array(wordcts[d])
			phinorm = np.zeros(len(ids))
			for i in range(0, len(ids)):
				temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
				tmax = max(temp)
				phinorm[i] = np.log(sum(np.exp(temp - tmax))) + tmax
			score += np.sum(cts * phinorm)
		#             oldphinorm = phinorm
		#             phinorm = n.dot(expElogtheta[d, :], self._expElogbeta[:, ids])
		#             print oldphinorm
		#             print n.log(phinorm)
		#             score += n.sum(cts * n.log(phinorm))
		# E[log p(theta | alpha) - log q(theta | gamma)]
		score += np.sum((self._alpha - gamma) * Elogtheta)
		score += np.sum(gammaln(gamma) - gammaln(self._alpha))
		score += sum(gammaln(self._alpha * self._K) - gammaln(np.sum(gamma, 1)))

		# Compensate for the subsampling of the population of documents
		score = score * self._D / len(docs)

		# E[log p(beta | eta) - log q (beta | lambda)]
		score = score + np.sum((self._eta - self._lambda) * self._Elogbeta)
		score = score + np.sum(gammaln(self._lambda) - gammaln(self._eta))
		score = score + np.sum(gammaln(self._eta * self._W) -
		                       gammaln(np.sum(self._lambda, 1)))

		return score
