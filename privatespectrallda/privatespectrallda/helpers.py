import sys
import traceback
from functools import wraps
from itertools import permutations

import numpy as np
import scipy.stats

SEED = 42


def noiser(edge=None, symmetric=False):
	def interm(f):
		@wraps(f)
		def wrapper(self, *args, **kwargs):

			result = f(self, *args, **kwargs).copy()

			total_noise = np.zeros(result.shape)
			if edge and edge in self.variances:
				if symmetric and len(result.shape) == 3:
					dim = result.shape[0]
					perms = list(permutations(range(3)))
					noise = np.random.normal(0,
					                         np.sqrt(self.variances[edge]/len(perms)),
					                         size=(dim, dim, dim))

					for perm in perms:
						total_noise += np.transpose(noise, perm)

				else:
					total_noise += np.random.normal(0, np.sqrt(self.variances[edge]),
					                                size=result.shape)

				result += total_noise

			return result

		return wrapper

	return interm


def seed(seed_val=SEED):
	def interm(f):
		@wraps(f)
		def wrapper(*args, **kwargs):
			np.random.seed(seed_val)
			return f(*args, **kwargs)

		return wrapper

	return interm


def proj_l1_simplex(vec, l1_simplex_boundary=1.0):
	shape = vec.shape

	try:
		vec_sorted = -np.sort(-vec)
		vec_shifted = (vec_sorted - (vec_sorted.cumsum() - l1_simplex_boundary) / range(1, len(vec) + 1))
		
		try:
			rho = np.squeeze(np.where(vec_shifted > 0)).max() + 1
		except:
			vec = -vec
			vec_sorted = -np.sort(-vec)
			vec_shifted = (vec_sorted - (vec_sorted.cumsum() - l1_simplex_boundary) / range(1, len(vec) + 1))
			rho = np.squeeze(np.where(vec_shifted > 0)).max() + 1
		theta = (vec_sorted[:rho].sum() - l1_simplex_boundary) / rho
		retval = np.maximum(vec - theta, 0)
	except Exception as e:
		print(e)
		exc_info = sys.exc_info()
		traceback.print_exception(*exc_info)
		print('Simplex projection failed...')
		g = np.random.rand(*shape)
		g = g / g.sum()
		retval = g

	assert abs(retval.sum() - 1) < 1e-10
	return retval


def euclidean_proj_simplex(v, s=1):
	""" Compute the Euclidean projection on a positive simplex
	Solves the optimisation problem (using the algorithm from [1]):
		min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
	Parameters
	----------
	v: (n,) numpy array,
		n-dimensional vector to project
	s: int, optional, default: 1,
		radius of the simplex
	Returns
	-------
	w: (n,) numpy array,
		Euclidean projection of v on the simplex
	Notes
	-----
	The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
	Better alternatives exist for high-dimensional sparse vectors (cf. [1])
	However, this implementation still easily scales to millions of dimensions.
	References
	----------
	[1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
		John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
		International Conference on Machine Learning (ICML 2008)
		http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
	"""
	assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
	n, = v.shape  # will raise ValueError if v is not 1-D
	# check if we are already on the simplex
	if v.sum() == s and np.alltrue(v >= 0):
		# best projection: itself!
		return v
	# get the array of cumulative sums of a sorted (decreasing) copy of v
	u = np.sort(v)[::-1]
	cssv = np.cumsum(u)
	# get the number of > 0 components of the optimal solution
	rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
	# compute the Lagrange multiplier associated to the simplex constraint
	theta = (cssv[rho] - s) / (rho + 1.0)
	# compute the projection by thresholding v using theta
	w = (v - theta).clip(min=0)
	return w


def generate_document_word_counts(alpha, beta, n_docs, n_words_in_doc=(100_000, 200_000), seed=42):
	"""
	:param alpha: k*1 input
	:param beta: n_words*k input
	:param n_docs: Number of documents.
	:param n_words_in_doc: Range of number of words in a given doc
	:return: n_docs*n_words document_word_counts matrix
	"""
	np.random.seed(seed)
	k = len(alpha)
	n_words, also_k = beta.shape
	n_words_in_doc_min, n_words_in_doc_max = n_words_in_doc

	# Draw n_docs samples from a Dirichlet characterized by alpha - n_docs*k matrix.
	d_t_weights = scipy.stats.dirichlet.rvs(alpha, size=n_docs)
	fail_notprob = np.argwhere(d_t_weights.sum(axis=1) != 1).flatten().tolist()
	fail_lesszero = np.argwhere(d_t_weights < 0)[:, 0].flatten().tolist()
	fail_nans = np.argwhere(np.isnan(d_t_weights))[:, 0].flatten().tolist()

	changerows = set(fail_notprob + fail_lesszero + fail_nans)
	while len(changerows) > 0:
			print(f'{len(changerows)} to fix')
			d_t_weights[list(changerows)] = scipy.stats.dirichlet.rvs(alpha, size=len(changerows))
			fail_notprob = np.argwhere(d_t_weights.sum(axis=1) != 1).flatten().tolist()
			fail_lesszero = np.argwhere(d_t_weights < 0)[:, 0].flatten().tolist()
			fail_nans = np.argwhere(np.isnan(d_t_weights))[:, 0].flatten().tolist()
			changerows = set(fail_notprob + fail_lesszero + fail_nans)

	# Draw word counts from a range for each document - n_docs*1 matrix
	d_n_words = np.random.randint(n_words_in_doc_min,
	                              n_words_in_doc_max,
	                              size=n_docs)

	d_w_counts = np.array([scipy.stats.multinomial.rvs(d_n_word, beta.dot(d_t_weight)) for d_n_word, d_t_weight in
	                       zip(d_n_words, d_t_weights)])

	return d_w_counts
