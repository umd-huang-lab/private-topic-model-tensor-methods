import numpy as np
from scipy.stats import laplace

from .misc import *

#####################
# config 1: e3,e4,e8
# config 2: e6,e8
# config 3: e7,e8
# config 4: e9
# TODOS: check that m3_eigvals are the whitened eigenvals
#####################
mythreshold = 10e-2
machine_threshold = np.finfo(np.float32).eps

def delta2(d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, edge_epsilon, edge_delta):
	N = d_w_counts.shape[0]
	
	term1 = 2 / N
	term2 = (alpha0 / (alpha0 + 1)) * (4 / N)
	return (term1 + term2) * alpha0 * (alpha0+1)

e2_sensitivity = delta2
e4_sensitivity = delta2
e8_sensitivity = delta2

def delta3(d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, edge_epsilon, edge_delta):
	N = d_w_counts.shape[0]
	term1 = 2 / N
	term2 = (alpha0 / (alpha0 + 2)) * (4 / N)
	term3 = 12 * (alpha0 ** 2 / ((alpha0 + 1) * (alpha0 + 2))) * ((N - 1) / (N * (N - 2)))
	return (term1 + term2 + term3) * alpha0 * (alpha0+1) * (alpha0+2) / 2

e3_sensitivity = delta3

'''def e4_sensitivity(d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, edge_epsilon, edge_delta):
	print('entered e4 sensitivity')
	N = d_w_counts.shape[0]
	epsilon1 = edge_epsilon
	delta1 = edge_delta


	delta2_ = delta2(d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, 0, 0)

	sigma_k = m2_eigvals[k - 1]
	sigma_k_hat = sigma_k + laplace.rvs(loc=0, scale=1 / (N * max(epsilon1,mythreshold)), size=1)[0]
	sigma_k_hat = max(sigma_k_hat - (2 / (N * max(epsilon1,mythreshold))) * np.log10(1 / (2 * max(delta1,mythreshold))), mythreshold)


	sigma_k_hat_minus_delta2 = max(sigma_k_hat - delta2_, mythreshold) #np.finfo(np.float32).eps
	
	return (np.sqrt(2*k) * delta2_ / ( sigma_k_hat *np.sqrt(sigma_k_hat_minus_delta2) )) / np.sqrt(alpha0 * (alpha0+1))'''


def e6_sensitivity(d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, edge_epsilon, edge_delta):
	if edge_epsilon == 0:
		return 0

	N = d_w_counts.shape[0]

	epsilon1 = edge_epsilon
	delta1 = edge_delta
	delta2_ = delta2(d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, 0, 0)
	delta3_ = delta3(d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, 0, 0)

	term1p1 = 1
	term1p2 = ((6 * alpha0) / (alpha0 + 2)) * (N / (N - 1))
	term1p3 = (6 * (alpha0 ** 2) * (N ** 3)) / ((alpha0 + 1) * (alpha0 + 2) * N * (N - 1) * (N - 2))
	term1 = term1p1 + term1p2 + term1p3

	sigma_k = m2_eigvals[k - 1]
	sigma_k_hat = sigma_k + laplace.rvs(loc=0, scale=delta2_/max(epsilon1,machine_threshold), size=1)[0]
	sigma_k_hat = max(sigma_k_hat - (2 / (N * max(epsilon1,machine_threshold))) * np.log10(1 / (2 * max(delta1,machine_threshold))), mythreshold)

	

	sigma_k_hat_minus_delta2 = max(sigma_k_hat - delta2_, mythreshold) # np.finfo(np.float32).eps

	term2 = (((2 * k) ** 1.5) * (delta2_ ** 3)) / (sigma_k_hat ** 3) * ((sigma_k_hat_minus_delta2) ** 1.5)
	term3 = (delta3_ * (k ** 1.5)) / (sigma_k_hat_minus_delta2 ** 1.5)

	return (term1 * term2 + term3) * (alpha0+2) * 0.5 / np.sqrt(alpha0* (alpha0+1))


def e7_sensitivity(d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, edge_epsilon, edge_delta):
	m3_eigvals = np.sort(m3_eigvals)[::-1] # make sure that eigenvalues are in sorted order.
	if edge_epsilon == 0:
		return 0

	N = d_w_counts.shape[0]

	epsilon1_prime = edge_epsilon / 3
	delta1_prime = edge_delta / 3
	delta3_ = delta3(d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, 0, 0)

	gamma_s_ = gamma_s(m3_eigvals, k)
	gamma_s_hat = gamma_s_ + laplace.rvs(loc=0, scale=delta3_ /max(epsilon1_prime,machine_threshold), size=1)[0]
	gamma_s_hat = max(gamma_s_hat - (2 / (N * max(epsilon1_prime,machine_threshold))) * np.log10(1 / (2 * max(delta1_prime,machine_threshold))), mythreshold)

	deltaT_ = deltaT(
		d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, edge_epsilon * 2 / 3, edge_delta * 2 / 3
	)

	return (2 * np.sqrt(k) * deltaT_ / gamma_s_hat) #* (alpha0+2) * 0.5 / np.sqrt(alpha0* (alpha0+1))


deltaT = e6_sensitivity
deltaMu = e7_sensitivity

'''def e8_sensitivity(d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, edge_epsilon, edge_delta):
	N = d_w_counts.shape[0]
	epsilon1 = edge_epsilon
	delta1 = edge_delta



	delta2_ = delta2(d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, 0, 0)


	sigma_k = m2_eigvals[k - 1]
	sigma_k_hat = sigma_k + laplace.rvs(loc=0, scale=1 / (N * max(epsilon1,mythreshold)), size=1)[0]
	sigma_k_hat = max(sigma_k_hat - (2 / (N * max(epsilon1,mythreshold))) * np.log10(1 / (2 * max(delta1,mythreshold))), mythreshold)

	sigma_one = m2_eigvals[0]
	sigma_one_hat = sigma_one + laplace.rvs(loc=0, scale=1 / (N * max(epsilon1,mythreshold)), size=1)[0]
	sigma_one_hat = max(sigma_one_hat - (2 / (N * max(epsilon1,mythreshold))) * np.log10(1 / (2 * max(delta1,mythreshold))), mythreshold)

	
	return (np.sqrt(sigma_one_hat) * delta2_ / sigma_k_hat ) * np.sqrt(alpha0 * (alpha0+1))'''

	
	

def e9_sensitivity(d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, edge_epsilon, edge_delta):
	"""
	We're only using the last term.
	"""
	m3_eigvals = np.sort(m3_eigvals)[::-1] # make sure that eigenvalues are in sorted order.
	N = d_w_counts.shape[0]

	if edge_epsilon == 0:
		return 0

	epsilon1 = edge_epsilon / 4
	delta1 = edge_delta / 4
	delta3_ = delta3(d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, 0, 0)
	# sigma_k = m2_eigvals[k - 1]
	# sigma_k_hat = sigma_k + laplace.rvs(loc=0, scale=1 / (N * epsilon1), size=1)[0]
	# sigma_k_hat = max(sigma_k_hat - (2 / (N * epsilon1)) * np.log2(1 / (2 * delta1)), mythreshold)

	sigma_1 = m2_eigvals[0]
	sigma_1_hat = sigma_1 + laplace.rvs(loc=0, scale=delta3_/ max(epsilon1,machine_threshold), size=1)[0]
	sigma_1_hat = max(sigma_1_hat - (2 / (N * max(epsilon1,machine_threshold))) * np.log10(1 / (2 * max(delta1,machine_threshold))), mythreshold)


	# term1 = ((alpha0 + 2) / (2 * np.sqrt((alpha0 + 1) * alpha0))) * ...
	# term2 = ((alpha0 + 2) / (2 * np.sqrt((alpha0 + 1) * alpha0))) * ...
	term3 = ((alpha0 + 2) / (2 * np.sqrt((alpha0 + 1) * alpha0))) * np.sqrt(sigma_1_hat + delta2(
		d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, 0, 0
	)) * deltaMu(
		d_w_counts, k, alpha0, m2_eigvals, m3_eigvals, edge_epsilon * 3 / 4, edge_delta * 3 / 4
	)

	return term3 * alpha0 * (alpha0+1) * (alpha0+2) * 0.5


# def config1_sensitivity(d_w_counts, k, m2_eigvals, m3_eigvals, edge_epsilon, edge_delta):
# 	N = d_w_counts.shape[0]
# 	return 1 / N
#
#
# def config2_sensitivity(d_w_counts, k, m2_eigvals, m3_eigvals, edge_epsilon, edge_delta):
# 	epsilon1 = edge_epsilon / 2
# 	delta1 = edge_delta / 2
#
# 	if epsilon1 == 0:
# 		return 0
#
# 	N = d_w_counts.shape[0]
#
# 	sigma_k = m2_eigvals[k - 1]
# 	sigma_k_hat = sigma_k + laplace.rvs(loc=0, scale=1 / (N * epsilon1), size=1)[0]
# 	sigma_k_hat = max(sigma_k_hat - (2 / (N * epsilon1)) * np.log2(1 / (2 * delta1)), mythreshold)
#
# 	numerator = k ** (3 / 2)
# 	denominator = N * (sigma_k_hat ** (3 / 2))
#
# 	return numerator / denominator
#
#
# def config3_sensitivity(d_w_counts, k, m2_eigvals, m3_eigvals, edge_epsilon, edge_delta):
# 	epsilon1 = epsilon1_prime = edge_epsilon / 3
# 	delta1 = delta1_prime = edge_delta / 3
#
# 	if epsilon1 == 0:
# 		return 0
#
# 	N = d_w_counts.shape[0]
#
# 	sigma_k = m2_eigvals[k - 1]
# 	sigma_k_hat = sigma_k + laplace.rvs(loc=0, scale=1 / (N * epsilon1), size=1)[0]
# 	sigma_k_hat = max(sigma_k_hat - (2 / (N * epsilon1)) * np.log2(1 / (2 * delta1)), mythreshold)
#
# 	gamma_s_ = gamma_s(m3_eigvals, k)
# 	gamma_s_hat = gamma_s_ + laplace.rvs(loc=0, scale=1 / (N * epsilon1_prime), size=1)[0]
# 	gamma_s_hat = max(gamma_s_hat - (2 / (N * epsilon1_prime)) * np.log2(1 / (2 * delta1_prime)), mythreshold)
#
# 	numerator = k ** 2
# 	denominator = gamma_s_hat * N * (sigma_k_hat ** (3 / 2))
#
# 	return numerator / denominator
#
#
# def config4_sensitivity(d_w_counts, k, m2_eigvals, m3_eigvals, edge_epsilon, edge_delta):
# 	epsilon1 = epsilon1_prime = edge_epsilon / 3
# 	delta1 = delta1_prime = edge_delta / 3
#
# 	if epsilon1 == 0:
# 		return 0
#
# 	N = d_w_counts.shape[0]
#
# 	sigma_k = m2_eigvals[k - 1]
# 	sigma_k_hat = sigma_k + laplace.rvs(loc=0, scale=1 / (N * epsilon1), size=1)[0]
# 	sigma_k_hat = max(sigma_k_hat - (2 / (N * epsilon1)) * np.log2(1 / (2 * delta1)), mythreshold)
#
# 	gamma_s_ = gamma_s(m3_eigvals, k)
# 	gamma_s_hat = gamma_s_ + laplace.rvs(loc=0, scale=1 / (N * epsilon1_prime), size=1)[0]
# 	gamma_s_hat = max(gamma_s_hat - (2 / (N * epsilon1_prime)) * np.log2(1 / (2 * delta1_prime)), mythreshold)
#
# 	numerator = (k ** 2) * np.sqrt(m2_eigvals[0])
# 	denominator = gamma_s_hat * N * (sigma_k_hat ** (3 / 2))
#
# 	return numerator / denominator


SENSITIVITIES = {
	'e3': e3_sensitivity,
	'e4': e4_sensitivity,
	'e6': e6_sensitivity,
	'e7': e7_sensitivity,
	'e8': e8_sensitivity,
	'e9': e9_sensitivity,
}


def config1_utility_loss_bound(k, m2_eigvals, m3_eigvals, d, N, composite_epsilon, delta) -> float:
	gamma = gamma_s(m3_eigvals, k)

	t1p1 = np.sqrt(m2_eigvals[0] * k) / gamma
	t1p2 = ((np.sqrt(d) * tau(composite_epsilon, delta)) / (N * (m2_eigvals[k - 1] ** 1.5))) ** 3
	t1p3 = ((np.sqrt(d) * tau(composite_epsilon, delta)) / (N * (m2_eigvals[k - 1] ** 1.5)))

	t1 = t1p1 * (t1p2 + t1p3)

	t2 = np.sqrt(m2_eigvals[0] * d * tau(composite_epsilon, delta)) / (m2_eigvals[k - 1] * N)

	t3p1 = np.sqrt(m2_eigvals[0] + (np.sqrt(d) * tau(composite_epsilon, delta) / N)) * np.sqrt(k) / gamma_s(m3_eigvals,
	                                                                                                        k)
	t3p2 = t1p2
	t3p3 = t1p3

	t3 = t3p1 * (t3p2 + t3p3)

	return t1 + t2 + t3


def config2_utility_loss_bound(k, m2_eigvals, m3_eigvals, d, N, composite_epsilon, delta) -> float:
	gamma = gamma_s(m3_eigvals, k)

	t1 = (np.sqrt(m2_eigvals[0] * (k ** 2.5)) * tau(composite_epsilon, delta)) / (
			gamma * N * (m2_eigvals[k - 1] ** 1.5))
	t2 = np.sqrt(m2_eigvals[0] * d) * tau(composite_epsilon, delta) / (m2_eigvals[k - 1] * N)
	t3 = np.sqrt(m2_eigvals[0] + ((np.sqrt(d) * tau(composite_epsilon, delta)) / N)) * (
			(k ** 2.5) * tau(composite_epsilon, delta)) / (
			     gamma * N * (m2_eigvals[0] ** 1.5))

	return t1 + t2 + t3


def config3_utility_loss_bound(k, m2_eigvals, m3_eigvals, d, N, composite_epsilon, delta) -> float:
	gamma = gamma_s(m3_eigvals, k)
	t1 = (np.sqrt(m2_eigvals[0] * (k ** 2.5)) * tau(composite_epsilon, delta)) / (
			gamma * N * (m2_eigvals[k - 1] ** 1.5))
	t2 = np.sqrt(m2_eigvals[0] * d) * tau(composite_epsilon, delta) / (m2_eigvals[k - 1] * N)
	t3 = np.sqrt(m2_eigvals[0] + ((np.sqrt(d) * tau(composite_epsilon, delta)) / N)) * (
			(k ** 2) * tau(composite_epsilon, delta)) / (
			     gamma * N * (m2_eigvals[0] ** 1.5))

	return t1 + t2 + t3


def config4_utility_loss_bound(k, m2_eigvals, m3_eigvals, d, N, composite_epsilon, delta) -> float:
	return (np.sqrt(m2_eigvals[0] * d) * (k ** 2) * tau(composite_epsilon, delta)) / (
			gamma_s(m3_eigvals, k) * N * (m2_eigvals[k - 1] ** 1.5))
