from .misc import *


def delta2_constraint(k, m2_eigvals, _) -> float:
	return (m2_eigvals[k - 1] - m2_eigvals[k]) / 2


def deltaT_constraint(k, _, m3_eigvals) -> float:
	"""
	T -> Tau refers to M3
	"""

	gamma = gamma_s(m3_eigvals, k)
	return (gamma * m3_eigvals[k - 1]) / (2 * np.sqrt(k))
