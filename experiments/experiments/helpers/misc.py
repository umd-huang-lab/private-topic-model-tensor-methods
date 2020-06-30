import numpy as np
from autodp import privacy_calibrator


def gamma_s(m3_eigvals, k):
	return min((m3_eigvals[i] - m3_eigvals[i + 1]) / 4 for i in range(k - 2))


def tau(edge, edge_epsilon, edge_delta):
	"""
	Adjust privacy budget based on privacy calibration based on DP release of parameters.
	"""

	if edge in ['e6']:
		epsilon = edge_epsilon / 2
		delta = edge_delta / 2
	elif edge in ['e7']:
		epsilon = edge_epsilon / 3
		delta = edge_delta / 3
	elif edge in ['e9']:
		epsilon = edge_epsilon / 4
		delta = edge_delta / 4
	elif edge in ['e3', 'e4', 'e8']:
		epsilon = edge_epsilon
		delta = edge_delta
	else:
		raise Exception(f'{edge} not supported.')

	if epsilon == 0:
		return 0

	if epsilon < 1:
		result = privacy_calibrator.classical_gaussian_mech(epsilon, delta)
		return result['sigma'] if not isinstance(result, int) else result
	else:
		return privacy_calibrator.gaussian_mech(epsilon, delta)['sigma']


def zeroed_edges():
	return {
		'e3': 0,
		'e4': 0,
		'e6': 0,
		'e7': 0,
		'e8': 0,
		'e9': 0,
	}
