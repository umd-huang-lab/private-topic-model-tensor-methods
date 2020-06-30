def generate_epsilon_distributions(composite_epsilon):
	epsilon_distributions = {
		'config1': [
			{
				'e3': 0.33,
				'e4': 0.33,
				'e6': 0,
				'e7': 0,
				'e8': 0.33,
				'e9': 0,
			},
			# {
			# 	'e3': 0.5,
			# 	'e4': 0.25,
			# 	'e6': 0,
			# 	'e7': 0,
			# 	'e8': 0.25,
			# 	'e9': 0,
			# },
			# {
			# 	'e3': 0.25,
			# 	'e4': 0.5,
			# 	'e6': 0,
			# 	'e7': 0,
			# 	'e8': 0.25,
			# 	'e9': 0,
			# },
			# {
			# 	'e3': 0.25,
			# 	'e4': 0.25,
			# 	'e6': 0,
			# 	'e7': 0,
			# 	'e8': 0.5,
			# 	'e9': 0,
			# },
		],
		'config2': [
			{
				'e3': 0,
				'e4': 0,
				'e6': 0.5,
				'e7': 0,
				'e8': 0.5,
				'e9': 0,
			},
			# {
			# 	'e3': 0,
			# 	'e4': 0,
			# 	'e6': 0.75,
			# 	'e7': 0,
			# 	'e8': 0.25,
			# 	'e9': 0,
			# },
			# {
			# 	'e3': 0,
			# 	'e4': 0,
			# 	'e6': 0.25,
			# 	'e7': 0,
			# 	'e8': 0.75,
			# 	'e9': 0,
			# },
		],
		'config3': [
			{
				'e3': 0,
				'e4': 0,
				'e6': 0,
				'e7': 0.5,
				'e8': 0.5,
				'e9': 0,
			},
			# {
			# 	'e3': 0,
			# 	'e4': 0,
			# 	'e6': 0,
			# 	'e7': 0.75,
			# 	'e8': 0.25,
			# 	'e9': 0,
			# },
			# {
			# 	'e3': 0,
			# 	'e4': 0,
			# 	'e6': 0,
			# 	'e7': 0.25,
			# 	'e8': 0.75,
			# 	'e9': 0,
			# },
		],
		'config4': [
			{
				'e3': 0,
				'e4': 0,
				'e6': 0,
				'e7': 0,
				'e8': 0,
				'e9': 1,
			},
		],
	}

	for config, eds in epsilon_distributions.items():
		for epsilon_distribution in eds:
			for edge in epsilon_distribution:
				epsilon_distribution[edge] *= composite_epsilon

	return epsilon_distributions
