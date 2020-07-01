# Intro: An end-to-end Differentially Private Latent Dirichlet Allocation Using a Spectral Algorithm

We provide an end-to-end differentially private spectral algorithm for learning LDA, based on matrix/tensor decompositions, and establish theoretical guarantees on utility/consistency of the estimated model parameters. We represent the spectral algorithm as a computational graph. Noise can be injected along the edges of this graph to obtain differential privacy. We identify \emph{subsets of edges}, named ``configurations'', such that adding noise to all edges in such a subset guarantees differential privacy of the end-to-end spectral algorithm. We characterize the sensitivity of the edges with respect to the input and thus estimate the amount of noise to be added to each edge for any required privacy level. We then characterize the utility loss  for each configuration as a function of injected noise.  Overall, by combining the sensitivity and utility characterization, we obtain an end-to-end differentially private spectral algorithm for LDA and identify which configurations outperform others under specific regimes. We are the first to achieve utility guarantees under a required level of differential privacy for learning in LDA. We additionally show that our method systematically outperforms differentially private variational inference.

# Citation
@inproceedings{huang2018learning,
  title={An end-to-end Differentially Private Latent Dirichlet Allocation Using a Spectral Algorithm},
  author={DeCarolis, Christopher and Ram, Mukul and Esmaeili, Seyed A and Wang, Yu-Xiang and Huang, Furong},
  booktitle={International Conference on Machine Learning},
  year={2020}
}

# Experimental Setup

Build docker:
docker-compose kill && docker-compose build && docker-compose up -d


