import pandas
import numpy as np
from scipy.spatial.distance import cdist

# Symmetric Add KL Divergence // Log Score
def symmetric_kl_divergence(model_1, model_2, N):
    model_1_samples = model_1.sample(N)
    model_1_log_pdf = model_1.log_prob(model_1_samples)
    model_2_log_pdf = model_2.log_prob(model_1_samples)

    kl_12 = (model_1_log_pdf - model_2_log_pdf).mean()

    model_2_samples = model_1.sample(N)
    model_1_log_pdf = model_1.log_prob(model_2_samples)
    model_2_log_pdf = model_2.log_prob(model_2_samples)

    kl_21 = (model_2_log_pdf - model_1_log_pdf).mean()

    return kl_12 + kl_21


# Add Energy Score
def energy_score(model_1, model_2, N):
    model_1_samples = model_1.sample(N)
    model_2_samples = model_2.sample(N)

    # Vectorized euclidean distances
    d11 = cdist(model_1_samples, model_1_samples, metric='euclidean')
    d22 = cdist(model_2_samples, model_2_samples, metric='euclidean')
    d12 = cdist(model_1_samples, model_2_samples, metric='euclidean')

    # Energy score sums
    e11 = np.sum(d11)
    e22 = np.sum(d22)
    e12 = np.sum(d12)

    # Energy score - simplifies due to same sample sizes
    return (2*e12 - e11 - e22)/(2*N)  