
import numpy as np
from scipy.stats import multivariate_normal, bernoulli

def em_algorithm(y_matrix, K):
    p = y_matrix.shape[1]
    mu_components = np.random.sample(K*p).reshape(K,p)
    sigma_components = np.array([np.identity(p) for i in range(K)])
    phi = np.ones(K) / K
    eps = 1e-4
    max_iter = 100
    iter = 0
    error = 1
    loglik = loglikelihood(y_matrix, mu_components, sigma_components, phi)
    while (iter < max_iter) or (error < eps): 
        phi_posterior = estep(y_matrix, mu_components,
                              sigma_components, phi)

        m_results = mstep(y_matrix, phi_posterior)
        mu_components = m_results["mu_data"]
        sigma_components = m_results["sigma_data"]
        phi = m_results["phi"]

        new_loglik = loglikelihood(y_matrix, mu_components, sigma_components, phi)
        error = np.abs(new_loglik - loglik)
        iter += 1



def loglikelihood(y_matrix, mu_matrix, sigma_list, phi):
    K = len(sigma_list)
    n = y_matrix.shape[0]
    loglik = 0
    for j in range(K):
        for i in range(n):
            mv_pdf = multivariate_normal.pdf(y_matrix[i,:], 
                                            mean = mu_matrix[j,:], 
                                            cov = sigma_list[j])
            loglik += np.log(phi[j]*mv_pdf)
    return loglik


def estep(y_matrix, mu_matrix, sigma_list, phi):
    K = len(sigma_list)
    n = y_matrix.shape[0]
    phi_posterior = np.zeros(n*K).reshape(n,K)
    for i in range(n):
        for j in range(K):
            mv_pdf = multivariate_normal.pdf(y_matrix[i,:], 
                                            mean = mu_matrix[j,:], 
                                            cov = sigma_list[j])
            phi_posterior[i,j] = phi[j]*mv_pdf
        phi_posterior[i,:] = phi_posterior[i,:] / phi_posterior[i,:].sum()
    return phi_posterior


# y: p x 1
# y_matrix: n x p
# phi_posterior_matrix: n x K
def mstep(y_matrix, phi_posterior_matrix):
    phi = phi_posterior_matrix.mean(axis = 0)
    phi_sum = phi_posterior_matrix.sum(axis = 0)

    mu_list = []
    K = phi_posterior_matrix.shape[1]
    n = phi_posterior_matrix.shape[0]
    p = y_matrix.shape[1]
    
    for j in range(K):
        mu_j = np.zeros(p)
        for i in range(n):
            mu_j += y_matrix[i,:] * \
                phi_posterior_matrix[i,j]  
        
        mu_j = mu_j / phi_sum[j]      
        mu_list.append(mu_j) 
    
    mu_matrix = np.array(mu_list) # K x p

    sigma_list = []
    for j in range(K):
        sigma = np.zeros(p*p).reshape(p,p)
        for i in range(n):
            rvec = (mu_matrix[j,:] - y_matrix[i,:]).transpose()
            sigma += phi_posterior_matrix[i,j] * \
                  rvec.reshape(p,1) @ rvec.reshape(1,p)
        sigma = sigma / phi_sum[j]
        sigma_list.append(sigma)

    return {
        "mu_data":np.array(mu_list),
        "sigma_data":np.array(sigma_list),
        "phi":phi
    }

