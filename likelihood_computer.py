import numpy as np
#np.seterr(all='raise')


def log_likelihood(theta, L, feature_Y_sum):
    first_term = np.sum(np.matmul(np.transpose(theta), feature_Y_sum))
    sign, logdet = np.linalg.slogdet(L + np.identity(L.shape[0])) #takes log of determinant
    return first_term - logdet


def grad_log_likelihood(L, features, feature_Y_sum):
    if not np.all(np.isfinite(L)):
        print("hello")
    eigenvalues_L, eigenvectors_L = np.linalg.eigh(L)  #L symmetric so use eigh
    eigenvalues_K = eigenvalues_L / (eigenvalues_L + 1) #divides element-wise
    K_diag = np.matmul(np.square(eigenvectors_L), eigenvalues_K)
    return feature_Y_sum - np.matmul(features, K_diag)


def compute_conditional_kernel(theta, features, S):
    quality_vec = np.exp(0.5 * np.matmul(np.transpose(theta), features))  # this gives form [q_1, q_2, ..., q_number_of_sentences]
    Q = np.diag(quality_vec)
    L = np.matmul(np.matmul(Q, S), Q)
    return L
