import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt


def produce_L(num_dimensions, sigma):
    x = np.linspace(0, 1, num_dimensions)
    y = np.linspace(0, 1, num_dimensions)
    L = np.zeros((num_dimensions**2, num_dimensions**2))
    for j in range(num_dimensions**2):
        for i in range(num_dimensions**2):
            if j > i:
                L[i, j] = L[j, i]
                continue
            L[i, j] = np.exp(-0.5 * sigma**(-2) * ((x[i % num_dimensions]-x[j % num_dimensions])**2 + (y[i // num_dimensions] -y[j // num_dimensions])**2))
            # We count the vertices across first and then up. So labels for the points in the grid below ares
            # . . . 6 7 8
            # . . . 3 4 5
            # . . . 0 1 2
            # In this case Num_dimensions = 3. From number 7 we see x_coord = 7%3 = 1, y_coord = 7//3 = 2 => (1,2)
    return L


def dpp_sample(num_dimensions, sigma):
    print("producing L")
    L = produce_L(num_dimensions, sigma)
    print("finished producing L")
    print("started calculating eigenvalues/eigenvectors")
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    print("finished calculating eigenvalues/eigenvectors")
    J = []
    N = len(eigenvalues)
    for n in range(N):
        p = np.random.uniform()
        if 0 <= p <= eigenvalues[n] / (eigenvalues[n] + 1):
            J.append(n)
    V = eigenvectors[:, J]
    Y = []

    basis_vecs = np.identity(N)
    count = 0
    while len(V[0]) > 0:
        print("count", count)
        i = 0
        q = np.random.uniform()
        prob_sum = 0
        while q > prob_sum:
            prob_sum += 1/len(V[0]) * sum(np.matmul(V[:, col], basis_vecs[:, i])**2 for col in range(len(V[0])))
            if q > prob_sum:
                i += 1
        Y.append(i)

        #V columns are orthonormal basis of V, span V
        #e, vector we need to find a subspace orthogonal to
        e = basis_vecs[:, i]
        dotted = np.matmul(e.T, V)
        #dotted is the ith row of V
        dotted_as_accepted_matrix = np.array([list(dotted)])
        null_vector_space = null_space(dotted_as_accepted_matrix) #Complains about array having size (n, )

        assert null_vector_space.shape[1] == dotted_as_accepted_matrix.shape[1] - 1

        V = np.matmul(V, null_vector_space)
        count += 1
    return Y

num_dimensions = 50
sigma = 0.1  #only values in [0,0.8] produce L[i,j] which differ much from 1.
             #good range [0.05, 0.2]

sampled_numbers = dpp_sample(num_dimensions, sigma)
#DPP points
x_points = []
y_points = []
for number in sampled_numbers:
    #Undoes process of allocating numbers given in produce_L
    x_coordinate = number % num_dimensions
    x_points.append(x_coordinate)
    y_coordinate = number // num_dimensions
    y_points.append(y_coordinate)
num_dpp_points = len(x_points)

#Independent points
x = np.linspace(0, 1, num_dimensions)
y = np.linspace(0, 1, num_dimensions)
xx, yy = np.meshgrid(x, y)

sample = np.zeros(num_dimensions**2).astype(bool)
sample[: num_dpp_points] = True
np.random.shuffle(sample)
sample = sample.reshape((num_dimensions, num_dimensions))

#sets  double plot output
fig, axes = plt.subplots(1, 2)

axes[0].scatter(x_points, y_points, s = 3)
axes[0].set_title("DPP")
axes[1].scatter(xx[sample], yy[sample], s = 3)
axes[1].set_title("Independent")

axes[0].set_aspect('equal', 'box')
axes[1].set_aspect('equal', 'box')

axes[0].set_xlim((0, num_dimensions))
axes[0].set_ylim((0, num_dimensions))
axes[1].set_xlim((0, 1))
axes[1].set_ylim((0, 1))

axes[0].set_xticklabels([])
axes[0].set_yticklabels([])
axes[1].set_xticklabels([])
axes[1].set_yticklabels([])

axes[0].tick_params(axis=u'both', which=u'both',length=0)
axes[1].tick_params(axis=u'both', which=u'both',length=0)


plt.show()