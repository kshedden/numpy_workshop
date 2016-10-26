import numpy as np

"""
This file contains a simple implementation of the k-means algorithm.
"""


def kmeans(da, k, maxiter=20):
    """
    Cluster the rows of a matrix using the k-means algorithm.

    Parameters
    ----------
    da : ndarray with two dimensions
        The data to be clustered
    k : int
        The number of clusters

    Returns
    -------
    g : ndarray with one dimension
        The cluster labels
    dist : ndarray with two dimensions
        The cluster centers, stored by row
    """

    # Number of objects to cluster
    n = da.shape[0]

    # Initial cluster centers
    ix = np.random.choice(np.arange(n), k)
    cent = da[ix, :]

    # The distance from each object to each cluster center
    dist = np.empty((n, k))

    # The closest cluster center to each object
    g = np.empty(n, dtype=np.int64)
    g1 = np.empty(n, dtype=np.int64)

    for iter in range(maxiter):

        # Calculate the distance from each object to each cluster centroid
        for j, i in enumerate(ix):
            dist[:, j] = ((da - cent[j, :])**2).sum(1)

        # The closest group to each object
        dist.argmin(1, out=g1)
        if (g1 == g).all():
            # Reached a fixed point
            break
        g, g1 = g1, g

        # Update the cluster centers
        for j in range(k):
            cent[j, :] = da[g == j, :].mean(0)

    return g, cent


def test_kmeans():

    k = 5     # Number of clusters
    p = 10    # Dimension of each data vector (object)
    n = 1000  # Number of objects to cluster

    centers = np.random.normal(size=(k, p))
    gt = np.random.randint(0, k, n)
    da = np.random.normal(size=(n, p))
    for j in range(k):
        da[gt == j, :] += centers[j, :]

    eg, ecent = kmeans(da, k)

    # Assess agreement with a concordance statistic
    m = 10000
    q = np.random.randint(0, n, size=2*m)
    q = q.reshape((m, 2))
    ii = q[:, 0] != q[:, 1]
    q = q[ii, :]

    # Calculate the mismatch rate
    mm = (eg[q[:, 0]] == eg[q[:, 1]]) & (gt[q[:, 0]] != gt[q[:, 1]])
    mm = mm.mean()

mm = test_kmeans()
