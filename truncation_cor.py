import numpy as np

"""
This script uses simulation to assess the biasing effect of left
truncation on correlation coefficients.
"""

mcrep = 10000


def vcor(x, y):
    """
    vcor uses vectorization to calculate many correlation coefficients

    Parameters
    ----------
    x, y : two ndarray objects with the same shape

    Result
    ------
    A vector containing the Pearson correlation coefficients between
    corresponding rows of x and y.
    """

    x = x - x.mean(1)[:, None]
    x /= x.std(1)[:, None]
    y = y - y.mean(1)[:, None]
    y /= y.std(1)[:, None]

    r_est = (x * y).mean(1)

    return r_est

# Change this parameter to control the amount of truncation
t = 1

for n in 10, 20:
    for r in np.linspace(0, 0.9, 10):

        x = t + np.random.normal(size=(mcrep, n))
        y = t + np.random.normal(size=(mcrep, n))
        y = r*x + np.sqrt(1 - r**2)*y

        # Proportion of truncated values
        p0 = (np.mean(x < 0) + np.mean(y < 0)) / 2

        x = np.clip(x, 0, np.inf)
        y = np.clip(y, 0, np.inf)

        r_est = vcor(x, y)
        r_est[np.isnan(r_est)] = 0

        print("%5d %5.2f %5.2f %5.2f" % (n, r, p0, np.mean(r_est)))
