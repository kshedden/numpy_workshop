import numpy as np

"""
Assess the biasing effect of left truncation on correlation coefficients.
"""


mcrep = 10000

def vcor(x, y):

    x = x - x.mean(1)[:, None]
    x /= x.std(1)[:, None]
    y = y - y.mean(1)[:, None]
    y /= y.std(1)[:, None]

    r_est = (x * y).mean(1)
    r_est[np.isnan(r_est)] = 0

    return r_est


for n in 5,10,20:
    for r in np.linspace(0, 0.9, 10):

        x = 1 + np.random.normal(size=(mcrep, n))
        y = 1 + np.random.normal(size=(mcrep, n))
        y = r*x + np.sqrt(1 - r**2)*y

        x[x < 0] = 0
        y[y < 0] = 0

        r_est = vcor(x, y)

        print("%5d %5.2f %5.2f" % (n, r, np.mean(r_est)))
