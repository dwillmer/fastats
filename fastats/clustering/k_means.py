
import numpy as np
from numba import jit


@jit
def _predict(point, means) -> int:
    d_min = np.inf
    for j in range(len(means)):
        d = np.sum((means[j] - point)**2)

        if d < d_min:
            prediction = j
            d_min = d
    return prediction


def k_means(data, k=5, iterations=100):
    """
    A simple implementation of k-means clustering,
    which takes any arbitrary dimensional data and
    returns the cluster index for each item in the
    first dimension.
    """
    assignments = np.zeros((data.shape[0], 1), dtype=np.int32)

    # select random points as means
    indices = np.random.choice(data.shape[0], k)

    # TODO : np.take doesn't jit yet, potentially
    # due to the axis arg
    means = np.take(data, indices, axis=0)

    for _ in range(iterations):
        for i in range(len(data)):
            assignments[i] = _predict(data[i], means)
        for j in range(k):
            cluster = data[(assignments == j)[:, 0]]
            if cluster.shape[0] > 0:
                # TODO : np.mean with kwarg also won't jit.
                means[j, :] = np.mean(cluster, axis=0)

    return assignments


