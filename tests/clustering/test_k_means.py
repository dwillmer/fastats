
import numpy as np
from numpy.testing import assert_array_equal

from fastats.core.ast_transforms.convert_to_jit import convert_to_jit
from fastats.clustering.k_means import k_means
from fastats.utilities import SeedContext

k_means_jit = convert_to_jit(k_means)

# Randomly generated using np.random.random((10, 2))
DATA_2D = np.array([
    [0.68216138, 0.11072853],
    [0.65435081, 0.67607348],
    [0.49031091, 0.08890905],
    [0.5517702, 0.57918261],
    [0.8814244, 0.24188243],
    [0.37134984, 0.51795646],
    [0.89444973, 0.3481619],
    [0.8463001, 0.91651841],
    [0.48509058, 0.06978188],
    [0.34843015, 0.43412657]
])


def test_basic_sanity():
    with SeedContext(1):
        assigns = k_means(DATA_2D, k=5)

    expected = np.array([4., 3., 1., 3., 4., 2., 4., 0., 1., 2.])
    assert_array_equal(assigns[:, 0], expected)


def test_k_means_jit_compiles():
    with SeedContext(1):
        assigns_jit = k_means_jit(DATA_2D, k=5)

    expected = np.array([4., 3., 1., 3., 4., 2., 4., 0., 1., 2.])
    assert_array_equal(assigns_jit[:, 0], expected)


def test_3d_input():
    # Randomly generated using np.random.random((10, 2, 2))
    data = np.array([
        [[0.6510911, 0.00658404],
         [0.6372308, 0.33078461]],
        [[0.3095511, 0.76275762],
         [0.02859159, 0.17715778]],
        [[0.23854938, 0.59996384],
         [0.85231901, 0.20272616]],
        [[0.75336878, 0.40487868],
         [0.02554637, 0.87820604]],
        [[0.24171355, 0.42849439],
         [0.63835978, 0.14192977]],
        [[0.4683422, 0.61078412],
         [0.85691167, 0.55485588]],
        [[0.00680853, 0.46833516],
         [0.12561526, 0.03754262]],
        [[0.60177049, 0.24180041],
         [0.65914184, 0.61669609]],
        [[0.29884666, 0.81695866],
         [0.75265079, 0.59522059]],
        [[0.51705579, 0.49255249],
         [0.10918967, 0.70732275]]
    ])

    with SeedContext(1):
        assigns = k_means(data, k=3)

    expected = np.array([0., 2., 0., 2., 0., 1., 2., 0., 1., 2.])
    assert_array_equal(assigns[:, 0], expected)

    with SeedContext(1):
        assigns_4 = k_means(data, k=4)

    expected_4 = np.array([0., 2., 0., 2., 0., 3., 2., 0., 1., 2.])
    assert_array_equal(assigns_4[:, 0], expected_4)


if __name__ == '__main__':
    import pytest
    pytest.main()
