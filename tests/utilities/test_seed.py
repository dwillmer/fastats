
from numpy import random

from fastats.utilities import SeedContext


def test_basic_sanity():
    """
    This is quite a subtle test -
    The numpy RNG uses a Mersenne Twister, which
    has internal state that can represent more
    integers than we can in small integers.

    As a result, when we get the seed state
    we are returned a tuple consisting of the
    Mersenne Twister type and an array of seed
    values.

    When we set a new seed, we just set the
    first item in the array, so we care about
    the second item (the array), and the first
    index of that, hence the [1][0] indexing
    below.
    """
    orig = random.get_state()[1][0]

    with SeedContext(1):
        assert random.get_state()[1][0] == 1

    assert random.get_state()[1][0] == orig


if __name__ == '__main__':
    import pytest
    pytest.main()
