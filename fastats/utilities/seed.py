
from numpy import random


class SeedContext:
    def __init__(self, seed):
        self._original = None
        self.seed = seed

    def __enter__(self):
        self._original = random.get_state()
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        random.set_state(self._original)
