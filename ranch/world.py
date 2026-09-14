"""The generative world: owns the sampling noise and the random stream. Learners never see
stimulus vectors, only glimpses."""
import numpy as np


class World:
    def __init__(self, sigma_true=0.0, seed=None):
        self.sigma_true = float(sigma_true)
        self.rng = np.random.default_rng(seed)

    @property
    def noiseless(self):
        return self.sigma_true == 0.0

    def glimpse(self, stimulus):
        """One noisy sample of a stimulus vector: z = stimulus + N(0, sigma_true^2 I).
        (Draw order matches granch_fast.metrics.infant_trajectories so seeded runs agree.)"""
        v = np.asarray(stimulus, dtype=float)
        if self.noiseless:
            return v.copy()
        return v + self.rng.normal(0.0, self.sigma_true, size=len(v))
