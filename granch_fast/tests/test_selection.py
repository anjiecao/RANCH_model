"""ENGINEERING_PLAN §3.6 -- Monte-Carlo hygiene: the winner's curse (failure #7) and
seeded reproducibility."""
import numpy as np
import pytest

from granch_fast import metrics as M


def test_grid_selection_without_reevaluation_is_optimistic(human_cm):
    """Null model: condition means are pure noise (8 rollouts). Over 1,000 grid cells the
    best R2 is large; re-evaluating the top cells on 32 fresh rollouts returns ~0."""
    y = 0.5 * (human_cm.LT_odd + human_cm.LT_even).values
    rng = np.random.default_rng(0)
    C, R, K = 1000, 8, len(y)

    def r2(x):
        return np.corrcoef(x, y)[0, 1] ** 2

    grid = np.array([r2(rng.normal(size=(R, K)).mean(0)) for _ in range(C)])
    top = np.argsort(grid)[-20:]
    reeval = np.array([r2(rng.normal(size=(32, K)).mean(0)) for _ in top])
    assert grid.max() > 0.3
    assert grid[top].mean() > 0.3
    assert reeval.mean() < 0.15


def test_seeded_rollouts_are_reproducible(ref_inferred, stim_pair):
    cfg, grid = ref_inferred
    fam, dev = stim_pair
    a = M.infant_trajectories(cfg, grid, fam, dev, 3, T_max=6, rng=np.random.default_rng(11), sigma_true=0.1, want=("mi", "kl"))
    b = M.infant_trajectories(cfg, grid, fam, dev, 3, T_max=6, rng=np.random.default_rng(11), sigma_true=0.1, want=("mi", "kl"))
    c = M.infant_trajectories(cfg, grid, fam, dev, 3, T_max=6, rng=np.random.default_rng(12), sigma_true=0.1, want=("mi", "kl"))
    for m in ("mi", "kl"):
        assert np.array_equal(a[m], b[m])
        assert not np.array_equal(a[m], c[m])
