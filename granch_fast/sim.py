"""Analytic granch simulation: full RANCH loop with the grid-free inference core.

Mirrors main_sim_tensor.granch_main_simulation (EIG linking hypothesis) and the
decision rule in init_model_tensor.make_decision, but uses the analytic
per-feature posterior + EIG instead of the 4D random grid.
"""
import numpy as np
from .analytic_core import SigmaEpsGrid, FeaturePosterior
from . import eig as EIG


class GranchConfig:
    def __init__(self, mu_prior=0.0, V_prior=1.0, alpha_prior=1.0, beta_prior=1.0,
                 epsilon=1e-4, mu_epsilon=1e-3, sd_epsilon=0.5, world_EIGs=1e-4,
                 max_observation=500, forced_exposure_max=5, n_feature=3,
                 sigma_box=(0.001, 1.8), eps_box=(1e-6, 1.0),
                 n_sigma=120, n_eps=30, n_z=5, spacing="linear", infer_eps=True,
                 eig_scale=1.0):
        self.__dict__.update(locals())
        del self.self


def make_grid(cfg):
    return SigmaEpsGrid(cfg.sigma_box[0], cfg.sigma_box[1], cfg.n_sigma,
                        cfg.eps_box[0], cfg.eps_box[1], cfg.n_eps,
                        spacing=cfg.spacing, infer_eps=cfg.infer_eps)


class AnalyticSim:
    def __init__(self, cfg, grid=None, rng=None):
        self.cfg = cfg
        self.grid = grid if grid is not None else make_grid(cfg)
        self.rng = rng if rng is not None else np.random.default_rng()

    def _new_posteriors(self):
        c = self.cfg
        return [FeaturePosterior(self.grid, c.mu_prior, c.V_prior, c.alpha_prior,
                                 c.beta_prior, c.mu_epsilon, c.sd_epsilon)
                for _ in range(c.n_feature)]

    def eig(self, fps, stats, cur_idx, stim_vals):
        c = self.cfg
        # sum of per-feature E terms (see harness_eig: grid combines features
        # additively up to a constant; that constant folds into eig_scale /
        # world_EIGs, both fit per model).
        total = 0.0
        for d in range(c.n_feature):
            _, E = EIG.feature_eig_terms(fps[d], stats[d], cur_idx,
                                         stim_vals[d], c.epsilon, c.n_z)
            total += E
        return c.eig_scale * total

    def run_sequence(self, stim_seq):
        """stim_seq: list of feature-vectors (np arrays), one per trial (B/D).
        Returns per-trial sample counts; the last entry is the test-trial proxy
        for looking time."""
        c = self.cfg
        fps = self._new_posteriors()
        n_trial = len(stim_seq)
        stats = [[[0.0, 0.0, 0.0] for _ in range(n_trial)] for _ in range(c.n_feature)]
        counts = [0] * n_trial
        forced = c.forced_exposure_max
        use_forced = not (forced is None or (isinstance(forced, float) and np.isnan(forced)))

        t = 0
        k = 0
        cur_t = 0
        while t < c.max_observation and k < n_trial:
            stim = stim_seq[k]
            # observe one noisy sample of the current stimulus (per feature)
            for d in range(c.n_feature):
                z = stim[d] + self.rng.normal(0, c.epsilon)
                n0, zb0, S0 = stats[d][k]
                n1 = n0 + 1
                zb1 = (n0 * zb0 + z) / n1
                stats[d][k] = [n1, zb1, S0 + (z - zb0) * (z - zb1)]
            for d in range(c.n_feature):
                seen = [s for s in stats[d] if s[0] > 0]
                fps[d].update([s[0] for s in seen], [s[1] for s in seen], [s[2] for s in seen])
            counts[k] += 1

            eig = self.eig(fps, stats, k, stim)
            p_look = max(min(c.world_EIGs / (eig + c.world_EIGs), 1.0), 0.0)

            if use_forced and k < n_trial - 1:
                if cur_t < forced - 1:
                    decision_away = False
                else:
                    decision_away = True
            else:
                decision_away = self.rng.binomial(1, p_look) == 1

            if decision_away:
                k += 1
                cur_t = 0
            else:
                cur_t += 1
            t += 1
        return counts
