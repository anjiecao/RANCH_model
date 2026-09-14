"""A learner: a Model instantiated as belief state (granch_fast.metrics.State) plus the
registered decision variables it computes on every glimpse."""
import numpy as np

from granch_fast import metrics as M
from .decision import ALL_VARIABLES, RealizedGain


class Learner:
    def __init__(self, model, n_slots, variables=ALL_VARIABLES, forced_exposure_max=5, max_observation=500):
        self.model = model
        self.variables = tuple(variables)
        rgs = [v for v in self.variables if isinstance(v, RealizedGain)]
        if len(rgs) > 1:
            raise ValueError("one RealizedGain specification per learner")
        self.realized_gain = rgs[0] if rgs else None
        hw, n_z = (self.realized_gain.half_width, self.realized_gain.n_z) if rgs else (1e-4, 1)
        self.cfg = model.fast_config(window_half_width=hw, n_z=n_z, forced_exposure_max=forced_exposure_max,
                                     max_observation=max_observation)
        self.grid = model.grid(self.cfg)
        self.state = M.State(self.cfg, self.grid, n_slots)
        self.want = tuple(v.key for v in self.variables)
        self.last = None

    # ---- input -------------------------------------------------------------
    def expose(self, slot, z):
        """Forced exposure: record a glimpse of exemplar `slot` without a decision."""
        z = np.asarray(z, dtype=float)
        for d in range(self.cfg.n_feature):
            self.state.add_sample(d, slot, z[d])

    def refresh(self):
        self.state.ensure_init()
        for d in range(self.cfg.n_feature):
            self.state._refresh(d)

    def observe(self, slot, z, truth=None):
        """Observe glimpse z of exemplar `slot`; return {engine key: value} for every
        registered variable (the decision about the NEXT glimpse). `truth` is used only by
        RealizedGain(window='oracle')."""
        z = np.asarray(z, dtype=float)
        rg = self.realized_gain
        window = rg.window if rg is not None else "observed"
        if window == "oracle" and truth is None:
            raise ValueError("RealizedGain(window='oracle') needs the true stimulus")
        center = M.window_center(self.state, slot, z, truth, window)
        self.last = self.state.step(slot, z, center, want=self.want)
        return self.last

    def reset_slot(self, slot):
        self.state.reset_stim(slot)

    # ---- belief state ------------------------------------------------------
    @property
    def posterior(self):
        """Per feature: (node weights over (sigma^2, eps), m_mu, v_mu)."""
        self.state.ensure_init()
        return [(fp.post.copy(), fp.m_mu.copy(), fp.v_mu.copy()) for fp in self.state.fps]
