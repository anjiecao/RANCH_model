"""Paradigm runners returning typed results. forced_exposure_then_test = the infant design
(fresh exemplar per presentation, forced glimpses, then a test trial); self_paced = the adult
design (Luce stopping, deviant probes after D familiar trials), mean-field or stochastic."""
from dataclasses import dataclass, field

import numpy as np

from granch_fast import metrics as M
from .decision import ALL_VARIABLES, RealizedGain
from .learner import Learner


def _check_oracle(variables, world, allow_oracle):
    for v in variables:
        if isinstance(v, RealizedGain) and v.window == "oracle" and not world.noiseless and not allow_oracle:
            raise ValueError("RealizedGain(window='oracle') under a noisy world leaks the true stimulus; "
                             "pass allow_oracle=True only to reproduce the published code")


@dataclass
class Result:
    trajectories: dict                 # engine key -> array (rollouts, T)
    config: dict
    seeds: list = field(default_factory=list)

    def expected_samples(self, variable, w, offset=0.0, max_obs=500):
        """Mean over rollouts of the survival-product expectation E[samples | w]."""
        tr = self.trajectories[variable.key]
        return float(np.mean([M.expected_samples(t + offset, w, max_obs) for t in tr]))


def forced_exposure_then_test(model, world, fam, test, exposures, samples_per_exposure=5, T_max=40,
                              variables=ALL_VARIABLES, rollouts=1, allow_oracle=False):
    _check_oracle(variables, world, allow_oracle)
    fam = np.asarray(fam, float); test = np.asarray(test, float)
    keys = [v.key for v in variables]
    out = {k: np.empty((rollouts, T_max)) for k in keys}
    for r in range(rollouts):
        L = Learner(model, exposures + 1, variables, forced_exposure_max=samples_per_exposure)
        for k in range(exposures):
            for _ in range(samples_per_exposure):
                L.expose(k, world.glimpse(fam))
        L.refresh()
        trajs = {k: out[k][r] for k in keys}
        for t in range(T_max):
            o = L.observe(exposures, world.glimpse(test), truth=test)
            for k in keys:
                trajs[k][t] = o[k]
            if world.noiseless and M._plateaued(trajs, t):        # deterministic: plateau-fill as the engine does
                for k in keys:
                    trajs[k][t + 1:] = trajs[k][t]
                break
    return Result(out, dict(model=model, world=dict(sigma_true=world.sigma_true), exposures=exposures,
                            samples_per_exposure=samples_per_exposure, T_max=T_max,
                            variables=[getattr(v, "label", v.key) for v in variables], rollouts=rollouts))


@dataclass(frozen=True)
class LucePolicy:
    """p_away = w / (value + offset + w) after each glimpse of the driving variable
    (`offset` = the shifted-surprisal constant; 0 for every other variable)."""
    w: float
    variable: object
    offset: float = 0.0

    def stop(self, value, rng):
        p = min(max(self.w / (value + self.offset + self.w), 0.0), 1.0)
        return rng.random() < p


def self_paced(model, world, fam, dev, policy, max_D=10, mode="stochastic", rollouts=1, T_cap=80,
               variables=None, allow_oracle=False, probe_at=None, T_max=60):
    """Familiar trials 1..max_D+1 under Luce stopping, with a deviant probe (on a scratch
    slot, then reset) after each of the familiar-trial counts D in `probe_at`
    (default: every D = 1..max_D; Exp-2 blocks use probe_at=(1, 3, 5)).
    mode='mean_field' propagates E[samples] deterministically (noiseless worlds only;
    T_max = trajectory length before the geometric tail); mode='stochastic' rolls out
    realized sample counts (draw order matches phase1b_adults_selfcons.rollout and
    run_phase2_selfcons.adult_exp2_mc: glimpse noise, then the Luce coin, from world.rng).
    Result.trajectories['dev'] is ordered by the probe D's."""
    variables = tuple(variables) if variables is not None else (policy.variable,)
    if policy.variable not in variables:
        variables = variables + (policy.variable,)
    _check_oracle(variables, world, allow_oracle)
    probes = sorted(range(1, max_D + 1) if probe_at is None else probe_at)
    fam = np.asarray(fam, float); dev = np.asarray(dev, float)
    info = dict(model=model, mode=mode, w=policy.w, offset=policy.offset, probe_at=probes,
                variable=getattr(policy.variable, "label", policy.variable.key), world=dict(sigma_true=world.sigma_true))
    if mode == "mean_field":
        L = Learner(model, max_D + 2, variables, max_observation=T_cap)
        bg, dv = M.adult_curves(L.cfg, L.grid, fam, dev, policy.variable.key, policy.w, max_D=max_D, T_max=T_max,
                                sigma_true=world.sigma_true, offset=policy.offset)
        return Result({"bg": np.asarray(bg)[None, :], "dev": np.array([dv[D] for D in probes])[None, :]}, info)
    if mode != "stochastic":
        raise ValueError(mode)
    bgs, dvs = [], []
    scratch = max_D + 1
    for _ in range(rollouts):
        L = Learner(model, max_D + 2, variables, max_observation=T_cap)

        def trial(slot, stim, commit):
            t = 0
            while True:
                t += 1
                val = L.observe(slot, world.glimpse(stim), truth=stim)[policy.variable.key]
                if policy.stop(val, world.rng) or t >= T_cap:
                    break
            if not commit:
                L.reset_slot(slot)
            return t

        bg, dv = [], []
        for k in range(max_D + 1):
            bg.append(trial(k, fam, True))
            if k + 1 in probes:
                dv.append(trial(scratch, dev, False))
        bgs.append(bg); dvs.append(dv)
    return Result({"bg": np.array(bgs, float), "dev": np.array(dvs, float)}, dict(info, rollouts=rollouts, T_cap=T_cap))
