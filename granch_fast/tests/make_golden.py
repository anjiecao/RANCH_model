"""Write tests/golden.json from the current Phase-1/2 outputs (ENGINEERING_PLAN §3.7).

Run deliberately, after a *reviewed* regeneration of the tables; test_golden.py compares
the cached outputs (and one live reference trajectory) against these pinned numbers.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RANCH = os.environ.get("RANCH_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
GF = f"{RANCH}/RANCH_model/granch_fast"
P1 = f"{GF}/phase1"
METRICS = ["eig_code", "eig_within", "kl", "mi", "surprisal_b"]


def main():
    g = {"_note": "pinned from the reproduction command's outputs of 2026-09-17 (job 44027063: infants, deterministic kinds) "
                  "and the adult regeneration of 2026-09-18 (job 44122930): eq-15 fix; exemplar-mean window under noise; "
                  "concept EIG; clamped linkings; infants re-evaluated at R32; adults on 120 log eps nodes, selected from a "
                  "re-evaluated shortlist and reported on independent seeds over every stimulus pair (1180 x 4 rollouts in "
                  "Exp 1, 845 x 12 in Exp 2; plan §0 #14, settings.QUADRATURE, pipeline.PAIRS)"}
    sc = pd.read_csv(f"{P1}/infant_scores_main.csv")
    g["infant_main_best"] = {m: dict(r2=float(sc[sc.metric == m].pooled_r2.max()),
                                     rmse=float(sc[sc.metric == m].pooled_rmse.min())) for m in METRICS}
    inf = pd.read_csv(f"{P1}/infant_scores_infeps.csv")
    g["infant_infeps_best"] = {m: dict(r2=float(inf[inf.metric == m].pooled_r2.max()),
                                       rmse=float(inf[inf.metric == m].pooled_rmse.min())) for m in METRICS}
    a21 = pd.read_csv(f"{P1}/adult_scores21_main.csv")
    g["adult21_main_best_r2"] = {m: float(a21[a21.metric == m].r2_21.max()) for m in METRICS}
    p2 = pd.read_csv(f"{P1}/phase2_results.csv")
    g["phase2_paper_rule"] = {r.metric: dict(inf=float(r.inf_exp2_r2), adu=float(r.adu_exp2_r2),
                                             inf_order=r.inf_order, adu_order=r.adu_order)
                              for r in p2[p2.rule == "paper"].itertuples()}
    ps = pd.read_csv(f"{P1}/phase2_selfcons_results.csv")
    g["phase2_selfcons_paper_rule"] = {r.metric: dict(inf=float(r.inf_exp2_r2), adu=float(r.adu_exp2_r2),
                                                      inf_order=r.inf_order, adu_order=r.adu_order)
                                       for r in ps[ps.rule == "paper"].itertuples()}
    cm = pd.read_csv(f"{GF}/config_map.csv")
    g["config_map"] = {r.config: [float(r.hab), float(r.dis)] for r in cm.itertuples()}
    iw = pd.read_csv(f"{P1}/infant_winners.csv")
    g["infant_winners_R32"] = {r.metric: dict(r2=float(r.r2), grid_r2=float(r.grid_r2), hab=float(r.hab), dis=float(r.dis))
                               for r in iw[iw.rule == "r2"].itertuples()}
    w64 = pd.read_csv(f"{P1}/adult_winners_R64.csv")                     # the drivers' Stage B (R = 64, grid argmax): legacy pin
    g["adult_winners_R64_r2rule"] = {r.metric: float(r.r2_21_R64) for r in w64[w64.rule == "r2"].itertuples()}
    aw = pd.read_csv(f"{P1}/adult_winners.csv")
    g["adult_winners"] = {f"{r.metric}|{r.rule}": dict(r2=float(r.r2_21_reeval), r2_mc_lo=float(r.r2_mc_lo), r2_mc_hi=float(r.r2_mc_hi),
                                                     grid_r2=float(r.r2_21_grid), hab=float(r.hab), dis=float(r.dis), setting=int(r.setting),
                                                     w=float(r.world_EIGs), rollouts=int(r.rollouts), pairs=int(r.pairs))
                          for r in aw.itertuples()}
    pf = pd.read_csv(f"{GF}/paper_panels_concept_fits.csv")
    g["paper_panels_concept"] = {r.figure: dict(r2=float(r.r2), setting=r.setting) for r in pf.itertuples()}
    ch = pd.read_csv(f"{GF}/channels_decomp.csv")
    ratios = {}
    for w in ch.world.unique():
        ratios[w] = {}
        for c in ch.channel.unique():
            s = ch[(ch.world == w) & (ch.channel == c)].set_index(["test_type", "t"]).y
            ratios[w][c] = [float(s[("deviant", t)] / s[("background", t)]) for t in (1, 5)]
    g["channel_ratios_t1_t5"] = ratios
    json.dump(g, open(f"{HERE}/golden.json", "w"), indent=1)
    print("wrote tests/golden.json")


if __name__ == "__main__":
    main()
