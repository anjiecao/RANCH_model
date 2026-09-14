"""Write tests/golden.json from the current Phase-1/2 outputs (ENGINEERING_PLAN §3.7).

Run deliberately, after a *reviewed* regeneration of the tables; test_golden.py compares
the cached outputs (and one live reference trajectory) against these pinned numbers.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RANCH = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch")
GF = f"{RANCH}/RANCH_model/granch_fast"
P1 = f"{GF}/phase1"
METRICS = ["eig_code", "eig_within", "kl", "mi", "surprisal_b"]


def main():
    g = {"_note": "pinned from outputs of 2026-09-14 (clamped linkings; selfcons R32 re-evaluation)"}
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
    g["selfcons_winners_R32"] = {"eig_code": 0.035, "kl": 0.231, "mi": 0.729, "surprisal_b": 0.579,
                                 "_source": "phase1/stageB_R32.log, 2026-09-14"}
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
