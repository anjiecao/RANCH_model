"""The infant results under the protocol of 2026-09-24 (200 simulated samples per test trial, no cap, no saturation
filter: pipeline.INFANT_T_MAX / INFANT_CAP) next to the record's (40 samples, cap 500). Reads the record's tables from git
(HEAD) and the regenerated ones from a directory (the side run's granch_fast/phase1). Light: seconds.
usage: infant_protocol_compare.py NEW_PHASE1_DIR"""
import io
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NEW = sys.argv[1]


def record(fn):
    txt = subprocess.run(["git", "show", f"HEAD:granch_fast/phase1/{fn}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return pd.read_csv(io.StringIO(txt))


def label(r):
    return f"V{r.V_prior:g} a{r.alpha_prior:g} b{r.beta_prior:g} sd{r.sd_epsilon:g} st{r.sigma_true:g} w{r.world_EIGs:.2g}"


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for fn, what in (("infant_winners.csv", "EXPERIMENT 1 WINNERS"), ("infant_winners_lesion_infants.csv", "LESION WINNERS")):
        old, new = record(fn), pd.read_csv(f"{NEW}/{fn}")
        rows = []
        for tag, d in (("record", old), ("new", new)):
            for _, r in d.iterrows():
                rows.append(dict(variable=r.metric, rule=r.rule, protocol=tag, cell=label(r), r2=r.r2, r2_group_sd=r.r2_group_sd, rmse=r.rmse,
                                 hab=r.hab, dis=r.dis, looking_past_simulation=r.get("looking_past_simulation", float("nan")),
                                 first_presentation=r.get("bg_1", float("nan"))))
        print(f"{what}\n" + pd.DataFrame(rows).sort_values(["variable", "rule", "protocol"], ascending=[True, True, False])
              .to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n")
    old, new = record("phase2_selfcons_results.csv"), pd.read_csv(f"{NEW}/phase2_selfcons_results.csv")
    cols = ["metric", "rule", "inf_setting", "inf_exp1_r2", "inf_exp2_r2", "inf_exp2_r2_mc_lo", "inf_exp2_r2_mc_hi", "inf_exp2_rmse", "inf_order"]
    both = pd.concat([old[cols].assign(protocol="record"), new[cols].assign(protocol="new")]).sort_values(["metric", "rule", "protocol"], ascending=[True, True, False])
    print("EXPERIMENT 2, INFANTS\n" + both.to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n")
    adu = ["metric", "rule", "adu_exp2_r2", "adu_order"]
    j = old[adu].merge(new[adu], on=["metric", "rule"], suffixes=("_record", "_new"))
    print("EXPERIMENT 2, ADULTS (unchanged code and seeds: must be identical)\n" + j.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
