#!/bin/bash
#SBATCH -p mcfrank
#SBATCH --job-name=ranch-win2
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
# Window pass, part 2 (submit with --dependency=afterok:<regen>:<part1>): bring part 1's
# outputs into the main tree, merge the exemplar-mean implemented-EIG rows into the
# regenerated adult sweeps, then the stages that need both.
set -euo pipefail
module load python/3.12.1
export RANCH_ROOT=$HOME/ranch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source $RANCH_ROOT/venv/bin/activate
cd $RANCH_ROOT/RANCH_model
# code-only sync from GitHub (never touch the output tables in granch_fast/phase1/); every line aborts the job on failure
git fetch -q origin exact-inference-reboot
git reset -q --soft FETCH_HEAD
git ls-tree -r --name-only HEAD | grep -v "^granch_fast/phase1/" | tr '\n' '\0' | xargs -0 git checkout -q HEAD --   # NUL-separated: four tracked paths contain spaces
if git status --short | grep -v "^?? " | grep -v "granch_fast/phase1/" | grep -q .; then echo "SYNC FAILED: tracked code differs from HEAD"; git status --short | grep -v "^?? " | head; exit 1; fi
git log --oneline -1
PY=python
A=granch_fast/phase1
B=$HOME/ranch_window/RANCH_model/granch_fast/phase1
step() { echo "=== $(date '+%F %T')  $*"; }

step keep the oracle-window versions for the record
for f in infant_traj_selfcons.npz infant_traj_selfcons_ext.npz infant_scores_selfcons.csv selfcons_winners_R32.npz; do
  [ -f $A/$f ] && cp $A/$f $A/${f%.*}_oraclewin.${f##*.}; done
cp $A/adult_preds_selfcons.csv $A/adult_preds_selfcons_oraclewin.csv
cp $A/adult_preds_selfcons_ext.csv $A/adult_preds_selfcons_ext_oraclewin.csv

step import part-1 outputs
cp $B/infant_traj_selfcons.npz $B/infant_traj_selfcons_ext.npz $B/infant_settings_selfcons*.csv \
   $B/infant_scores_selfcons.csv $B/selfcons_winners_R32.npz $A/
cp $HOME/ranch_window/RANCH_model/granch_fast/channels_decomp.csv granch_fast/

step merge exemplar-mean eig_code rows into the regenerated adult sweeps
$PY - <<EOF
import pandas as pd
for suf in ("", "_ext"):
    a = pd.read_csv("$A/adult_preds_selfcons%s.csv" % suf)
    b = pd.read_csv("$B/adult_preds_selfcons%s.csv" % suf)
    out = pd.concat([a[a.metric != "eig_code"], b[b.metric == "eig_code"]], ignore_index=True)
    out.sort_values(["setting", "metric", "world_EIGs"]).to_csv("$A/adult_preds_selfcons%s.csv" % suf, index=False)
    print(suf or "base", len(a), "->", len(out), "rows; eig_code window:", b[b.metric == "eig_code"].window.unique())
EOF

step score adults selfcons base;     $PY granch_fast/score_phase1_adults.py --which selfcons
                                     $PY granch_fast/score_phase1_adults21.py selfcons
step adults winners R64;             $PY granch_fast/phase1e_adults_winners.py --procs 24
step phase2 selfcons;                $PY granch_fast/run_phase2_selfcons.py --procs 24 --adults ext
step done
