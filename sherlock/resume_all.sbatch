#!/bin/bash
#SBATCH -p mcfrank
#SBATCH --job-name=ranch-resume
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out
# Resume after the 2026-09-14 regeneration died at its first scoring step (hardcoded
# scorer paths, fixed in commit after 3c4b8d0). Reuses every grid that completed:
#   ~/ranch        : infant_traj_main / infeps (fixed eq-15, deterministic)      <- regen
#   ~/ranch_window : infant_traj_selfcons / _ext with exemplar-mean centering    <- win1 (if complete)
# and runs everything downstream with the exemplar-mean window throughout.
# Submit with --dependency=afterany:<win1 jobid>.
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

step keep the regen's oracle-window selfcons grids for the record
for f in infant_traj_selfcons.npz infant_traj_selfcons_ext.npz; do [ -f $A/$f ] && mv $A/$f $A/${f%.npz}_oraclewin.npz; done

step import win1's exemplar-mean selfcons grids if complete, else recompute
if $PY - <<EOF
import numpy as np, sys
try:
    a = np.load("$B/infant_traj_selfcons.npz")["traj"]; b = np.load("$B/infant_traj_selfcons_ext.npz")["traj"]
    ok = a.shape[0] == 16 and b.shape[0] == 80 and np.isfinite(a).all() and np.isfinite(b).all()
except Exception as e:
    ok = False
print("win1 grids complete:", ok); sys.exit(0 if ok else 1)
EOF
then cp $B/infant_traj_selfcons.npz $B/infant_traj_selfcons_ext.npz $B/infant_settings_selfcons*.csv $A/
else
  $PY granch_fast/phase1_selfconsistent.py --which base --procs 24
  $PY granch_fast/phase1_selfconsistent.py --which ext --procs 24
fi

step score infants;              $PY granch_fast/score_phase1_infants.py --which main --procs 24
                                 $PY granch_fast/score_phase1_infants.py --which infeps --procs 24
step score selfcons;             $PY granch_fast/score_phase1_selfcons.py --procs 24
step selfcons winners R32;       $PY granch_fast/phase1c_selfcons_winners.py --procs 24
step adults main mean-field;     $PY granch_fast/phase1_adults.py --which main --procs 24 --pairs 6
step score adults;               $PY granch_fast/score_phase1_adults.py --which main
                                 $PY granch_fast/score_phase1_adults21.py main
step phase2 deterministic;       $PY granch_fast/run_phase2.py
step adults selfcons base;       $PY granch_fast/phase1b_adults_selfcons.py --which base --procs 24
step score adults selfcons base; $PY granch_fast/score_phase1_adults.py --which selfcons
                                 $PY granch_fast/score_phase1_adults21.py selfcons
step adults selfcons ext;        $PY granch_fast/phase1b_adults_selfcons.py --which ext --procs 24
step adults winners R64;         $PY granch_fast/phase1e_adults_winners.py --procs 24
step phase2 selfcons;            $PY granch_fast/run_phase2_selfcons.py --procs 24 --adults ext
step lesions;                    $PY granch_fast/phase1d_lesions_selfcons.py --procs 24
step channels;                   $PY granch_fast/gen_figs_channels.py
step done
