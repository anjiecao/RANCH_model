#!/bin/bash
#SBATCH -p mcfrank
#SBATCH --job-name=ranch-concept
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out
# The concept-only EIG (mi_concept, eps a nuisance) through the noisy-world study
# (submit with --dependency=afterany:<finish jobid>). The self-consistent infant grids are
# recomputed with the extra variable (same seeds: the other four variables reproduce
# exactly); the noisy-adult sweeps get mi_concept rows merged in.
set -euo pipefail
module load python/3.12.1
export RANCH_ROOT=$HOME/ranch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source $RANCH_ROOT/venv/bin/activate
cd $RANCH_ROOT/RANCH_model
git fetch -q ~/ranch_reboot.bundle exact-inference-reboot && git reset -q --hard FETCH_HEAD && git log --oneline -1
PY=python
step() { echo "=== $(date '+%F %T')  $*"; }

step selfcons base with mi_concept;   $PY granch_fast/phase1_selfconsistent.py --which base --procs 24
step selfcons ext with mi_concept;    $PY granch_fast/phase1_selfconsistent.py --which ext --procs 24
step score selfcons;                  $PY granch_fast/score_phase1_selfcons.py --procs 24
step selfcons winners R32;            $PY granch_fast/phase1c_selfcons_winners.py --procs 24
step adults base mi_concept merge;    $PY granch_fast/phase1b_adults_selfcons.py --which base --metrics mi_concept --merge --procs 24
step score adults selfcons base;      $PY granch_fast/score_phase1_adults.py --which selfcons
                                      $PY granch_fast/score_phase1_adults21.py selfcons
step adults ext mi_concept merge;     $PY granch_fast/phase1b_adults_selfcons.py --which ext --metrics mi_concept --merge --procs 24
step adults winners R64;              $PY granch_fast/phase1e_adults_winners.py --procs 24
step phase2 selfcons;                 $PY granch_fast/run_phase2_selfcons.py --procs 24 --adults ext
step done
