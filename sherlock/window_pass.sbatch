#!/bin/bash
#SBATCH -p mcfrank
#SBATCH --job-name=ranch-window
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out
# Second pass after regen_pipeline: recompute everything that involves the implemented
# functional under a NOISY world with the exemplar-mean window centering (team decision
# 2026-09-14) instead of the published oracle centering. Syncs the code first (the
# bundle is copied to ~/ranch_reboot.bundle before submission), so it must only start
# after the regeneration has finished (submit with --dependency=afterok:<regen jobid>).
set -euo pipefail
module load python/3.12.1
export RANCH_ROOT=$HOME/ranch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source $RANCH_ROOT/venv/bin/activate
cd $RANCH_ROOT/RANCH_model
git fetch -q ~/ranch_reboot.bundle exact-inference-reboot && git reset -q --hard FETCH_HEAD && git log --oneline -1
PY=python
step() { echo "=== $(date '+%F %T')  $*"; }

step selfcons base exemplar_mean;   $PY granch_fast/phase1_selfconsistent.py --which base --procs 24
step selfcons ext exemplar_mean;    $PY granch_fast/phase1_selfconsistent.py --which ext --procs 24
step score selfcons;                 $PY granch_fast/score_phase1_selfcons.py --procs 24
step selfcons winners R32;           $PY granch_fast/phase1c_selfcons_winners.py --procs 24
step adults base eig_code merge;   $PY granch_fast/phase1b_adults_selfcons.py --which base --metrics eig_code --merge --procs 24
step score adults selfcons base;     $PY granch_fast/score_phase1_adults.py --which selfcons
                                     $PY granch_fast/score_phase1_adults21.py selfcons
step adults ext eig_code merge;    $PY granch_fast/phase1b_adults_selfcons.py --which ext --metrics eig_code --merge --procs 24
step adults winners R64;             $PY granch_fast/phase1e_adults_winners.py --procs 24
step phase2 selfcons;                $PY granch_fast/run_phase2_selfcons.py --procs 24 --adults ext
step channels;                       $PY granch_fast/gen_figs_channels.py
step done
