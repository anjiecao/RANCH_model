#!/bin/bash
#SBATCH -p owners
#SBATCH --requeue
#SBATCH --job-name=ranch-win1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
# Window pass, part 1: everything that needs the exemplar-mean centering but NOT the
# regeneration's outputs, run in parallel with regen_pipeline on the owners partition
# from a SEPARATE clone ($HOME/ranch_window) so the two jobs never touch the same files.
# Preemptible (--requeue): every step is idempotent, a requeue just restarts the chain.
set -euo pipefail
module load python/3.12.1
export RANCH_ROOT=$HOME/ranch_window
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source $HOME/ranch/venv/bin/activate
cd $RANCH_ROOT/RANCH_model
P=${SLURM_CPUS_PER_TASK:-24}
PY=python
step() { echo "=== $(date '+%F %T')  $*"; }
git log --oneline -1

step selfcons base exemplar_mean;   $PY granch_fast/phase1_selfconsistent.py --which base --procs $P
step selfcons ext exemplar_mean;    $PY granch_fast/phase1_selfconsistent.py --which ext --procs $P
step score selfcons;                $PY granch_fast/score_phase1_selfcons.py --procs $P
step selfcons winners R32;          $PY granch_fast/phase1c_selfcons_winners.py --procs $P
step adults base eig_code rows;     $PY granch_fast/phase1b_adults_selfcons.py --which base --metrics eig_code --procs $P
step adults ext eig_code rows;      $PY granch_fast/phase1b_adults_selfcons.py --which ext --metrics eig_code --procs $P
step channels;                      $PY granch_fast/gen_figs_channels.py
step done
