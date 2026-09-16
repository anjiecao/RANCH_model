#!/bin/bash
#SBATCH -p mcfrank
#SBATCH --job-name=ranch-finish
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
# The last four stages of the 2026-09-14/15 regeneration (resume_all died at the R=64
# winners step on a NameError in phase1e, fixed). Everything upstream is complete in ~/ranch.
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
step() { echo "=== $(date '+%F %T')  $*"; }

step adults winners R64;   $PY granch_fast/phase1e_adults_winners.py --procs 24
step phase2 selfcons;      $PY granch_fast/run_phase2_selfcons.py --procs 24 --adults ext
step lesions;              $PY granch_fast/phase1d_lesions_selfcons.py --procs 24
step channels;             $PY granch_fast/gen_figs_channels.py
step done
