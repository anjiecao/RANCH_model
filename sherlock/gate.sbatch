#!/bin/bash
#SBATCH -p mcfrank
#SBATCH --job-name=ranch-gate
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out
# Phase-B gate (ENGINEERING_PLAN §4): run the grids through the API pipeline into
# phase1_ranch/ and diff against the legacy drivers' outputs in phase1/. Submit with
# --dependency=afterany:<concept study jobid> so the legacy selfcons outputs carry the
# same five variables. The adult ext sweep is skipped (5 h; adult_base proves the path).
set -euo pipefail
module load python/3.12.1
export RANCH_ROOT=$HOME/ranch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$RANCH_ROOT/pkbb_paper_writing
source $RANCH_ROOT/venv/bin/activate
cd $RANCH_ROOT/RANCH_model
# code-only sync from GitHub (never touch the output tables in granch_fast/phase1/); every line aborts the job on failure
git fetch -q origin exact-inference-reboot
git reset -q --soft FETCH_HEAD
git ls-tree -r --name-only HEAD | grep -v "^granch_fast/phase1/" | tr '\n' '\0' | xargs -0 git checkout -q HEAD --   # NUL-separated: four tracked paths contain spaces
if git status --short | grep -v "^?? " | grep -v "granch_fast/phase1/" | grep -q .; then echo "SYNC FAILED: tracked code differs from HEAD"; git status --short | grep -v "^?? " | head; exit 1; fi
git log --oneline -1
OUTR=$RANCH_ROOT/RANCH_model/granch_fast/phase1_ranch
mkdir -p $OUTR
R() { python -m ranch "$@" --out $OUTR --procs 24; }
step() { echo "=== $(date '+%F %T')  $*"; }

step grid main;             R grid --kind main
step grid infeps;           R grid --kind infeps
step grid selfcons base;    R grid --kind selfcons_base
step grid selfcons ext;     R grid --kind selfcons_ext
step score main;            R score --kind main
step score infeps;          R score --kind infeps
step score selfcons base;   R score --kind selfcons_base
step score selfcons ext;    R score --kind selfcons_ext
step adults main;           R adults --kind main --mode mean_field --pairs 6
step score adults main;     R score-adults --which main
step adults base;           R adults --kind adult_base --mode stochastic --pairs 6 --rollouts 16
step score adults base;     R score-adults --which selfcons
step gate;                  python -m ranch.gate --ranch $OUTR --legacy $RANCH_ROOT/RANCH_model/granch_fast/phase1
step done
