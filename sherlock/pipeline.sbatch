#!/bin/bash
#SBATCH -p mcfrank
#SBATCH --job-name=ranch-pipeline
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out
# THE reproduction command: every table and figure datum of the revision from `python -m ranch`
# (ENGINEERING_PLAN §4, Phase B step 3). Stages and their legacy equivalents are documented in
# ranch/pipeline.py; the chain is smoke-tested end to end by sherlock/smoke_pipeline.sh.
# Rough timings on the 24-core owner node: infant grids main 4 min, infeps 10 min, selfcons
# base 42 min, ext 2.7 h, lesion 5 min; adults main 20 min, base 1.7 h, ext 5 h, lesion 15 min;
# winners 35 min + 5 min; phase 2 20 min; figures 10 min -- about 12 h in all.
# Set STAGES to a subset (space-separated) to rerun part of the chain.
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
R() { python -m ranch "$@" --procs 24; }
step() { echo "=== $(date '+%F %T')  $*"; }
STAGES=${STAGES:-"infants adults winners phase2 lesion figures"}
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has infants; then
  step grid main;             R grid --kind main
  step grid infeps;           R grid --kind infeps
  step grid selfcons base;    R grid --kind selfcons_base
  step grid selfcons ext;     R grid --kind selfcons_ext
  step score infants;         R score --kind main
                              R score --kind infeps
                              R score --kind selfcons_base
                              R score --kind selfcons_ext
fi
if has adults; then
  step adults main;           R adults --kind main --mode mean_field --pairs 6
                              R score-adults --which main
  step adults base;           R adults --kind adult_base --mode stochastic --pairs 6 --rollouts 16
                              R score-adults --which selfcons
  step adults ext;            R adults --kind adult_ext --mode stochastic --pairs 6 --rollouts 16
                              R score-adults --which selfcons_ext
fi
if has winners; then
  step winners infants R32;   R winners --population infants --kind selfcons --rollouts 32
  step winners adults R64;    R winners --population adults --which selfcons_ext --rollouts 64 --pairs 6
fi
if has phase2; then
  step phase2 main;           R phase2 --kind main --rules paper,within,joint
  step phase2 selfcons;       R phase2 --kind selfcons --rules paper,r2
fi
if has lesion; then
  step lesion infants;        R grid --kind lesion_infants --rollouts 16
                              R score --kind lesion_infants
  step lesion adults;         R adults --kind lesion_adults --mode stochastic --pairs 6 --rollouts 16
                              R score-adults --which lesion
fi
if has figures; then
  step figures;               R figures
fi
step done
