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
# winners 35 min + 25 min (adults at 512 rollouts per pair); phase 2 about 1.5 h (Exp-2 adults at 512 rollouts per
# pair: ranch.pipeline.ROLLOUTS); figures 15 min -- about 14 h in all.
# Set STAGES to a subset (space-separated) to rerun part of the chain; `adults` is all three adult
# sweeps, adults-main / adults-base / adults-ext one of them; likewise winners(-infants|-adults) and
# phase2(-main|-selfcons). Not part of the default chain: `snapshot` (copy every table aside first),
# `adults-nu` (the nu < 1 extension of the noisy adult grid), `adults-beta` (the beta < .1 extension) and `add-variable`
# (VARIABLE=kl_concept: one decision variable computed alone into the record's tables, seeded as a full chain). ADULT_WHICH
# chooses the adult sweep that the adult winners and the noisy Phase 2 work from (selfcons_ext = the record;
# selfcons_nu / selfcons_beta = the extensions), ADULT_METRICS restricts those adult stages to some decision variables
# (e.g. mi_concept). Adult protocol since 2026-09-18: 120 log eps nodes (settings.QUADRATURE) and every stimulus pair of
# the experiment behind each reported number (pipeline.PAIRS); the timings above predate it (about 3.3x per adult stage).
set -euo pipefail
module load python/3.12.1
export RANCH_ROOT=${RANCH_ROOT:-$HOME/ranch}      # another root = a separate clone: own tree, own outputs (side runs never touch the record)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$RANCH_ROOT/pkbb_paper_writing
source ${RANCH_VENV:-$HOME/ranch/venv}/bin/activate
cd $RANCH_ROOT/RANCH_model
# code-only sync from GitHub (never touch the output tables in granch_fast/phase1/); every line aborts the job on failure
git fetch -q origin exact-inference-reboot
git reset -q --soft FETCH_HEAD
git ls-tree -r --name-only HEAD | grep -v "^granch_fast/phase1/" | tr '\n' '\0' | xargs -0 git checkout -q HEAD --   # NUL-separated: four tracked paths contain spaces
STALE=$(git diff --cached --name-only --diff-filter=A HEAD | grep -v "^granch_fast/phase1/" || true)                     # tracked before, gone from HEAD: drop them
if [ -n "$STALE" ]; then printf '%s\n' "$STALE" | tr '\n' '\0' | xargs -0 git rm -q --cached --; printf '%s\n' "$STALE" | tr '\n' '\0' | xargs -0 rm -f; fi
if git status --short | grep -v "^?? " | grep -v "granch_fast/phase1/" | grep -q .; then echo "SYNC FAILED: tracked code differs from HEAD"; git status --short | grep -v "^?? " | head; exit 1; fi
git log --oneline -1
R() { python -m ranch "$@" --procs ${SLURM_CPUS_PER_TASK:-24}; }
step() { echo "=== $(date '+%F %T')  $*"; }
STAGES=${STAGES:-"infants adults winners phase2 lesion figures"}
ADULT_WHICH=${ADULT_WHICH:-selfcons_ext}
MET=${ADULT_METRICS:+--metrics $ADULT_METRICS}
has() { [[ " $STAGES " == *" $1 "* ]]; }

step check inputs;            R check      # a missing or changed data file stops the job here, not at the stage that reads it

if has snapshot; then
  SNAP=$RANCH_ROOT/snapshots/${SLURM_JOB_ID:-manual}_before
  step snapshot to $SNAP;     mkdir -p $SNAP/phase1 && cp -p granch_fast/phase1/*.csv $SNAP/phase1/ && cp -p granch_fast/*.csv $SNAP/
fi
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
if has adults || has adults-main; then
  step adults main;           R adults --kind main --mode mean_field --pairs 6
                              R score-adults --which main
fi
if has adults || has adults-base; then       # noisy sweeps: 96 pairs x 1 rollout per cell (pipeline.PAIRS / ROLLOUTS)
  step adults base;           R adults --kind adult_base --mode stochastic --pairs 96 --rollouts 1
                              R score-adults --which selfcons
fi
if has adults || has adults-ext; then
  step adults ext;            R adults --kind adult_ext --mode stochastic --pairs 96 --rollouts 1
                              R score-adults --which selfcons_ext
fi
if has adults-nu; then
  step adults nu;             R adults --kind adult_nu --mode stochastic --pairs 96 --rollouts 1 $MET
                              R score-adults --which selfcons_nu
fi
if has adults-beta; then
  step adults beta;           R adults --kind adult_beta --mode stochastic --pairs 96 --rollouts 1 $MET
                              R score-adults --which selfcons_beta
fi
if has winners || has winners-infants; then
  step winners infants R32;   R winners --population infants --kind selfcons --rollouts 32
fi
if has winners || has winners-adults; then      # the 10 best grid cells per variable and rule re-evaluated (every pair, 4 rollouts), selected among, reported on fresh seeds
  step winners adults;        R winners --population adults --which $ADULT_WHICH --pairs all --shortlist 10 $MET
fi
if has phase2 || has phase2-main; then
  step phase2 main;           R phase2 --kind main --rules paper,within,joint
fi
if has phase2 || has phase2-selfcons; then
  step phase2 selfcons;       R phase2 --kind selfcons --which $ADULT_WHICH --rules paper,r2 --adult-cells winners $MET
fi
if has add-variable; then      # a decision variable added to the record (VARIABLE=...): computed alone, into the record's
  V=${VARIABLE:?set VARIABLE}   # tables, with the seeds a full chain would give it (grid/adults --metrics, winners --only)
  step grid selfcons base $V; R grid --kind selfcons_base --metrics $V
                              R score --kind selfcons_base --metrics $V
  step grid selfcons ext $V;  R grid --kind selfcons_ext --metrics $V
                              R score --kind selfcons_ext --metrics $V
  step winners infants $V;    R winners --population infants --kind selfcons --rollouts 32 --only $V
  step adults ext $V;         R adults --kind adult_ext --mode stochastic --pairs 96 --rollouts 1 --metrics $V
                              R score-adults --which selfcons_ext
  step winners adults $V;     R winners --population adults --which selfcons_ext --pairs all --shortlist 10 --only $V
  step phase2 selfcons $V;    R phase2 --kind selfcons --which selfcons_ext --rules paper,r2 --adult-cells winners --metrics $V
fi
if has lesion; then           # the no-noise-learner lesion under the adult protocol of 2026-09-18, re-evaluated like the record
  step lesion infants;        R grid --kind lesion_infants --rollouts 16
                              R score --kind lesion_infants
                              R winners --population infants --kind lesion_infants --rollouts 32
  step lesion adults;         R adults --kind lesion_adults --mode stochastic --pairs 96 --rollouts 1
                              R score-adults --which lesion
                              R winners --population adults --which lesion --pairs all --shortlist 11
fi
if has figures; then
  step figures;               R figures
fi
step done
