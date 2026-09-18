#!/bin/bash
# End-to-end smoke of the whole `python -m ranch` chain (the reproduction command, sherlock/
# pipeline.sbatch) at reduced size: one trial row per condition (--every 24; rows are ordered by
# trial number), 1 rollout, 1 adult pair, 4 adult jobs, --smoke for the re-evaluations and Phase 2.
# Every step is the real stage with the real arguments plus size knobs. Run against a SCRATCH copy
# of the tree (RANCH_model with the current tables; the deterministic main grid npz linked in),
# never the real one -- the outputs are tiny and overwrite the real file names:
#   RANCH_ROOT=/path/to/scratch PY=/path/to/python PROCS=6 bash sherlock/smoke_pipeline.sh
set -euo pipefail
: "${RANCH_ROOT:?set RANCH_ROOT to a scratch copy of the tree}"
PY=${PY:-python}; P=${PROCS:-6}
export RANCH_ROOT OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $RANCH_ROOT/RANCH_model
R() { $PY -m ranch "$@" --procs $P; }
step() { echo "=== $(date '+%F %T')  $*"; }

step check inputs;            R check
step grid selfcons base;      R grid --kind selfcons_base --rollouts 1 --every 24
step grid selfcons ext;       R grid --kind selfcons_ext --rollouts 1 --every 24
step score selfcons;          R score --kind selfcons_base
                              R score --kind selfcons_ext
step winners infants;         R winners --population infants --kind selfcons --smoke --every 24
step adults base;             R adults --kind adult_base --mode stochastic --pairs 1 --rollouts 1 --limit 4
step adults ext;              R adults --kind adult_ext --mode stochastic --pairs 2 --rollouts 4 --limit 11 --metrics mi_concept   # setting 0 = the fitted adult cell's
step score adults;            R score-adults --which selfcons
                              R score-adults --which selfcons_ext
step winners adults;          R winners --population adults --which selfcons_ext --pairs 2 --rollouts 8 --shortlist 2 --metrics mi_concept
step phase2 selfcons;         R phase2 --kind selfcons --smoke --adult-cells winners
step adults nu;               R adults --kind adult_nu --mode stochastic --pairs 1 --rollouts 1 --limit 4
                              R score-adults --which selfcons_nu
step winners adults nu;       R winners --population adults --which selfcons_nu --smoke --pairs 1 --shortlist 2 --metrics eig_code
step phase2 selfcons nu;      R phase2 --kind selfcons --which selfcons_nu --smoke --adult-cells winners --metrics eig_code
step adults beta;             R adults --kind adult_beta --mode stochastic --pairs 1 --rollouts 1 --limit 4 --metrics mi_concept
                              R score-adults --which selfcons_beta
step winners adults beta;     R winners --population adults --which selfcons_beta --smoke --pairs 1 --shortlist 2 --metrics mi_concept
step phase2 selfcons beta;    R phase2 --kind selfcons --which selfcons_beta --smoke --adult-cells winners --metrics mi_concept
step phase2 main;             R phase2 --kind main --rules paper,within,joint --metrics eig_code,mi
step lesion infants;          R grid --kind lesion_infants --rollouts 1 --every 24
                              R score --kind lesion_infants
step lesion adults;           R adults --kind lesion_adults --mode stochastic --pairs 1 --rollouts 1 --limit 2
                              R score-adults --which lesion
step figures;                 R figures --smoke
step done
