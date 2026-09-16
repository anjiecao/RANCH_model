#!/bin/bash
# End-to-end smoke of the concept_study chain, and of the gate's CLI, at reduced size: one
# trial row per condition (--every 24; the rows are ordered by trial number), 1 rollout,
# 1 adult pair, 4 adult jobs. Every step is the real script with the real arguments plus size
# knobs, so a code-path failure shows up here in ~15 min instead of hours into a job.
# Run against a SCRATCH copy of the tree (RANCH_model with the current tables), never the
# real one -- the outputs are tiny and overwrite the real file names:
#   RANCH_ROOT=/path/to/scratch PY=/path/to/python PROCS=6 bash sherlock/smoke_concept.sh
set -euo pipefail
: "${RANCH_ROOT:?set RANCH_ROOT to a scratch copy of the tree}"
PY=${PY:-python}; P=${PROCS:-6}
export RANCH_ROOT OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$RANCH_ROOT/pkbb_paper_writing
cd $RANCH_ROOT/RANCH_model
step() { echo "=== $(date '+%F %T')  $*"; }

step selfcons base;        $PY granch_fast/phase1_selfconsistent.py --which base --rollouts 1 --every 24 --procs $P
step selfcons ext;         $PY granch_fast/phase1_selfconsistent.py --which ext --rollouts 1 --every 24 --procs $P
step score selfcons;       $PY granch_fast/score_phase1_selfcons.py --procs $P
step selfcons winners;     $PY granch_fast/phase1c_selfcons_winners.py --procs $P --rollouts 8 --every 24
step adults base merge;    $PY granch_fast/phase1b_adults_selfcons.py --which base --metrics mi_concept --merge --rollouts 1 --pairs 1 --limit 4 --procs $P
step score adults base;    $PY granch_fast/score_phase1_adults.py --which selfcons
                           $PY granch_fast/score_phase1_adults21.py selfcons
step adults ext merge;     $PY granch_fast/phase1b_adults_selfcons.py --which ext --metrics mi_concept --merge --rollouts 1 --pairs 1 --limit 4 --procs $P
step adults winners;       $PY granch_fast/phase1e_adults_winners.py --procs $P --pairs 1 --rollouts 1
step phase2 selfcons;      $PY granch_fast/run_phase2_selfcons.py --procs $P --adults ext --smoke
step channels;             $PY granch_fast/gen_figs_channels.py

OUTR=$RANCH_ROOT/RANCH_model/granch_fast/phase1_ranch; mkdir -p $OUTR
R() { $PY -m ranch "$@" --out $OUTR --procs $P; }
step ranch grid;           R grid --kind selfcons_base --rollouts 1 --every 24
step ranch score;          R score --kind selfcons_base
step ranch adults;         R adults --kind adult_base --mode stochastic --pairs 1 --rollouts 1
step ranch score-adults;   R score-adults --which selfcons
step gate;                 $PY -m ranch.gate --ranch $OUTR --legacy $RANCH_ROOT/RANCH_model/granch_fast/phase1 || true   # adult sizes differ here (mismatch expected); the selfcons grid must be OK
step done
