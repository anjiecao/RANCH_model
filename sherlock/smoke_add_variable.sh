#!/bin/bash
# Smoke of the `add-variable` stage of pipeline.sbatch (one decision variable computed alone into the record's tables,
# seeded as a full chain) at toy size, on a SCRATCH copy of the record's tables -- never the record itself:
#   RANCH_ROOT=/path/to/scratch VARIABLE=kl_concept PY=python PROCS=8 bash sherlock/smoke_add_variable.sh
# Checks the code paths (grid/score --metrics into a suffixed npz and merged scores, winners --only with merged shortlist
# and winners tables, adults --metrics merged predictions, phase2 --metrics merged results) and that every other
# variable's rows survive unchanged.
set -euo pipefail
: "${RANCH_ROOT:?set RANCH_ROOT to a scratch copy of the tree}"
V=${VARIABLE:-kl_concept}; PY=${PY:-python}; P=${PROCS:-8}
export RANCH_ROOT OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $RANCH_ROOT/RANCH_model
R() { $PY -m ranch "$@" --procs $P; }
step() { echo "=== $(date '+%F %T')  $*"; }
T=granch_fast/phase1
before=$(mktemp -d); cp $T/infant_scores_selfcons_base.csv $T/infant_scores_selfcons_ext.csv $T/infant_winners.csv $T/adult_preds_selfcons_ext.csv \
  $T/adult_shortlist.csv $T/adult_winners.csv $T/phase2_selfcons_results.csv $before/

step check inputs;            R check
step grid + score $V;         R grid --kind selfcons_base --rollouts 1 --every 24 --metrics $V
                              R score --kind selfcons_base --metrics $V
                              R grid --kind selfcons_ext --rollouts 1 --every 24 --metrics $V
                              R score --kind selfcons_ext --metrics $V
step winners infants $V;      R winners --population infants --kind selfcons --smoke --every 24 --only $V
step adults $V;               R adults --kind adult_ext --mode stochastic --pairs 1 --rollouts 1 --limit 3 --metrics $V
                              R score-adults --which selfcons_ext
step winners adults $V;       R winners --population adults --which selfcons_ext --smoke --pairs 1 --shortlist 1 --only $V
step phase2 $V;               R phase2 --kind selfcons --which selfcons_ext --rules paper --smoke --adult-cells winners --metrics $V
step other variables intact
$PY - "$before" "$T" "$V" <<'EOF'
import sys
import pandas as pd
before, T, V = sys.argv[1:]
for f in ("infant_scores_selfcons_base.csv", "infant_scores_selfcons_ext.csv", "infant_winners.csv", "adult_preds_selfcons_ext.csv",
          "adult_shortlist.csv", "adult_winners.csv", "phase2_selfcons_results.csv"):
    old, new = pd.read_csv(f"{before}/{f}"), pd.read_csv(f"{T}/{f}")
    keep_old, keep_new = old[old.metric != V].reset_index(drop=True), new[new.metric != V].reset_index(drop=True)
    sort = [c for c in ("setting", "metric", "world_EIGs", "rule") if c in keep_old]
    pd.testing.assert_frame_equal(keep_new.sort_values(sort).reset_index(drop=True), keep_old.sort_values(sort).reset_index(drop=True),
                                  check_exact=True)
    print(f"  {f:34s} other variables unchanged ({len(keep_old)} rows); {V}: {int((new.metric == V).sum())} rows")
EOF
step done
