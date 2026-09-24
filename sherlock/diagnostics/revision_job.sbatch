#!/bin/bash
#SBATCH -p owners
#SBATCH --requeue
#SBATCH --job-name=ranch-revision
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
# One of the revision analyses (sherlock/diagnostics/*.py) in its own clone, RANCH_ROOT=~/ranch_rev/<name>; the clone's
# phase1 tables are the record's (from git) and the record's infant trajectory files are linked in read-only.
#   RANCH_ROOT=~/ranch_rev/wb SCRIPT=w_boundary.py ARGS="granch_fast/phase1 out_w_boundary" sbatch sherlock/diagnostics/revision_job.sbatch
set -euo pipefail
module load python/3.12.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$RANCH_ROOT/pkbb_paper_writing
source $HOME/ranch/venv/bin/activate
cd $RANCH_ROOT/RANCH_model
git log -1 --oneline
echo "=== $(date '+%F %T')  $SCRIPT $ARGS"
python sherlock/diagnostics/$SCRIPT $ARGS ${SLURM_CPUS_PER_TASK:-24}
echo "=== $(date '+%F %T')  done"
