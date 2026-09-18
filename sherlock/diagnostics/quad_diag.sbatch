#!/bin/bash
#SBATCH -p mcfrank
#SBATCH --job-name=ranch-quad
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
# Quadrature diagnostic for the record adult cell (concept EIG): Exp 1 + Exp 2 at 512 rollouts per pair on the
# record seeds, for the record quadrature (80x30 linear: must reproduce the record tables) and finer ones.
set -euo pipefail
module load python/3.12.1
export RANCH_ROOT=$HOME/ranch_smoke                 # side clone at the record commit; the record tree is never touched
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$RANCH_ROOT/pkbb_paper_writing
source $HOME/ranch/venv/bin/activate
cd $HOME/ranch_diag
echo "$(cd $RANCH_ROOT/RANCH_model && git log -1 --oneline)"
for spec in "80 30 linear" "80 120 linear" "160 120 linear" "80 30 log"; do
  set -- $spec
  echo "=== $(date '+%F %T')  quadrature ${1}x${2} $3"
  python adult_quadrature_diag.py $1 $2 $3 512 quad_${1}x${2}_$3.csv $SLURM_CPUS_PER_TASK
done
echo "=== $(date '+%F %T')  done"
