#!/bin/bash
#SBATCH -p mcfrank
#SBATCH --job-name=ranch-adults-R64
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out

module load python/3.12.1
export RANCH_ROOT=$HOME/ranch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source $RANCH_ROOT/venv/bin/activate
cd $RANCH_ROOT/RANCH_model
python granch_fast/phase1e_adults_winners.py --procs 24
