#!/bin/bash
#SBATCH -p mcfrank
#SBATCH --job-name=ranch-regen
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out
# Regenerate every table that depends on the true EIG after the eq-15 fix of
# 2026-09-14 (ENGINEERING_PLAN §5.6): the whole mi-affected pipeline, in order,
# all metrics (the caches are monolithic), fixed code, one node. ~10 h.
set -euo pipefail
module load python/3.12.1
export RANCH_ROOT=$HOME/ranch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source $RANCH_ROOT/venv/bin/activate
cd $RANCH_ROOT/RANCH_model
PY=python
step() { echo "=== $(date '+%F %T')  $*"; }

step phase1 infants main;        $PY granch_fast/phase1_infants.py --which main --procs 24
step phase1 infants infeps;      $PY granch_fast/phase1_infants.py --which infeps --procs 24
step selfcons base;              $PY granch_fast/phase1_selfconsistent.py --which base --procs 24
step selfcons ext;               $PY granch_fast/phase1_selfconsistent.py --which ext --procs 24
step score infants;              $PY granch_fast/score_phase1_infants.py --which main --procs 24
                                 $PY granch_fast/score_phase1_infants.py --which infeps --procs 24
step score selfcons;             $PY granch_fast/score_phase1_selfcons.py --procs 24
step selfcons winners R32;       $PY granch_fast/phase1c_selfcons_winners.py --procs 24
step adults main mean-field;     $PY granch_fast/phase1_adults.py --which main --procs 24 --pairs 6
step score adults;               $PY granch_fast/score_phase1_adults.py --which main
                                 $PY granch_fast/score_phase1_adults21.py main
step phase2 deterministic;       $PY granch_fast/run_phase2.py
step adults selfcons base;       $PY granch_fast/phase1b_adults_selfcons.py --which base --procs 24
step score adults selfcons base; $PY granch_fast/score_phase1_adults.py --which selfcons
                                 $PY granch_fast/score_phase1_adults21.py selfcons
step adults selfcons ext;        $PY granch_fast/phase1b_adults_selfcons.py --which ext --procs 24
step adults winners R64;         $PY granch_fast/phase1e_adults_winners.py --procs 24
step phase2 selfcons;            $PY granch_fast/run_phase2_selfcons.py --procs 24 --adults ext
step lesions;                    $PY granch_fast/phase1d_lesions_selfcons.py --procs 24
step channels;                   $PY granch_fast/gen_figs_channels.py
step done
