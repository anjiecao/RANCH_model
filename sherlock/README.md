# Running the granch_fast promotion jobs on Sherlock

Layout expected on Sherlock (`RANCH_ROOT=$HOME/ranch`):

```
$HOME/ranch/
  RANCH_model/            # this repo, branch exact-inference-reboot
  pkbb_paper_writing/     # data subset only (rsync'd, not the repo):
    reproduce_cv.py
    data/infants/exp1.csv
    data/infants/unscaled_model_data/trial_info.csv
    data/adults/adult_exposure_duration.csv
  venv/                   # python -m venv; pip install numpy scipy pandas
```

All `granch_fast` paths resolve through the `RANCH_ROOT` env var
(default = the laptop layout `/Users/mcfrank/Projects/ranch`).

Setup (once):

```bash
module load python/3.12.1
mkdir -p ~/ranch && cd ~/ranch
git clone -b exact-inference-reboot https://github.com/anjiecao/RANCH_model
python3 -m venv venv && source venv/bin/activate && pip install numpy scipy pandas
# then rsync the pkbb data subset from the laptop (see paths above)
```

Jobs (all on the owner node, `-p mcfrank`, 24 cores):

```bash
cd ~/ranch/RANCH_model/sherlock
jid=$(sbatch --parsable adults_ext.sbatch)      # 72-setting noisy-adult sweep, ~4 h
sbatch --dependency=afterok:$jid adults_winners.sbatch   # R=64 winner re-evaluation
sbatch lesions.sbatch                           # no-noise lesion of model B, ~1 h
```

Outputs land in `granch_fast/phase1/*.csv` (small; commit or rsync back):
`adult_preds_selfcons_ext.csv`, `adult_scores21_selfcons_ext.csv`,
`adult_winners_R64.csv`, `lesion_noiseless_learner_{infants,adults}.csv`.

Sync back to the laptop:

```bash
rsync -av sherlock:ranch/RANCH_model/granch_fast/phase1/'*'{selfcons_ext,R64,lesion}'*'.csv \
      /Users/mcfrank/Projects/ranch/RANCH_model/granch_fast/phase1/
```
