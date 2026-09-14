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
sbatch regen_pipeline.sbatch                    # the whole true-EIG-dependent pipeline, ~10 h
# or the pieces:
jid=$(sbatch --parsable adults_ext.sbatch)      # 32-setting noisy-adult sweep (6 pairs x 16 rollouts)
sbatch --dependency=afterok:$jid adults_winners.sbatch   # R=64 winner re-evaluation
sbatch lesions.sbatch                           # no-noise lesion of model B, minutes
```

Sizing note (2026-09-14): the first ext design (72 settings, 10 pairs, 24 rollouts) measured
~25 h on the node — realized looks under noise are long (~40 samples/trial) — and was cut to
32 settings x 6 pairs x 16 rollouts. Sync code with a git bundle (no GitHub write access):
`git bundle create /tmp/b exact-inference-reboot; scp /tmp/b sherlock:ranch_reboot.bundle;`
then on Sherlock `git fetch ~/ranch_reboot.bundle exact-inference-reboot && git reset --hard FETCH_HEAD`.

Outputs land in `granch_fast/phase1/*.csv` (small; commit or rsync back):
`adult_preds_selfcons_ext.csv`, `adult_scores21_selfcons_ext.csv`,
`adult_winners_R64.csv`, `lesion_noiseless_learner_{infants,adults}.csv`.

Sync back to the laptop:

```bash
rsync -av sherlock:ranch/RANCH_model/granch_fast/phase1/'*'{selfcons_ext,R64,lesion}'*'.csv \
      /Users/mcfrank/Projects/ranch/RANCH_model/granch_fast/phase1/
```
