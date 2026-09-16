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

Parallel scheme (2026-09-14): `regen_pipeline.sbatch` runs on the owner node from `~/ranch`;
`window_part1.sbatch` runs concurrently on `-p owners --requeue` from a SEPARATE clone
(`~/ranch_window`, same bundle + copied data) so the two never touch the same files; then
`window_part2.sbatch` (`--dependency=afterok:<regen>:<part1>`) imports part 1's outputs, merges
the exemplar-mean implemented-EIG rows into the regenerated adult sweeps, and runs the stages
that need both. `window_pass.sbatch` is the serial alternative. Sherlock's git has no `-C`.

Sizing note (2026-09-14): the first ext design (72 settings, 10 pairs, 24 rollouts) measured
~25 h on the node — realized looks under noise are long (~40 samples/trial) — and was cut to
32 settings x 6 pairs x 16 rollouts.

Code sync (2026-09-15, replaces the git-bundle + scp scheme): push the branch to GitHub
(`git push origin exact-inference-reboot`; both repos are public and writable); Sherlock's clone
has `origin` = GitHub. Every sbatch file starts with the same code-only sync block (tested by
`tests/test_ops.py`, which applies it to a clone one commit behind):

```bash
git fetch -q origin exact-inference-reboot
git reset -q --soft FETCH_HEAD
git ls-tree -r --name-only HEAD | grep -v "^granch_fast/phase1/" | tr '\n' '\0' | xargs -0 git checkout -q HEAD --
if git status --short | grep -v "^?? " | grep -v "granch_fast/phase1/" | grep -q .; then echo "SYNC FAILED"; exit 1; fi
```

Two rules behind it. (1) **Never `git reset --hard` on Sherlock**: `granch_fast/phase1/*.csv` are
tracked, so a hard reset overwrites the regenerated tables with the committed laptop versions
(it cost three tables on 2026-09-15). (2) **NUL-separated paths**: the repo tracks four paths
containing spaces (`diagnostics/.Rproj.user/...`); plain `xargs` split them, git rejected the
whole checkout batch, and — because a failing pipeline inside an `&&` list does not trip
`set -e` — two jobs silently ran two-commits-old code (5.4 node-hours, 2026-09-15). The block
above is line-by-line, so any failure aborts the job, and the guard line checks the result.
The sbatch file itself is read at submission, so run the block on the login node before `sbatch`.

Before submitting a chain, run it end to end at reduced size against a scratch copy of the tree:
`RANCH_ROOT=<scratch> PY=<python> PROCS=6 bash sherlock/smoke_concept.sh` (~15 min on the laptop;
one trial row per condition, 1 rollout, 1 pair). Five consecutive jobs died at steps that had
never been exercised; the smoke is the cheap answer.

Outputs land in `granch_fast/phase1/*.csv` (small; commit or rsync back):
`adult_preds_selfcons_ext.csv`, `adult_scores21_selfcons_ext.csv`,
`adult_winners_R64.csv`, `lesion_noiseless_learner_{infants,adults}.csv`.

Sync back to the laptop:

```bash
rsync -av sherlock:ranch/RANCH_model/granch_fast/phase1/'*'{selfcons_ext,R64,lesion}'*'.csv \
      /Users/mcfrank/Projects/ranch/RANCH_model/granch_fast/phase1/
```
