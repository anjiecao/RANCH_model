# Running the RANCH pipeline on Sherlock

Layout expected on Sherlock (`RANCH_ROOT=$HOME/ranch`):

```
$HOME/ranch/
  RANCH_model/            # this repo, branch exact-inference-reboot (origin = GitHub)
  pkbb_paper_writing/     # data subset only (rsync'd, not the repo):
    reproduce_cv.py
    data/infants/{exp1.csv, exp2_zoom.csv, exp2_lookit.csv, unscaled_model_data/*}
    data/adults/adult_exposure_duration.csv
    data/results_plots/*.csv
  RANCH_cluster/sim_info/ # the Exp-2 stimulus-pair tables
  venv/                   # python -m venv; pip install numpy==1.26.4 scipy==1.13.1 pandas==2.2.3
```

Every path resolves through the `RANCH_ROOT` env var (default = the laptop layout
`/Users/mcfrank/Projects/ranch`). The pipeline is the `ranch` package: `python -m ranch <stage>`
(see `python -m ranch --help`); the drivers that produced the 2026-08/09 tables are frozen under
`granch_fast/legacy/` for the identity tests and are never scheduled.

Setup (once):

```bash
module load python/3.12.1
mkdir -p ~/ranch && cd ~/ranch
git clone -b exact-inference-reboot https://github.com/anjiecao/RANCH_model
python3 -m venv venv && source venv/bin/activate && pip install numpy==1.26.4 scipy==1.13.1 pandas==2.2.3
# then rsync the pkbb data subset and RANCH_cluster/sim_info from the laptop (see paths above)
```

Jobs (owner node `-p mcfrank`, 24 cores, 192 GB):

```bash
cd ~/ranch/RANCH_model/sherlock
sbatch pipeline.sbatch                          # THE reproduction command: every table + figure datum, ~12 h
STAGES="winners phase2 figures" sbatch pipeline.sbatch   # a subset of the chain
sbatch gate.sbatch                              # the API outputs vs the frozen legacy tables (phase1_ranch/ vs phase1/)
```

Code sync: push the branch to GitHub (`git push origin exact-inference-reboot`); every sbatch
file starts with the same code-only sync block (tested by `tests/test_ops.py`, which applies it to
a clone one commit behind):

```bash
git fetch -q origin exact-inference-reboot
git reset -q --soft FETCH_HEAD
git ls-tree -r --name-only HEAD | grep -v "^granch_fast/phase1/" | tr '\n' '\0' | xargs -0 git checkout -q HEAD --
if git status --short | grep -v "^?? " | grep -v "granch_fast/phase1/" | grep -q .; then echo "SYNC FAILED"; exit 1; fi
```

Two rules behind it. (1) **Never `git reset --hard` on Sherlock**: `granch_fast/phase1/*.csv` are
tracked, so a hard reset overwrites the regenerated tables with the committed versions (it cost
three tables on 2026-09-15, and five more surfaced in the first gate run). (2) **NUL-separated
paths**: the repo tracks four paths containing spaces (`diagnostics/.Rproj.user/...`); plain
`xargs` split them, git rejected the whole checkout batch, and -- because a failing pipeline
inside an `&&` list does not trip `set -e` -- two jobs silently ran two-commits-old code (5.4
node-hours, 2026-09-15). The block is line-by-line, so any failure aborts the job, and the guard
line checks the result. The sbatch file itself is read at submission, so run the block on the
login node before `sbatch`.

Before submitting the chain, run it end to end at reduced size against a scratch copy of the tree:
`RANCH_ROOT=<scratch> PY=<python> PROCS=6 bash sherlock/smoke_pipeline.sh` (~15 min on the laptop;
one trial row per condition, 1 rollout, 1 pair). Five consecutive jobs once died at steps that had
never been exercised; the smoke is the cheap answer.

Outputs land in `granch_fast/phase1/` (the tables are tracked; the `*.npz` grids are not -- they
stay on the cluster). Sync tables back to the laptop:

```bash
rsync -av sherlock:ranch/RANCH_model/granch_fast/phase1/'*.csv' /Users/mcfrank/Projects/ranch/RANCH_model/granch_fast/phase1/
rsync -av sherlock:ranch/RANCH_model/granch_fast/'*.csv' /Users/mcfrank/Projects/ranch/RANCH_model/granch_fast/   # figure data
```

Job logs and the gate verdicts are archived under `sherlock/logs/`.
