"""ENGINEERING_PLAN §3.9 -- operations: thread pinning under multiprocessing, time budgets,
cluster scripts."""
import glob
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from conftest import ROOT
from granch_fast import metrics as M

GF = f"{ROOT}/granch_fast"


def test_every_multiprocessing_driver_pins_blas_threads():
    """Any script that opens a Pool must pin OMP/BLAS threads to 1 (the load-200 incident).
    The package's pools rely on the caller's environment (the cluster scripts export the pins)."""
    offenders = []
    for f in glob.glob(f"{GF}/*.py") + glob.glob(f"{GF}/audit/*.py") + glob.glob(f"{GF}/legacy/*.py"):
        src = open(f).read()
        if "Pool(" in src and "OMP_NUM_THREADS" not in src:
            offenders.append(os.path.basename(f))
    assert not offenders, offenders


def test_driver_import_sets_thread_env_in_a_clean_process():
    env = {k: v for k, v in os.environ.items() if not k.endswith("_NUM_THREADS")}
    env["RANCH_ROOT"] = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch")
    code = ("import os, sys; sys.path.insert(0, %r); import granch_fast.legacy.phase1_selfconsistent; "
            "print(os.environ['OMP_NUM_THREADS'])" % ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1"


def test_sbatch_scripts_parse():
    for f in glob.glob(f"{ROOT}/sherlock/*.sbatch"):
        r = subprocess.run(["bash", "-n", f], capture_output=True, text=True)
        assert r.returncode == 0, (f, r.stderr)
        assert "-p mcfrank" in open(f).read() or "-p owners" in open(f).read()


SYNC_START, SYNC_END = "git fetch -q origin exact-inference-reboot", "git log --oneline -1"


def _sync_block(sbatch):
    lines = open(sbatch).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(SYNC_START))
    j = next(k for k, l in enumerate(lines) if k > i and l.startswith(SYNC_END))
    return "\n".join(lines[i:j + 1])


def test_cluster_jobs_sync_code_only_and_all_the_same_way():
    """`git reset --hard` in a job clobbers the tracked score tables in granch_fast/phase1
    with the committed (stale) versions -- it cost the 2026-09-15 regeneration three tables.
    Jobs must sync code only, and every job must use the one block exercised below."""
    blocks = set()
    for f in glob.glob(f"{ROOT}/sherlock/*.sbatch"):
        src = open(f).read()
        assert "reset --hard" not in src, f
        if "git fetch" in src:
            blocks.add(_sync_block(f))
    assert len(blocks) == 1, "sync blocks differ across sbatch files"
    (block,) = blocks
    assert 'grep -v "^granch_fast/phase1/"' in block and "xargs -0" in block


def test_sync_block_applies_to_a_clone_one_commit_behind(tmp_path):
    """The repo tracks four paths containing spaces; plain `xargs` split them and git rejected
    the whole checkout batch, so the 2026-09-15 concept study and gate silently ran stale
    code (5.4 node-hours lost). Apply the sbatch sync block to a clone one commit behind: it
    must end at HEAD with every tracked file outside granch_fast/phase1/ equal to HEAD."""
    if os.path.exists(f"{ROOT}/.git/shallow"):
        pytest.skip("shallow clone: no commit behind HEAD to reset to (the side-run clones on the cluster)")
    block = _sync_block(glob.glob(f"{ROOT}/sherlock/*.sbatch")[0])
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "--shared", "-b", "exact-inference-reboot", ROOT, str(clone)], check=True)
    subprocess.run(["git", "reset", "-q", "--hard", "HEAD~1"], cwd=clone, check=True)
    r = subprocess.run(["bash", "-euo", "pipefail", "-c", block], cwd=clone, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rev = lambda cwd: subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True).stdout.strip()
    assert rev(clone) == rev(ROOT)
    dirty = subprocess.run(["git", "status", "--short"], cwd=clone, capture_output=True, text=True).stdout.splitlines()
    assert all(l.startswith("?? ") or "granch_fast/phase1/" in l for l in dirty), dirty
    spaced = [p for p in subprocess.run(["git", "ls-files"], cwd=clone, capture_output=True, text=True).stdout.splitlines() if " " in p]
    assert spaced and all((clone / p).exists() for p in spaced)


def test_cluster_jobs_use_only_the_package_entry_points():
    """Every command a sbatch file runs is a `python -m ranch ...` stage or `python -m ranch.gate`
    (the legacy drivers are frozen under granch_fast/legacy and must not be scheduled), and every
    stage a job uses starts up (`--help`) in a clean process -- py_compile does not catch
    NameErrors at argparse construction (a 2026-09-15 job died that way at its last-but-three step)."""
    import re
    env = dict(os.environ, RANCH_ROOT=os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch"))
    stages = set()
    for sb in glob.glob(f"{ROOT}/sherlock/*.sbatch") + glob.glob(f"{ROOT}/sherlock/*.sh"):
        src = open(sb).read()
        assert not re.findall(r"granch_fast/[a-z0-9_]+\.py", src), sb
        stages |= set(re.findall(r"\bR (grid|score|adults|score-adults|winners|phase2|reevaluate|figures)\b", src))
    assert stages, "no pipeline stages found in the cluster scripts"
    failures = []
    for mod in ("ranch", "ranch.gate"):
        r = subprocess.run([sys.executable, "-m", mod, "--help"], capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
        if r.returncode != 0:
            failures.append((mod, r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "?"))
    r = subprocess.run([sys.executable, "-m", "ranch", "--help"], capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    for st in sorted(stages):
        if st not in r.stdout:
            failures.append((st, "stage missing from the CLI"))
    assert not failures, failures


def test_cluster_jobs_verify_their_inputs_before_computing():
    """`R check` (the data manifest: every input file present, hashes as pinned) runs before the first
    compute stage of every cluster script, and the figure code reads no input the manifest does not
    pin -- job 43856112 (2026-09-16) ran 8 h and died in its last stage on a file never copied to the cluster."""
    import json
    import re
    for sb in glob.glob(f"{ROOT}/sherlock/*.sbatch") + glob.glob(f"{ROOT}/sherlock/*.sh"):
        ran = re.findall(r"\bR (check|grid|score|adults|score-adults|winners|phase2|reevaluate|figures)\b", open(sb).read())
        assert ran and ran[0] == "check", (sb, ran[:2])
    pinned = set(json.load(open(f"{GF}/tests/data_manifest.json")))
    for f in glob.glob(f"{ROOT}/ranch/*.py"):
        for rel in re.findall(r'\{(?:data\.)?PAPER\}/([^"]+\.csv)', open(f).read()):
            assert f"pkbb_paper_writing/{rel}" in pinned, (os.path.basename(f), rel)


def test_package_has_no_hardcoded_laptop_paths():
    """Every path resolves through RANCH_ROOT (the 2026-09-14 regeneration died at its first
    scoring step because four scorers still hardcoded the laptop path). A literal
    '/Users/mcfrank' is allowed only as the RANCH_ROOT default inside os.environ.get(...)."""
    offenders = []
    for f in glob.glob(f"{ROOT}/ranch/*.py") + [f"{GF}/metrics.py", f"{GF}/eig.py", f"{GF}/analytic_core.py", f"{GF}/run_fast.py",
                                                f"{GF}/linking_mixed.py", f"{GF}/fit_infants.py"]:
        for line in open(f):
            if "/Users/mcfrank" in line and "os.environ.get(" not in line:
                offenders.append((os.path.relpath(f, ROOT), line.strip()))
    assert not offenders, offenders


@pytest.mark.slow
def test_time_budgets(ref_inferred, stim_pair):
    cfg, grid = ref_inferred
    fam, dev = stim_pair
    st = M.State(cfg, grid, 2)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(30):
        st.step(0, fam + rng.normal(0, 0.1, 3), fam, want=("mi",))
    assert (time.perf_counter() - t0) / 30 < 0.005
    t0 = time.perf_counter()
    M.infant_trajectories(cfg, grid, fam, dev, 8, 40, rng=rng, sigma_true=0.1, want=("eig_code", "mi", "kl", "surprisal"))
    assert time.perf_counter() - t0 < 0.4


def test_package_never_imports_the_frozen_drivers():
    """granch_fast/legacy holds the retired Phase-1/2 drivers for the identity tests only; the
    package must not depend on them (they will not be maintained)."""
    offenders = [os.path.relpath(f, ROOT) for f in glob.glob(f"{ROOT}/ranch/*.py")
                 if any(("granch_fast.legacy" in line or "from granch_fast import" in line) and not line.strip().startswith("#")
                        and "import" in line and "legacy" in line for line in open(f))]
    assert not offenders, offenders
