# granch_fast test suite (ENGINEERING_PLAN.md, Phase A)

```bash
cd RANCH_model/granch_fast
../../.venv/bin/python -m pytest                 # fast suite
../../.venv/bin/python -m pytest -m slow         # statistical / long-running
../../.venv/bin/python -m pytest -m torch        # faithfulness to the original torch grid code
```

| file | plan § | what it guards |
|---|---|---|
| `test_inference.py` | 3.1 | closed-form posterior, Kalman property (Eq. 7), sufficient stats, quadrature, the published-spec degeneracy |
| `test_decision_variables.py` | 3.2 | Eq.-5 identity, KL / MI vs brute force, channel sum, properties, **no-oracle** (strict xfail for the implemented functional), symmetries |
| `test_runners.py` | 3.3 | exposure semantics, engine vs legacy runner, mean-field validity, Luce stopping, exemplar designs |
| `test_linking.py` | 3.4 | slope-sign regression, hand-checked CV, within-subject recovery under a cohort confound, baselines |
| `test_data.py` | 3.5 | loader snapshot counts (exposure-row / subject-key regressions), data manifest, provenance guard |
| `test_selection.py` | 3.6 | winner's curse, seeded reproducibility |
| `test_golden.py` | 3.7 | pinned Phase-1/2 numbers (`golden.json`) + a live reference trajectory |
| `test_phenomena.py` | 3.8 | the configuration map as a test (cached + live) |
| `test_ops.py` | 3.9 | thread pinning, time budgets, sbatch scripts |
| `test_faithfulness_torch.py` | 3.2 | the analytic engine vs the original grid implementation on dense grids |

Regenerate pinned artefacts deliberately: `make_manifest.py` (data hashes), `make_golden.py`
(numbers, after a reviewed regeneration of the tables). The original print-style audit
scripts remain in `../audit/` and are referenced by the report.
