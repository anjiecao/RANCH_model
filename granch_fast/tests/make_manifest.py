"""Write tests/data_manifest.json: SHA-256 of every data file the pipeline reads
(ENGINEERING_PLAN §3.5). Run deliberately when a data file is *meant* to change."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RANCH = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch")
FILES = [
    "RANCH_model/resnet50_downscaled.csv",
    "pkbb_paper_writing/data/infants/exp1.csv",
    "pkbb_paper_writing/data/infants/unscaled_model_data/trial_info.csv",
    "pkbb_paper_writing/data/infants/unscaled_model_data/linked_aligned_eig_unscaled_onlyanimals.csv",
    "pkbb_paper_writing/data/infants/crossvalidation/resnet50_EIG_CV.csv",
    "pkbb_paper_writing/data/infants/exp2_zoom.csv",
    "pkbb_paper_writing/data/infants/exp2_lookit.csv",
    "pkbb_paper_writing/data/adults/adult_exposure_duration.csv",
    "pkbb_paper_writing/data/results_plots/exp1_data_plot.csv",
    "pkbb_paper_writing/data/results_plots/exp2_infant_plot.csv",
    "pkbb_paper_writing/data/results_plots/exp2_adult_plot.csv",
    "pkbb_paper_writing/data/results_plots/exp1_adult_sim_plot.csv",
    "RANCH_cluster/sim_info/trial_info/stimulus_type/infants/stimuli_pair_info.csv",
    "RANCH_cluster/sim_info/trial_info/stimulus_type/adults/stimuli_pair_info.csv",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    man = {}
    for rel in FILES:
        path = f"{RANCH}/{rel}"
        if not os.path.exists(path):
            print(f"  missing (skipped): {rel}", file=sys.stderr)
            continue
        man[rel] = sha256(path)
    json.dump(man, open(f"{HERE}/data_manifest.json", "w"), indent=1)
    print(f"wrote {len(man)} hashes to tests/data_manifest.json")


if __name__ == "__main__":
    main()
