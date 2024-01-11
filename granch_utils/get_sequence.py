
def param_to_scheme(trial_info):
    fam_duration = trial_info["fam_duration"]
    violation_type = trial_info["violation_type"]
    scheme = "B" * (fam_duration + 1)

    if violation_type != "background": 
       scheme = scheme[:-1] + "D"
    return scheme
    