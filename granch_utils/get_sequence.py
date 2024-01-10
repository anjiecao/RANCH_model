
def param_to_scheme(trial_info):
    fam_duration = trial_info["fam_duration"]
    test_type = trial_info["test_type"]
    scheme = "B" * (fam_duration + 1)

    if test_type == "novel": 
       scheme = scheme[:-1] + "D"
    return scheme
    