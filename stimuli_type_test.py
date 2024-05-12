import numpy as np
import torch 
import pyro
import re
import pandas as pd

import importlib
import time
import pickle
import os

from granch_utils import init_model_tensor, init_stimuli_tensor,init_params_tensor, main_sim_tensor, lesioned_sim, compute_prob_tensor,  num_stab_help, proxy_sim, get_embedding, get_sequence
#importlib.reload(granch_utils)
importlib.reload(num_stab_help)
importlib.reload(init_model_tensor)
importlib.reload(init_stimuli_tensor)
importlib.reload(main_sim_tensor)
importlib.reload(lesioned_sim)
importlib.reload(get_embedding)
importlib.reload(get_sequence)

importlib.reload(main_sim_tensor)
importlib.reload(lesioned_sim)
importlib.reload(init_params_tensor)
importlib.reload(init_model_tensor)
importlib.reload(compute_prob_tensor)
importlib.reload(proxy_sim)



summarized_main_sim  = []

 #trial_info = pd.read_csv("/om2/scratch/tmp/galraz/RANCH/RANCH_cluster/sim_info/trial_info/trial_info_graded_dishab.csv")\
trial_info = pd.read_csv("trial_info.csv")

    # just first row
    #trial_info = trial_info.iloc[2]
    #trial_info = trial_info.to_dict()

background_trials = trial_info[trial_info["violation_type"]=="background"]
identity_trials = trial_info[trial_info["violation_type"]=="identity"]
number_trials = trial_info[trial_info["violation_type"]=="number"]
pose_trials = trial_info[trial_info["violation_type"]=="pose"]
animacy_trials = trial_info[trial_info["violation_type"]=="animacy"]


background_trial_info = background_trials.iloc[np.random.randint(0, len(background_trials))].to_dict()
identity_trial_info = identity_trials.iloc[np.random.randint(0, len(identity_trials))].to_dict()
number_trial_info = number_trials.iloc[np.random.randint(0, len(number_trials))].to_dict()
pose_trial_info = pose_trials.iloc[np.random.randint(0, len(pose_trials))].to_dict()
animacy_trial_info = animacy_trials.iloc[np.random.randint(0, len(animacy_trials))].to_dict()

    # 2. Convert Stimuli_info into actual embedding 
    #fam, test = get_embedding.string_to_embedding(trial_info = trial_info)
#background_fam, background_test = get_embedding.string_to_embedding(trial_info = background_trial_info)
#identity_fam, identity_test = get_embedding.string_to_embedding(trial_info = identity_trial_info)
#num_fam, num_test = get_embedding.string_to_embedding(trial_info = number_trial_info)
#pose_fam, pose_test = get_embedding.string_to_embedding(trial_info = pose_trial_info)
#animacy_fam, animacy_test = get_embedding.string_to_embedding(trial_info = animacy_trial_info)

background_fam, background_test = torch.tensor([0.4533, -0.9634, -0.7922]), torch.tensor([0.4533, -0.9634, -0.7922])
identity_fam, identity_test = torch.tensor([ 0.8621, -0.0764,  0.2498]), torch.tensor([1.5216,  0.1411, -0.1434])
num_fam, num_test = torch.tensor([ -0.3219,  0.8121,  0.7056]), torch.tensor([-0.5816,  1.4219,  0.2670])
pose_fam, pose_test = torch.tensor([ 1.8067, -0.2556, -0.6386]), torch.tensor([1.6578, -0.1109, -0.7233])
animacy_fam, animacy_test = torch.tensor([-1.3875, -1.4487,  0.0324]), torch.tensor([1.8204, -0.4493, -0.0524])
    
    

    #print("fam:", fam, "test:", test)


    # 3. Convert trial information into sequence
    #sequence_scheme = get_sequence.param_to_scheme(trial_info=trial_info)

sequence_scheme = "BBBBBD"
    #background_sequence_scheme = get_sequence.param_to_scheme(trial_info=background_trial_info)
    #identity_sequence_scheme = get_sequence.param_to_scheme(trial_info=identity_trial_info)
    #number_sequence_scheme = get_sequence.param_to_scheme(trial_info=number_trial_info)
    #pose_sequence_scheme = get_sequence.param_to_scheme(trial_info=pose_trial_info)
    #animacy_sequence_scheme = get_sequence.param_to_scheme(trial_info=animacy_trial_info)
background_sequence_scheme = sequence_scheme
identity_sequence_scheme = sequence_scheme
number_sequence_scheme = sequence_scheme
pose_sequence_scheme = sequence_scheme
animacy_sequence_scheme = sequence_scheme
    # ------ Set up simulation raw material ------ #
    # 4. Set up stimuli
background_s = init_stimuli_tensor.granch_stimuli(background_trial_info["feature_n"], background_sequence_scheme)
background_s.add_stimuli_sequence(background_fam, background_test)

identity_s = init_stimuli_tensor.granch_stimuli(identity_trial_info["feature_n"], identity_sequence_scheme)
identity_s.add_stimuli_sequence(identity_fam, identity_test)

number_s = init_stimuli_tensor.granch_stimuli(number_trial_info["feature_n"], number_sequence_scheme)
number_s.add_stimuli_sequence(num_fam, num_test)

pose_s = init_stimuli_tensor.granch_stimuli(pose_trial_info["feature_n"], pose_sequence_scheme)
pose_s.add_stimuli_sequence(pose_fam, pose_test)

animacy_s = init_stimuli_tensor.granch_stimuli(animacy_trial_info["feature_n"], animacy_sequence_scheme)
animacy_s.add_stimuli_sequence(animacy_fam, animacy_test)



for i in range(100): 
    print(i)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    BATCH_INFO = {
        "jitter_n": 1, 
        "total_batch_n": 1, 
        "jitter_mode": "sampling"
    }

    GRID_INFO = {
        "grid_mu_start": -4, "grid_mu_end": 4, "grid_mu_step": 5, 
        "grid_sigma_start": 0.001, "grid_sigma_end": 1.8, "grid_sigma_step": 5, 
        "grid_y_start": -4, "grid_y_end": 4, "grid_y_step": 5, 
        "grid_epsilon_start": 0.00000000001, "grid_epsilon_end": 1, "grid_epsilon_step": 5, 
        "hypothetical_obs_grid_n": 10
    }


    BATCH_GRID_INFO = num_stab_help.get_batch_grid(BATCH_INFO, GRID_INFO)

    PRIOR_INFO = {
        "mu_prior": 0,  
        "V_prior":3, 
        "alpha_prior": 10, 
        "beta_prior": 0.1, 
        "epsilon": 0.0001, "mu_epsilon":0.001	, "sd_epsilon": 0.1, 
        "hypothetical_obs_grid_n": 5, 
        "world_EIGs": 0.0001	, "max_observation": 500
    }

    # tensor_stimuli = num_stab_help.sample_spore_experiment(pair_each_stim = 1, n_feature=3)

   
    #tensor_model =  init_model_tensor.granch_model(PRIOR_INFO['max_observation'], s)
    
    background_tensor_model =  init_model_tensor.granch_model(PRIOR_INFO['max_observation'], background_s)
    identity_tensor_model =  init_model_tensor.granch_model(PRIOR_INFO['max_observation'], identity_s)
    number_tensor_model =  init_model_tensor.granch_model(PRIOR_INFO['max_observation'], number_s)
    pose_tensor_model =  init_model_tensor.granch_model(PRIOR_INFO['max_observation'], pose_s)
    animacy_tensor_model =  init_model_tensor.granch_model(PRIOR_INFO['max_observation'], animacy_s)



    params = init_params_tensor.granch_params(
                    grid_mu =  BATCH_GRID_INFO["grid_mus"][0].to(device),
                    grid_sigma = BATCH_GRID_INFO["grid_sigmas"][0].to(device),
                    grid_y = BATCH_GRID_INFO["grid_ys"][0].to(device),
                    grid_epsilon = BATCH_GRID_INFO["grid_epsilons"][0].to(device),
                    hypothetical_obs_grid_n = PRIOR_INFO["hypothetical_obs_grid_n"], 
                    mu_prior = PRIOR_INFO["mu_prior"],
                    V_prior = PRIOR_INFO["V_prior"], 
                    alpha_prior = PRIOR_INFO["alpha_prior"], 
                    beta_prior = PRIOR_INFO["beta_prior"],
                    epsilon  = PRIOR_INFO["epsilon"], 
                    mu_epsilon = PRIOR_INFO["mu_epsilon"], 
                    sd_epsilon = PRIOR_INFO["sd_epsilon"], 
                    world_EIGs = PRIOR_INFO["world_EIGs"],
                    max_observation = PRIOR_INFO["max_observation"], 
                    forced_exposure_max= np.nan, 
                    linking_hypothesis = "EIG")
            
                # add the various different cached bits
    params.add_meshed_grid()
    params.add_lp_mu_sigma()
    params.add_y_given_mu_sigma()
    params.add_lp_epsilon()
    params.add_priors()
    pd.set_option('display.max_rows', None)


    if params.linking_hypothesis == "EIG": 
        background_res = main_sim_tensor.granch_main_simulation(params, background_tensor_model, background_s).output
        identity_res = main_sim_tensor.granch_main_simulation(params, identity_tensor_model, identity_s).output
        number_res = main_sim_tensor.granch_main_simulation(params, number_tensor_model, number_s).output
        pose_res = main_sim_tensor.granch_main_simulation(params, pose_tensor_model, pose_s).output
        animacy_res = main_sim_tensor.granch_main_simulation(params, animacy_tensor_model, animacy_s).output

        #lesioned_res = lesioned_sim.granch_no_noise_simulation(params, tensor_model, s)

        background_res["run_id"] = i
        identity_res["run_id"] = i
        number_res["run_id"] = i
        pose_res["run_id"] = i
        animacy_res["run_id"] = i

        background_res["v_type"] = "background"
        identity_res["v_type"] = "identity"
        number_res["v_type"] = "number"
        pose_res["v_type"] = "pose"
        animacy_res["v_type"] = "animacy"

        summarized_main_sim.append(background_res)
        summarized_main_sim.append(identity_res)
        summarized_main_sim.append(number_res)
        summarized_main_sim.append(pose_res)
        summarized_main_sim.append(animacy_res)
        print(summarized_main_sim)

        
    else: 
        #print(params.linking_hypothesis)
        res = proxy_sim.granch_proxy_sim(params, tensor_model, s)
        output = res.output

        output["run_id"] = i
        print(i)
        summarized_main_sim.append(output)

   

main_sims_df = pd.concat(summarized_main_sim)


#res = lesioned_sim.granch_no_learning_simulation(params, tensor_model, tensor_stimuli[0])
#res = lesioned_sim.granch_no_noise_simulation(params, tensor_model, tensor_stimuli[0])


#res.behavior.to_csv("diagnostics/s.csv")

main_sims_df.to_csv("stim_type_replicate_cluster.csv")