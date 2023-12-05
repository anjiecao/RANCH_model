# version of the main simulation to support matrix multiplication instead

import torch
from . import compute_prob_tensor
import numpy as np
#import ipdb

# main simulation function
def granch_main_simulation(params, model, stimuli):
    print("running main sim") 

    stimulus_idx = 0
    t = 0 # following python tradition we are using 0-indexed
    current_stim_t = 0
    while t < params.max_observation and stimulus_idx < stimuli.n_trial: 
        print(t)

        # update model behavior with current t and current stimulus_idx 
        model.current_t = t 
        model.current_stimulus_idx = stimulus_idx
    
        # get all possible observation on current stimulus 
        # if we change stimulus 
        if model.current_t == 0 or (not model.if_same_stimulus_as_previous_t()): 

            model.update_possible_observations(params.epsilon, params.hypothetical_obs_grid_n)

            # update the previous likelihood to be the "current likelihood"
            model.prev_likelihood = model.cur_likelihood


        # update model stimulus id 
        # update the noisy observation on current model stimulus 
        model.update_model_stimulus_id()
        model.update_noisy_observation(params.epsilon)

        current_likelihood = compute_prob_tensor.score_likelihood(model, params, hypothetical_obs=False)
        model.cur_likelihood = current_likelihood
        
        current_posterior = compute_prob_tensor.score_posterior(model,params, hypothetical_obs=False)
        model.cur_posterior = current_posterior     
        

        # in the tensor mode we don't need to iterate through possibilities anymore
        model.ps_likelihood = compute_prob_tensor.score_likelihood(model, params, hypothetical_obs=True)
        model.ps_posteriror = compute_prob_tensor.score_posterior(model, params, hypothetical_obs=True)
        model.ps_kl = compute_prob_tensor.kl_div(model.ps_posteriror, model.cur_posterior)
        model.ps_pp = compute_prob_tensor.score_post_pred(model, params)
       
    

        # compute EIG
       
        eig = torch.sum(model.ps_kl * model.ps_pp)

       
        
        
        # threshold at 0 for now to deal with negative EIG's
        #eig = torch.clamp(eig, min=0)

        model.update_model_eig(eig.item())
        stimulus_idx, current_stim_t = model.make_decision(params, stimulus_idx, current_stim_t, eig)

    
        t += 1  
        current_stim_t += 1 
    
    # end of while loop
    output  = model.behavior.groupby("stimulus_id").size()
    output_df = output.reset_index(name='sample_n')
    # only saving the last because we are in the forced exposure paradigm
    output_df = output_df.tail(1)
    
    model.output = output_df[["sample_n"]]

    print("end one run of main sim")
    return(model)








