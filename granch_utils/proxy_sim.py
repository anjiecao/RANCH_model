
import torch
from . import compute_prob_tensor
import numpy as np
import pandas as pd
import math
import ipdb

def granch_proxy_sim(params, model, stimuli): 

    prev_observation_posterior = None
    stimulus_idx = 0
    t = 0 # following python tradition we are using 0-indexed
    current_stim_t = 0
    while t < params.max_observation and stimulus_idx < stimuli.n_trial: 
        # update model behavior with current t and current stimulus_idx 
        model.current_t = t 
        model.current_stimulus_idx = stimulus_idx

        if model.current_t == 0: 
            prior =  params.lp_epsilon.mean(dim = 2) + params.lp_mu_sigma.mean(dim = 2)
            padded_prior = prior.unsqueeze(0).repeat(3, 1, 1, 1)   
            normalizing_term = torch.logsumexp(padded_prior, dim = (1, 2, 3)).view(3, 1, 1, 1)
            normalized_prior= torch.exp(padded_prior - normalizing_term)
            normalized_prior[normalized_prior < 0.0000000000000000000000000000001] = 0.0000000000000000000000000000001
            prev_observation_posterior = normalized_prior

        else: 
            prev_observation_posterior = model.cur_posterior

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

        #  calculate surprisal here 
        if params.linking_hypothesis == "surprisal": 

            surprisal = compute_prob_tensor.score_surprisal(model, params, prev_observation_posterior)
            model.update_model_surprisal(surprisal.item())
            stimulus_idx, current_stim_t = model.make_decision(params, stimulus_idx, current_stim_t, metric = surprisal)
            # experiment with surpirsal function here 
            #f_s = surprisal.item() 
            #f_s = math.exp(f_s)
            #a = -5  # Assuming the parabola opens downwards
            #f_s = math.exp(a * (f_s - 0.8)**2)
            #world_eig = math.exp(-4)

        current_likelihood = compute_prob_tensor.score_likelihood(model, params, hypothetical_obs=False)
        model.cur_likelihood = current_likelihood
        
        current_posterior = compute_prob_tensor.score_posterior(model,params, hypothetical_obs=False)
        model.cur_posterior = current_posterior     

        # CAN CALCULATE KL HERE
        if params.linking_hypothesis == "KL" or params.linking_hypothesis == "kl": 
            kl = compute_prob_tensor.kl_div(model.cur_posterior, prev_observation_posterior, context = "proxy")
            kl_sum  = torch.sum(kl)
            model.update_model_kl(kl_sum.item())
            stimulus_idx, current_stim_t = model.make_decision(params, stimulus_idx, current_stim_t, metric = kl_sum)
    
        
        if params.linking_hypothesis == "quad_surprisal": 
            continue # need to fill this out

        t += 1  
        current_stim_t += 1 

    #return(model)
        # end of while loop
    output  = model.behavior.groupby("stimulus_id").size()
    output_df = output.reset_index(name='sample_n')

    if np.isnan(params.forced_exposure_max) == False:
        # only saving the last because we are in the forced exposure paradigm
        output_df = output_df.tail(1)
    
    model.output = output_df[["sample_n"]]
    return (model)