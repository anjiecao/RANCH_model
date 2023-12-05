from distutils.log import warn
from pickle import NONE, TRUE
import pandas as pd
import numpy as np
import torch 
import re 
from torch.distributions import Normal  
#import ipdb



class granch_model: 
    def __init__(self, max_observation, stimuli):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.current_t = 0
        self.current_stimulus_idx = 0

        self.stimuli = stimuli

        self.max_observation = max_observation
        self.behavior = pd.DataFrame(None, index=np.arange(max_observation),
                                     columns=["stimulus_id", "EIG", "Look_away"])
        
        
        self.behavior["surprisal"] = np.nan
        self.behavior["kl"] = np.nan
        self.behavior["prob"] = np.nan

        self.possible_observations = None 

        
        self.all_observations = pd.DataFrame(0, index=np.arange(max_observation),
                                            columns = np.arange(stimuli.n_feature))
        
       
        # try to just cached the last stimulus likelihood
        self.cur_likelihood = None
        self.prev_likelihood = None
        self.cur_posterior = None



        self.ps_kl = torch.tensor([])
        self.ps_pp = torch.tensor([])


    def update_model_stimulus_id(self): 
        self.behavior.at[self.current_t, "stimulus_id"] = self.current_stimulus_idx

    def update_model_surprisal(self, surprisal):
        self.behavior.at[self.current_t, "surprisal"] = surprisal
    
    def update_model_prob(self, prob):
        self.behavior.at[self.current_t, "prob"] = prob

    def update_model_kl(self, kl):
        self.behavior.at[self.current_t, "kl"] = kl
    
    def update_model_eig(self, eig):
        self.behavior.at[self.current_t, "EIG"] = eig

    def update_model_decision(self, decision): 
        self.behavior.at[self.current_t, "Look_away"] = decision

    def update_possible_observations(self, noise_epsilon, hypothetical_obs_grid_n): 
        
        current_stimuli = self.stimuli.stimuli_sequence[self.current_stimulus_idx]
        expanded_tensors = [torch.linspace(val - noise_epsilon, val + noise_epsilon, hypothetical_obs_grid_n) for val in current_stimuli]

        expanded_tensor = torch.stack(expanded_tensors, dim=1).t()
        d1 = expanded_tensor[0]
        d2 = expanded_tensor[1]
        d3 = expanded_tensor[2]

        grid_1, grid_2, grid_3= torch.meshgrid(d1, d2, d3)

        # Stack the grids to form tensor C
        all_possible_obs = torch.stack((grid_1, grid_2, grid_3), dim=-1).view(-1, 3)


        self.possible_observations = all_possible_obs
       

    def update_noisy_observation(self, noise_epsilon): 
        current_stimulus = self.stimuli.stimuli_sequence[self.current_stimulus_idx]
        self.all_observations.loc[self.current_t]  = Normal(current_stimulus, noise_epsilon).sample().tolist()

    def get_all_observations_on_current_stimulus(self): 
        obs_index = self.behavior.index[self.behavior['stimulus_id'] == 
                                    self.current_stimulus_idx].tolist()
        return torch.tensor(self.all_observations.iloc[obs_index].values).to(self.device)

    def get_current_observation(self):
        return torch.tensor(self.all_observations.iloc[self.current_t].values) 

    def get_last_stimuli_likelihood(self): 
        last_stimuli_last_obs_t = max(self.behavior.index[self.behavior['stimulus_id'] == 
                                    self.current_stimulus_idx-1].tolist())
        return self.all_likelihood[last_stimuli_last_obs_t]

    def if_same_stimulus_as_previous_t(self): 

        last_t_stimulus = max(self.behavior[pd.notnull(self.behavior["stimulus_id"])]["stimulus_id"])
        return (self.current_stimulus_idx == last_t_stimulus)
    
    def make_decision(self, params, stimulus_idx, current_stim_t, eig):
        
        if ~np.isnan(params.forced_exposure_max): 
            # if it's not the last trial, you still have to look
            if (stimulus_idx < (self.stimuli.n_trial - 1)) & (current_stim_t < params.forced_exposure_max - 1):
                self.update_model_decision(False)

            # if i'm in a fam trial and i reached the max exposure, i have to look away (to go to next stimulus)
            elif (stimulus_idx < (self.stimuli.n_trial - 1)) & (current_stim_t == params.forced_exposure_max - 1):
                self.update_model_decision(True)
                stimulus_idx += 1
                current_stim_t = -1 

            else:
                p_look_away = max(min(params.world_EIGs / (eig.item() + params.world_EIGs), 1), 0)
                    
                if (np.random.binomial(1, p_look_away) == 1): 
            # if the model is looking away, increment stimulus
                    stimulus_idx = stimulus_idx + 1
                    current_stim_t = -1 # -1 so it starts with 0 when incremented 
                    self.update_model_decision(True)
                else: 
                # otherwise keep looking at this one
                    self.update_model_decision(False)

        # if it's a self-paced paradigm
        else:
            # luce's choice rule 
            p_look_away = max(min(params.world_EIGs / (eig.item() + params.world_EIGs), 1), 0)
            #p_look_away = params.world_EIGs / (eig.item() + params.world_EIGs)
         
            if (np.random.binomial(1, p_look_away) == 1): 
            # if the model is looking away, increment stimulus
                stimulus_idx = stimulus_idx + 1
                current_stim_t = -1 # -1 so it starts with 0 when incremented 
                self.update_model_decision(True)
            else: 
            # otherwise keep looking at this one
                self.update_model_decision(False)



        

        
        



