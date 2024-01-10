from torch.distributions import uniform


def generate_jitter_grid(param_info):
    
    batch_n = param_info['batch_n']
    jitter_n = param_info['jitter_n']

    GRID_INFO = {
        "grid_mu_start": -4, "grid_mu_end": 4, "grid_mu_step": 5, 
        "grid_sigma_start": 0.001, "grid_sigma_end": 1.8, "grid_sigma_step": 5, 
        "grid_y_start": -4, "grid_y_end": 4, "grid_y_step": 5, 
        "grid_epsilon_start": 0.00000000001, "grid_epsilon_end": 1, "grid_epsilon_step": 5, 
        "hypothetical_obs_grid_n": 10
    }

    grid_mu_distribution = uniform.Uniform(GRID_INFO["grid_mu_start"], GRID_INFO["grid_mu_end"])
    grid_sigma_distribution = uniform.Uniform(max(0.0000001, GRID_INFO["grid_sigma_start"]), GRID_INFO["grid_sigma_end"])
    grid_y_distribution = uniform.Uniform(GRID_INFO["grid_y_start"], GRID_INFO["grid_y_end"])
    grid_epsilon_distribution = uniform.Uniform(max( 0.0000000000000000000000000000001, GRID_INFO["grid_epsilon_start"]), GRID_INFO["grid_epsilon_end"])

    all_jitter_grid = {
          "grid_mus": [], 
          "grid_sigmas": [], 
          "grid_ys": [],
          "grid_epsilons": []
    }

    for i in range(0, batch_n * jitter_n): 
        grid_mu = grid_mu_distribution.sample([GRID_INFO["grid_mu_step"], ])
        grid_sigma = grid_sigma_distribution.sample([GRID_INFO["grid_sigma_step"], ])
        grid_y = grid_y_distribution.sample([GRID_INFO["grid_y_step"], ])
        grid_epsilon = grid_epsilon_distribution.sample([GRID_INFO["grid_epsilon_step"], ])

        all_jitter_grid["grid_mus"].append(grid_mu)
        all_jitter_grid["grid_sigmas"].append(grid_sigma)
        all_jitter_grid["grid_ys"].append(grid_y)
        all_jitter_grid["grid_epsilons"].append(grid_epsilon)
    
    return all_jitter_grid



