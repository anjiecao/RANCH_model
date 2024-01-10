import pandas as pd
import torch
import ipdb

def string_to_embedding(trial_info): 
   
   # loading all the key information
   embedding_type = trial_info["embedding_type"]
   feature_n = trial_info["feature_n"]
   fam = trial_info['fam'] 
   test = trial_info['test']

   print(embedding_type)

   # loading the corresponding embedding file
   if embedding_type == "resnet_pa": 
      embeddings = pd.read_csv("/om2/scratch/tmp/galraz/RANCH/RANCH_cluster/sim_info/embeddings/resnet_pa.csv")
   elif embedding_type == "resnet_saycam":
      embeddings = pd.read_csv("../sim_info/embeddings/resnet_pa.csv")

   fam_raw = embeddings[embeddings.iloc[:,0] == fam]
   test_raw =  embeddings[embeddings.iloc[:,0] == test]

   f_val = torch.tensor(fam_raw.iloc[:, 1:feature_n+1].values[0])
   t_val = torch.tensor(test_raw.iloc[:, 1:feature_n+1].values[0])

   return (f_val, t_val)




   
#elif: 
   


