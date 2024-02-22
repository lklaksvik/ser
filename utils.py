import json
import torch
import os

save_dir = "results"

def make_hyperparameter_dict():
    return hyperparameter_dict

def write_json(name, data):
    os.makedirs(save_dir)

    hyperparameter_dict = make_hyperparameter_dict(data)

    with open(os.path.join(save_dir, f"{name}_json"), 'w') as f:
        json.dump(hyperparameter_dict, f)

def save_model(experiment_name,trained_model):
    os.makedirs(save_dir)
    torch.save(trained_model.state_dict(), save_dir+f"{experiment_name}_model.pt")

