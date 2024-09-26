import json
import torch
import os
import dataclasses

result_dir = "results"


def make_hyperparameter_dict(parameters):
    hyperparameter_dict = dataclasses.asdict(parameters)
    return hyperparameter_dict


def write_json(experiment_name, data):

    save_dir = os.path.join(result_dir,experiment_name)

    # create results folder, if does not exist
    os.makedirs(result_dir, exist_ok=True)
    # create save folder within results folder, if does not exist
    os.makedirs(save_dir, exist_ok=True)

    hyperparameter_dict = make_hyperparameter_dict(data)

    with open(os.path.join(save_dir, f"{experiment_name}.json"), 'w') as f:
        json.dump(hyperparameter_dict, f)


def save_model(experiment_name,trained_model):

    save_dir = os.path.join(result_dir,experiment_name)

    # create results folder, if does not exist
    os.makedirs(result_dir, exist_ok=True)
    # create save folder within results folder, if does not exist
    os.makedirs(save_dir, exist_ok=True)

    torch.save(trained_model.state_dict(), os.path.join(save_dir, f"{experiment_name}_model.pt"))

