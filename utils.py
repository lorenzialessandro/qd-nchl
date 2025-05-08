import yaml
from copy import deepcopy
from network import NCHL

def load_simple_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_config(config_file, task, seed):
    # Load the full configuration
    full_config = load_simple_config(config_file)
    
    # Create the final config by merging default with task-specific
    config = deepcopy(full_config['default'])
    config.update(full_config['tasks'][task])
    
    nchl = NCHL(config["nodes"], grad=False)
    config["seed"] = seed
    config["length"] = nchl.nparams
    config["dir"] = f"{task}_{seed}"
    
    return config