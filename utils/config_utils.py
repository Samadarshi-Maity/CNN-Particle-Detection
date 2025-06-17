# contains the function to load the configuration file

import yaml
from pathlib import Path

# define the function to load the paths 
def load_config(config_path):
    '''
    Loads the path to the config files
    
    Params:
        config_path: The path to the config.yaml file 
    Return: 
        Loads the config.yml file data
    '''
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config