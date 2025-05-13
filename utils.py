import os
import numpy as np
import yaml
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

from network import NCHL
from optimizer import MapElites

# -
# load config functions

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
    
    nchl = NCHL(nodes=config["nodes"])
    config["seed"] = seed
    config["length"] = nchl.nparams
    config["path_dir"] = f"exp/{config['task']}_{seed}"
    
    return config

# - 
# save log functions
def save_log(logs, path_dir):
    """
    Save the logs to a file.
    """
    with open(os.path.join(path_dir, "log.txt"), 'w') as f:
        for log in logs:
            f.write(log + "\n")

# -
# plotting functions

def visualize_archive(map_elites, cmap="viridis", annot=True, high=False, path_dir=None):
    """
    Visualize the archive using heatmap with actual descriptor values on axes.
    """
    if map_elites.empty():
        print("Archive is empty, nothing to visualize.")
        return
    
    # Create a 2D array to store the fitness values
    fitness_map = np.full(map_elites.map_size, np.nan)
    
    pos_solved = []

    for pos, (individual, fitness, descriptor) in map_elites.archive.items():
        fitness_map[pos] = fitness
        # Check if the fitness is greater than the threshold
        if high and fitness >= map_elites.threshold:
            pos_solved.append(pos)

    # Generate tick labels based on the bounds and map size
    x_min, x_max = map_elites.bounds[0]
    y_min, y_max = map_elites.bounds[1]
    
    # Create tick labels for x and y axes
    x_ticks = np.linspace(x_min, x_max, map_elites.map_size[0])
    y_ticks = np.linspace(y_min, y_max, map_elites.map_size[1])
    
    # Format the tick labels to show 2 decimal places
    x_ticklabels = [f"{val:.2f}" for val in x_ticks]
    y_ticklabels = [f"{val:.2f}" for val in y_ticks]

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(fitness_map, annot=annot, fmt=".2f", cmap=cmap, cbar=True,
                     xticklabels=x_ticklabels, yticklabels=y_ticklabels)
    
    # Highlight the maximum fitness position
    if high and pos_solved:
        for pos in pos_solved:
            ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, fill=False, edgecolor='red', lw=2))
        
    # Rotate x-axis labels if needed for better readability
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.title("Map-Elite Archive Fitness Values")
    plt.xlabel("std mean neuron activations") 
    plt.ylabel("std mean neuron weight changes")
    
    plt.tight_layout()
    path_dir = path_dir if path_dir else map_elites.path_dir
    plt.savefig(os.path.join(path_dir, "archive.png"), dpi=300, bbox_inches='tight')
    plt.close()
       
def plot_history(avg_fitnesses, best_fitnesses, path_dir):
    """
    Plot the history of fitness values.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(avg_fitnesses, label="avg fitness")
    plt.plot(best_fitnesses, label="best fitness")
    plt.legend()
    plt.title("Fitness over generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig(os.path.join(path_dir, "fitness.png"))
    plt.close()
    
    