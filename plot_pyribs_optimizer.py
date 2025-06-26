import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pickle
from sklearn.decomposition import PCA

from network import NCHL, Neuron
from pyribs import QDBase

import scienceplots
plt.style.use('science')

# Global font settings
mpl.rcParams['axes.titlesize'] = 25         # Title font size
mpl.rcParams['figure.titlesize'] = 25      # Figure title font size
mpl.rcParams['axes.labelsize'] = 15         # x, y, z label font size
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['legend.fontsize'] = 20       # Legend font size

# - 
# Global variables
descriptor_names = ["Activation Diversity", "Weight Change Diversity"]
optimizers_names = ["MAPElites", "CMAME", "CMAMAE"]
SAVE_IN_PDF = True  # Set to True to save plots in PDF format

def convert_optimizer_name(name):
    """
    Convert optimizer name to a more readable format.
    """
    if name == "MAPElites":
        return "MAP-Elites"
    elif name == "CMAME":
        return "CMA-ME"
    elif name == "CMAMAE":
        return "CMA-MAE"
    else:
        return name

# Utility function
def get_optimizer_paths(list_of_paths):
    optimizer_paths = {name: [] for name in optimizers_names}
    for path in list_of_paths:
        for name in optimizers_names:
            if name in path:
                optimizer_paths[name].append(path)
                break
            
    return optimizer_paths

# pyribs / GridArchive utility functions TODO: move these functions to pyribs.utils or similar module

def compute_archive_precision(archive):
    objectives = archive.data()['objective']
    if len(objectives) == 0:
        return 0.0
    max_fitness = np.max(objectives)
    rmse = np.sqrt(np.mean((objectives - max_fitness) ** 2))
    
    return rmse

def compute_qd_score(archive):
    # Compute the QD score as the sum of normalized fitness values.
    
    data = archive.data()
    objectives = data['objective']
    
    if len(objectives) == 0:
        return 0.0
    
    fitness_values = np.array(objectives)
    min_fitness = np.min(fitness_values)
    max_fitness = np.max(fitness_values)

    # Normalize fitness values to [0, 1] range
    normalized_fitness = (fitness_values - min_fitness) / (max_fitness - min_fitness)
    # QD score is the sum of all normalized fitness values
    qd_score = np.sum(normalized_fitness)
    return qd_score
    
# -
# Archive

def compute_avg_archive(list_of_paths):
    # Load first archive to get dimensions and bounds
    first_archive = pickle.load(open(list_of_paths[0], 'rb'))
    map_size = first_archive.dims
    bounds = []
    for i in range(len(map_size)):
        bounds.append((first_archive.lower_bounds[i], first_archive.upper_bounds[i]))
    
    # Initialize arrays to accumulate sums and counts
    total_objectives = np.zeros(map_size)
    counts = np.zeros(map_size)
    
    # Process each archive
    for path in list_of_paths:
        archive = pickle.load(open(path, 'rb'))
        data = archive.data()
        objectives = data['objective']
        indexes = data['index']
        
        # Convert linear indexes to grid coordinates and accumulate
        for i, idx in enumerate(indexes):
            # Convert linear index to 2D coordinates
            coords = np.unravel_index(idx, map_size)
        
            total_objectives[coords] += objectives[i]
            counts[coords] += 1
            
    average_objectives = np.divide(total_objectives, counts, 
                           out=np.full_like(total_objectives, np.nan), 
                           where=counts!=0)
    
    return average_objectives, bounds

def visualize_all_archives_per_optimizer(list_of_paths, output_dir, cmap="viridis"):
    # for each optimizer, visualize all archives in a single plot.
    # at end there will be one plot saved for each optimizer.
    optimizer_paths = get_optimizer_paths(list_of_paths)
    for optimizer_name in optimizers_names:
        if optimizer_name not in optimizer_paths:
            print(f"No archives found for optimizer {optimizer_name}. Skipping.")
            continue
        
        # Load all archives for this optimizer
        archives = [pickle.load(open(path, 'rb')) for path in optimizer_paths[optimizer_name]]
        
        # Create a figure with subplots for each archive
        n_archives = len(archives)
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten() if n_archives > 1 else [axes]
        
        if n_archives == 1:
            axes = [axes]
        # Compute global min and max for color normalization
        global_min = float('inf')
        global_max = float('-inf')
        for archive in archives:
            objectives = archive.data()['objective']
            valid_values = objectives[~np.isnan(objectives)]
            if len(valid_values) > 0:
                global_min = min(global_min, np.min(valid_values))
                global_max = max(global_max, np.max(valid_values))
        # Create heatmaps for each archive
        for ax, archive in zip(axes, archives):
            data = archive.data()
            objectives = data['objective']
            indexes = data['index']
            
            # Create a 2D array to store the fitness values
            fitness_map = np.full(archive.dims, np.nan)
            
            # Fill the fitness map with values
            for i, idx in enumerate(indexes):
                coords = np.unravel_index(idx, archive.dims)
                fitness_map[coords] = objectives[i]
            
            # Create heatmap with normalized color scale
            cax = ax.imshow(fitness_map, cmap=cmap, aspect='auto', 
                            extent=[archive.lower_bounds[1], archive.upper_bounds[1], 
                                    archive.lower_bounds[0], archive.upper_bounds[0]],
                            vmin=global_min, vmax=global_max,
                            origin='lower')
            
            # ax.set_title(f'{convert_optimizer_name(optimizer_name)} Archive')
            ax.set_xlabel(descriptor_names[1], fontsize=15)
            ax.set_ylabel(descriptor_names[0], fontsize=15)
            
            # Add colorbar
            fig.colorbar(cax, ax=ax, orientation='vertical')
            
        # Add title for the entire figure
        fig.suptitle(f'{convert_optimizer_name(optimizer_name)} Archives')
        # Save the figure
        extension = 'pdf' if SAVE_IN_PDF else 'png'
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{convert_optimizer_name(optimizer_name)}_archives.{extension}"), dpi=300, bbox_inches='tight')
    

def visualize_avg_archives(list_of_paths, output_dir, cmap="viridis"):
    # For each optimizer, compute the average archive across all runs 
    # Create a unique plot with each optimizer's average archive subplot side by side.
    
    optimizer_paths = get_optimizer_paths(list_of_paths)
    
    # First pass: compute all average archives and find global min/max
    avg_archives = {}
    global_min = float('inf')
    global_max = float('-inf')
    
    for optimizer_name in optimizers_names:
        avg_archive, bounds = compute_avg_archive(optimizer_paths[optimizer_name])
        avg_archives[optimizer_name] = (avg_archive, bounds)
        
        # Update global min/max (ignoring NaN values)
        valid_values = avg_archive[~np.isnan(avg_archive)]
        if len(valid_values) > 0:
            global_min = min(global_min, np.min(valid_values))
            global_max = max(global_max, np.max(valid_values))
    
    # Create a figure with subplots for each optimizer
    n_optimizers = len(optimizers_names)
    fig, axes = plt.subplots(1, n_optimizers, figsize=(5 * n_optimizers, 5))
    
    if n_optimizers == 1:
        axes = [axes]
    
    # Second pass: create plots with normalized colormap
    for ax, optimizer_name in zip(axes, optimizers_names):
        avg_archive, bounds = avg_archives[optimizer_name]
        
        # Create heatmap with normalized color scale
        cax = ax.imshow(avg_archive, cmap=cmap, aspect='auto', 
                        extent=[bounds[1][0], bounds[1][1], bounds[0][0], bounds[0][1]],
                        vmin=global_min, vmax=global_max,
                        origin='lower') 
        
        ax.set_title(f'{convert_optimizer_name(optimizer_name)}')
        ax.set_xlabel(descriptor_names[1])
        ax.set_ylabel(descriptor_names[0])
        
        # Add colorbar
        fig.colorbar(cax, ax=ax, orientation='vertical')
    
    extension = 'pdf' if SAVE_IN_PDF else 'png'
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"avg_archives.{extension}"), dpi=300, bbox_inches='tight')
    plt.close()

# -
# Fitness history 

def compute_fitness_history(logs_paths):
    """
    Compute the fitness history from the log files.
    Each log file should contain lines starting with "iteration" followed by best and average fitness.
    Returns a tuple of numpy arrays: (best_fitnesses, avg_fitnesses, cum_best_fitnesses)
    """
    all_best_fitnesses = []
    all_avg_fitnesses = []
    all_cum_best_fitnesses = []

    for path in logs_paths:
        with open(path, 'r') as f:
            lines = f.readlines()
            best_fitnesses = []
            avg_fitnesses = []
            for line in lines:
                if line.startswith("iteration"):
                    parts = line.split()
                    best_fitness = float(parts[2])
                    avg_fitness = float(parts[3])

                    best_fitnesses.append(best_fitness)
                    avg_fitnesses.append(avg_fitness)

            all_best_fitnesses.append(best_fitnesses)
            all_avg_fitnesses.append(avg_fitnesses)

            # Compute cumulative best for this run
            cum_best = np.maximum.accumulate(best_fitnesses)  # for maximization
            # cum_best = np.minimum.accumulate(best_fitnesses)  # if minimizing
            all_cum_best_fitnesses.append(cum_best)

    return (np.array(all_best_fitnesses), np.array(all_avg_fitnesses), np.array(all_cum_best_fitnesses))

def plot_fitness_history_compare(logs_paths, path_dir, complete=False, only_best=False, threshold=None):
    optimizer_paths = get_optimizer_paths(logs_paths)
            
    # Create a unique plot
    plt.figure(figsize=(10, 6))
    all_best_fitnesses = []
    all_avg_fitnesses = []
    all_cum_best_fitnesses = []
    
    len_x = 0
    
    for optimizer_name in optimizers_names:
        # Compute fitness history for this optimizer
        best_fitnesses, avg_fitnesses, cum_best_fitnesses = compute_fitness_history(optimizer_paths[optimizer_name])
        
        all_best_fitnesses.append(best_fitnesses)
        all_avg_fitnesses.append(avg_fitnesses)
        all_cum_best_fitnesses.append(cum_best_fitnesses)
        
        if not only_best:
            # Plot cumulative global best
            avg_cum_best = np.mean(cum_best_fitnesses, axis=0)
            std_cum_best = np.std(cum_best_fitnesses, axis=0)
            plt.plot(avg_cum_best, label=f"{convert_optimizer_name(optimizer_name)}", linewidth=2)
            plt.fill_between(range(len(avg_cum_best)), avg_cum_best - std_cum_best,
                            avg_cum_best + std_cum_best, alpha=0.2)
            
            len_x = max(len_x, len(avg_cum_best))

        if complete or only_best: 
            # Plot the best fitness
            avg_best_fitnesses = np.mean(best_fitnesses, axis=0)
            std_best_fitnesses = np.std(best_fitnesses, axis=0)
            plt.plot(avg_best_fitnesses, label=f"{convert_optimizer_name(optimizer_name)}", linewidth=2)
            plt.fill_between(range(len(avg_best_fitnesses)), avg_best_fitnesses - std_best_fitnesses,
                            avg_best_fitnesses + std_best_fitnesses, alpha=0.2)
            
            len_x = max(len_x, len(avg_best_fitnesses))

        if complete and not only_best:
            # Plot the average fitness
            avg_avg_fitnesses = np.mean(avg_fitnesses, axis=0)
            std_avg_fitnesses = np.std(avg_fitnesses, axis=0)
            plt.plot(avg_avg_fitnesses, label=f"{convert_optimizer_name(optimizer_name)}", linewidth=2)
            plt.fill_between(range(len(avg_avg_fitnesses)), avg_avg_fitnesses - std_avg_fitnesses,
                             avg_avg_fitnesses + std_avg_fitnesses, alpha=0.2)
            
            len_x = max(len_x, len(avg_avg_fitnesses))
        
    if threshold is not None:
        plt.hlines(y=threshold, xmin=0, xmax=len_x-1,
                     color='r', linestyle='--', label="Threshold", linewidth=2)
    # plt.title('Fitness History Comparison')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc='lower right')
    plt.tight_layout()
    appendix = "_complete" if complete else ""
    appendix += "_only_best" if only_best else ""
    extension = 'pdf' if SAVE_IN_PDF else 'png'
    plt.savefig(os.path.join(path_dir, f"fitness_history_compare{appendix}.{extension}"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fitness_history(logs_paths, path_dir, complete=False, threshold=None):
    # For each optimizer, read the logs and plot the fitness history.
    
    optimizer_paths = get_optimizer_paths(logs_paths)
    
    # Create a figure with subplots for each optimizer
    n_optimizers = len(optimizers_names)
    fig, axes = plt.subplots(1, n_optimizers, figsize=(5 * n_optimizers, 5))
    if n_optimizers == 1:
        axes = [axes]
    
    all_best_fitnesses = []
    all_avg_fitnesses = []
    all_cum_best_fitnesses = []
    
    len_x = 0
    
    # collect all data
    for optimizer_name in optimizers_names:
        # Compute fitness history for this optimizer
        best_fitnesses, avg_fitnesses, cum_best_fitnesses = compute_fitness_history(optimizer_paths[optimizer_name])
        
        all_best_fitnesses.append(best_fitnesses)
        all_avg_fitnesses.append(avg_fitnesses)
        all_cum_best_fitnesses.append(cum_best_fitnesses)
    
    # Compute global min and max across all optimizers
    all_values = []
    
    for i, (best_fitnesses, avg_fitnesses, cum_best_fitnesses) in enumerate(zip(all_best_fitnesses, all_avg_fitnesses, all_cum_best_fitnesses)):
        # Collect cumulative best values with their confidence intervals
        cum_best_values = np.mean(cum_best_fitnesses, axis=0)
        cum_best_std = np.std(cum_best_fitnesses, axis=0)
        
        all_values.extend(cum_best_values - cum_best_std)
        all_values.extend(cum_best_values + cum_best_std)
        all_values.extend(cum_best_values)
        
        if complete:
            # Collect best fitness values with their confidence intervals
            best_values = np.mean(best_fitnesses, axis=0)
            best_std = np.std(best_fitnesses, axis=0)
            
            all_values.extend(best_values - best_std)
            all_values.extend(best_values + best_std)
            all_values.extend(best_values)
            
        len_x = max(len_x, len(cum_best_values))
    
    # Compute true global min and max
    global_min = np.min(all_values)
    global_max = np.max(all_values)
    # Add some padding to the limits (5% of the range)
    range_padding = (global_max - global_min) * 0.05
    global_min -= range_padding
    global_max += range_padding
    
    # create plots
    for ax, optimizer_name, best_fitnesses, avg_fitnesses, cum_best_fitnesses in zip(axes, optimizers_names, all_best_fitnesses, all_avg_fitnesses, all_cum_best_fitnesses):
        
        # Plot cumulative global best
        avg_cum_best = np.mean(cum_best_fitnesses, axis=0)
        std_cum_best = np.std(cum_best_fitnesses, axis=0)
        ax.plot(avg_cum_best, label="Avg Global Best", linewidth=2)
        ax.fill_between(range(len(avg_cum_best)), avg_cum_best - std_cum_best,
                        avg_cum_best + std_cum_best, alpha=0.2)

        if complete: 
            # Plot the best fitness
            avg_best_fitnesses = np.mean(best_fitnesses, axis=0)
            std_best_fitnesses = np.std(best_fitnesses, axis=0)
            ax.plot(avg_best_fitnesses, label="Avg Best Fitness", linewidth=2)
            ax.fill_between(range(len(avg_best_fitnesses)), avg_best_fitnesses - std_best_fitnesses,
                            avg_best_fitnesses + std_best_fitnesses, alpha=0.2)
        
        if threshold is not None:
            ax.hlines(y=threshold, xmin=0, xmax=len_x-1,
                     color='r', linestyle='--', label="Threshold", linewidth=2)

        ax.set_title(f'{convert_optimizer_name(optimizer_name)}')
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.legend(loc='lower right')
        
        # Set global y-axis limits
        ax.set_ylim(global_min, global_max)
        
    plt.tight_layout()
    extension = 'pdf' if SAVE_IN_PDF else 'png'
    plt.savefig(os.path.join(path_dir, f"fitness_history.{extension}"), dpi=300, bbox_inches='tight')
    plt.close()
# - 
# Hebbian parameters
    
def compute_hebbian_distribution_per_params(list_of_paths, nodes, only_hidden=False): 
    rows = [] 
    for net_id, path in enumerate(list_of_paths): 
        archive = pickle.load(open(path, 'rb')) 
        best_elite = archive.best_elite 
        net = NCHL(nodes=nodes) 
        net.set_params(best_elite['solution']) 
         
        for layer_idx, layer in enumerate(net.neurons): 
            if only_hidden and (layer_idx == 0 or layer_idx == len(net.neurons) - 1): # skip input and output layers 
                continue 
            for neuron in layer: 
                rule = neuron.get_rule() 
                rule_values = [float(p.item()) if hasattr(p, 'item') else float(p) for p in rule] 
                row = { 
                    'Network': net_id, 
                    'Neuron': neuron.neuron_id, 
                    'pre-synaptic (A)': rule_values[0], 
                    'post-synaptic (B)': rule_values[1], 
                    'correlation (C)': rule_values[2], 
                    'decorrelation (D)': rule_values[3], 
                    'learning rate (eta)': rule_values[4], 
                } 
                rows.append(row) 
    df = pd.DataFrame(rows) 
    return df 
 
def plot_hebbian_distribution_per_params(list_of_paths, path_dir, nodes, only_hidden=False): 
    """ 
    Visualize the distribution of Hebbian learning parameters across networks. 
    Each parameter (A, B, C, D, eta) is displayed in its own row, with consistent color per parameter. 
    """ 
    # Get optimizer paths 
    optimizer_paths = get_optimizer_paths(list_of_paths) 
 
    # Parameter names 
    param_names = ['pre-synaptic (A)', 'post-synaptic (B)', 'correlation (C)', 'decorrelation (D)', 'learning rate (eta)'] 
     
    # Step 1: Collect all DataFrames with optimizer labels
    all_dfs = [] 
    for optimizer_name in optimizer_paths.keys():  # Use actual optimizer names from paths
        paths = optimizer_paths[optimizer_name] 
        df_long = compute_hebbian_distribution_per_params(paths, nodes, only_hidden=only_hidden) 
        df_long['Optimizer'] = optimizer_name  # Add optimizer column for identification
        all_dfs.append(df_long)
         
    combined_df = pd.concat(all_dfs, ignore_index=True) 
    
    # Create file with statistics for each parameter and optimizer
    stats_file = os.path.join(path_dir, "hebbian_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("Hebbian Parameters Statistics\n")
        f.write("================================\n")
        for param in param_names:
            f.write(f"\nParameter: {param}\n")
            for optimizer_name in optimizer_paths.keys():
                df_param = combined_df[combined_df['Optimizer'] == optimizer_name]
                if not df_param.empty:
                    mean_value = df_param[param].mean()
                    std_value = df_param[param].std()
                    f.write(f"{convert_optimizer_name(optimizer_name)}: Mean={mean_value:.4f}, Std={std_value:.4f}\n")
    
    # Step 2: Create subplots : one subplot per parameter, per optimizer 
    n_params = len(param_names) 
    optimizers_list = list(optimizer_paths.keys())  # Get actual optimizer names
    n_optimizers = len(optimizers_list) 
    fig, axes = plt.subplots(n_params, n_optimizers, figsize=(5 * n_optimizers, 4 * n_params), sharey=True) 
    
    # Handle single optimizer case
    if n_optimizers == 1: 
        axes = axes.reshape(-1, 1)  # Ensure 2D array structure
    elif n_params == 1:
        axes = axes.reshape(1, -1)  # Handle single parameter case
        
    palette = sns.color_palette("viridis", len(param_names))  
    param_colors = dict(zip(param_names, palette))
    
    # Step 3: Plot each parameter in its own row 
    for param_idx, param_name in enumerate(param_names): 
        color = param_colors[param_name]
        for opt_idx, optimizer_name in enumerate(optimizers_list): 
            ax = axes[param_idx, opt_idx] if n_optimizers > 1 else axes[param_idx, 0]
            
            # Filter data for current optimizer
            df_optimizer = combined_df[combined_df['Optimizer'] == optimizer_name]
            
            if not df_optimizer.empty:
                sns.boxplot(x='Network', y=param_name, data=df_optimizer, ax=ax, color=color, width=0.9) 
 
            if param_idx == 0:  # First row - show optimizer names
                ax.set_title(f'{convert_optimizer_name(optimizer_name)}')
            else:
                ax.set_title('')  # No title for other rows
        
            ax.set_ylabel(param_name, color=color)            
            ax.set_xlabel('Network') 
            ax.set_ylabel(param_name) 
            ax.grid(True, linestyle='--', alpha=0.7) 
            
            # Remove individual legends as we'll create a master legend
            if ax.get_legend():
                ax.get_legend().remove() 
    
    # # Add master title and layout
    # fig.suptitle('Hebbian Learning Rule Parameter Distribution Across Networks', fontsize=18) 
    # fig.subplots_adjust(top=0.95) 
    
    # Save the plot
    extension = 'pdf' if SAVE_IN_PDF else 'png'
    plt.savefig(os.path.join(path_dir, f"hebbian_distribution_per_params{'_hidden' if only_hidden else ''}.{extension}"), 
                dpi=300, bbox_inches='tight') 
    plt.close()    
    
def compute_hebbian_distribution_per_nets(list_of_paths, nodes, only_hidden=False):
    rows = []
    for net_id, path in enumerate(list_of_paths):
        archive = pickle.load(open(path, 'rb'))
        best_elite = archive.best_elite
        net = NCHL(nodes=nodes)
        net.set_params(best_elite['solution'])
        
        for layer_idx, layer in enumerate(net.neurons):
            for neuron in layer:
                if only_hidden and (layer_idx == 0 or layer_idx == len(net.neurons) - 1):
                    continue
                rule = neuron.get_rule()
                rule_values = [float(p.item()) if hasattr(p, 'item') else float(p) for p in rule]
                row = {
                    'Network': f'Net {net_id}',
                    'Neuron': neuron.neuron_id,
                    'pre-synaptic (A)': rule_values[0],
                    'post-synaptic (B)': rule_values[1],
                    'correlation (C)': rule_values[2],
                    'decorrelation (D)': rule_values[3],
                    'learning rate (eta)': rule_values[4],
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    # Convert to long-form DataFrame for seaborn boxplot
    df_long = df.melt(id_vars=['Network', 'Neuron'], 
                      value_vars=[
                          'pre-synaptic (A)', 
                          'post-synaptic (B)', 
                          'correlation (C)', 
                          'decorrelation (D)', 
                          'learning rate (eta)'
                      ],
                      var_name='Parameter', 
                      value_name='Value')
    
    return df_long

def plot_hebbian_distribution_per_nets(list_of_paths, path_dir, nodes, only_hidden=False):
    # Get optimizer paths
    optimizer_paths = get_optimizer_paths(list_of_paths)

    # Step 1: Collect all DataFrames and compute global min/max
    all_dfs = []
    for optimizer_name in optimizers_names:
        paths = optimizer_paths[optimizer_name]
        df_long = compute_hebbian_distribution_per_nets(paths, nodes, only_hidden=only_hidden)
        all_dfs.append(df_long)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    thread = 0.1  # Set a threshold for min/max values
    global_min = combined_df['Value'].min() - thread
    global_max = combined_df['Value'].max() + thread

    # Step 2: Create subplots
    n_optimizers = len(optimizers_names)
    fig, axes = plt.subplots(1, n_optimizers, figsize=(5 * n_optimizers, 6))
    if n_optimizers == 1:
        axes = [axes]

    legend_handles = None

    # Step 3: Plot each with shared y-axis limits
    for ax, optimizer_name in zip(axes, optimizers_names):
        df_long = compute_hebbian_distribution_per_nets(optimizer_paths[optimizer_name], nodes, only_hidden=only_hidden)

        sns.boxplot(x='Network', y='Value', hue='Parameter', data=df_long, ax=ax, palette='viridis', width=0.9)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        ax.set_title(f'{convert_optimizer_name(optimizer_name)}')
        ax.set_xlabel('Network')
        ax.set_ylabel('Value')
        ax.set_ylim(global_min, global_max)  # Apply shared y-axis
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.get_legend().remove()

    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(legend_labels), fontsize='medium')
    fig.suptitle('Hebbian Learning Rule Parameter Distribution Across Networks', fontsize=18)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    extension = 'pdf' if SAVE_IN_PDF else 'png'
    plt.savefig(os.path.join(path_dir, f"hebbian_distribution_per_nets{'_hidden' if only_hidden else ''}.{extension}"),
                dpi=300, bbox_inches='tight')
    plt.close()



# -
# Rules PCA Analysis

def get_layer_label(idx, total):
    if idx == 0:
        return "Input"
    elif idx == total - 1:
        return "Output"
    else:
        return "Hidden"

def compute_pca_rules(list_of_paths, nodes, only_hidden=False):
    all_rules = []
    net_ids = []
    layer_ids = []

    max_layer_count = 0  # to determine layer roles (input, hidden, output)

    for net_id, path in enumerate(list_of_paths):
        archive = pickle.load(open(path, 'rb'))
        best_elite = archive.best_elite
        net = NCHL(nodes=nodes)
        net.set_params(best_elite['solution'])
        
        max_layer_count = max(max_layer_count, len(net.neurons))
        for layer_idx, layer in enumerate(net.neurons):
            for neuron in layer:
                if only_hidden and (layer_idx == 0 or layer_idx == len(net.neurons) - 1):
                    continue
                rule = neuron.get_rule()
                rule_values = [float(p.item()) if hasattr(p, 'item') else float(p) for p in rule]
                all_rules.append(rule_values)
                net_ids.append(net_id)
                layer_ids.append(layer_idx)

    all_rules = np.array(all_rules)
    net_ids = np.array(net_ids)
    layer_ids = np.array(layer_ids)
    
    return all_rules, net_ids, layer_ids, max_layer_count

def plot_combined_pca_rules(list_of_paths, path_dir, nodes, only_hidden=False):
    """
    Plot PCA of rules with colors based on optimizer.
    """
    optimizer_paths = get_optimizer_paths(list_of_paths)
    
    # for each optimizer, compute PCA of rules
    all_rules = []
    net_ids = []
    layer_ids = []
    max_layer_count = 0  # to determine layer roles (input, hidden, output)
    for optimizer_name in optimizers_names:
        paths = optimizer_paths[optimizer_name]
        rules, nets, layers, max_layer_count = compute_pca_rules(paths, nodes, only_hidden=only_hidden)
        all_rules.append(rules)
        net_ids.append(nets)
        layer_ids.append(layers)
        
    all_rules = np.vstack(all_rules)
    net_ids = np.hstack(net_ids)
    layer_ids = np.hstack(layer_ids)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_rules)
    
    # Plot setup : color by optimizer, marker by layer type
    plt.figure(figsize=(12, 9))
    unique_nets = np.unique(net_ids)
    unique_layers = np.unique(layer_ids)
    layer_types = {}
    for layer_idx in unique_layers:
        layer_types[layer_idx] = get_layer_label(layer_idx, max_layer_count)
        
    cmap = plt.cm.viridis(np.linspace(0, 1, len(optimizers_names)))
    optimizer_colors = {name: cmap[i] for i, name in enumerate(optimizers_names)}
    
    # Plot data points
    for net_id in unique_nets:
        for layer_idx in unique_layers:
            mask = (net_ids == net_id) & (layer_ids == layer_idx)
            if not np.any(mask):
                continue
            
            # Get optimizer name and layer type
            optimizer_name = optimizers_names[net_id % len(optimizers_names)]
            layer_type = layer_types[layer_idx]
            
            # Plot points colored by optimizer and marker by layer type
            plt.scatter(
                pca_result[mask, 0],
                pca_result[mask, 1],
                color=optimizer_colors[optimizer_name],
                # marker='o' if layer_type == "Input" else ('^' if layer_type == "Hidden" else 's'),
                label=f'{convert_optimizer_name(optimizer_name)} - {layer_type}',
                alpha=0.7,
                s=300,
                linewidths=0.7
            )
    # Create optimizer and layer type legend
    optimizer_handles = []
    for optimizer_name, color in optimizer_colors.items():
        optimizer_handles.append(
            Line2D([0], [0], marker='o', color='w', 
                   label=f'{convert_optimizer_name(optimizer_name)}',
                   markerfacecolor=color, markersize=18, alpha=0.7, linewidth=0.7)
        )
    # layer_type_handles = []
    # for layer_type, color in layer_colors.items():
    #     layer_type_handles.append(
    #         Line2D([0], [0], marker='o', color='w', 
    #                label=f'{layer_type}',
    #                markerfacecolor=color, markersize=10)
    #     )
    
    # Add the legends for optimizers in horizontal layout and located at the lower center
    plt.legend(handles=optimizer_handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(optimizer_handles))

    
    # Plot formatting
    plt.xlabel(f"PCA 1") # - {pca.explained_variance_ratio_[0]:.2f}")
    plt.ylabel(f"PCA 2") # - {pca.explained_variance_ratio_[1]:.2f}")
    # plt.title("PCA of Hebbian Rules by Optimizer")
    plt.tight_layout()
    suffix = "_hidden" if only_hidden else ""
    extension = 'pdf' if SAVE_IN_PDF else 'png'
    plt.savefig(os.path.join(path_dir, f"pca_combined_rules{suffix}.{extension}"), dpi=300, bbox_inches='tight')
    plt.close()
   
# - 
# Descriptors

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def compute_descriptors_by_net(list_of_paths, k=5):
    all_descriptors = []
    for i, path in enumerate(list_of_paths):
        archive = pickle.load(open(path, 'rb'))
        data = archive.data()
        objectives = data['objective']
        best_indices = sorted(range(len(objectives)), key=lambda i: objectives[i], reverse=True)[:k] # get top k indices
        for idx in best_indices:
            measures = data['measures'][idx]
            x, y = measures[1], measures[0]
            all_descriptors.append((x, y))
            
    return all_descriptors

def plot_descriptors(list_of_paths, path_dir, k=5):
    """
    Plot the 2 descriptors in 2D space, one subplot for each optimizer, colored by network
    """
    
    optimizer_paths = get_optimizer_paths(list_of_paths)
    
    # Create a figure with subplots for each optimizer
    n_optimizers = len(optimizers_names)
    fig, axes = plt.subplots(1, n_optimizers, figsize=(5 * n_optimizers, 5))
    if n_optimizers == 1:
        axes = [axes]
    
    colors = sns.color_palette("viridis", n_optimizers)  # Use a color palette for different optimizers
    for ax, optimizer_name in zip(axes, optimizers_names):
        # Compute descriptors for this optimizer
        all_descriptors = compute_descriptors_by_net(optimizer_paths[optimizer_name], k=k)
        
        # Convert to numpy array for easier plotting
        all_descriptors = np.array(all_descriptors)
        
        # Plot the descriptors
        ax.scatter(all_descriptors[:, 0], all_descriptors[:, 1], 
                   color=colors[optimizers_names.index(optimizer_name)], 
                   label=optimizer_name, alpha=0.7, s=150)
        
        ax.set_title(f'{convert_optimizer_name(optimizer_name)}')
        ax.set_xlabel(descriptor_names[1])
        ax.set_ylabel(descriptor_names[0])
        # ax.legend()
        threshold = 0.1  # Example threshold for x and y limits
        ax.set_xlim(0 - threshold, 1 + threshold)
        ax.set_ylim(0 - threshold, 1 + threshold)
        
        ax.tick_params(axis='x', pad=5)
        ax.tick_params(axis='y', pad=5)
        
        # set tick digits
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        
    # plt.suptitle(f"Best {k} Descriptors by Optimizer", fontsize=16)
    plt.tight_layout()
    extension = 'pdf' if SAVE_IN_PDF else 'png'
    plt.savefig(os.path.join(path_dir, f"descriptors.{extension}"), dpi=300, bbox_inches='tight')
    plt.close()

# -
# Archive stats
def compute_archives_stats_complete(optimizer_paths):
    stats = {}
    for optimizer_name in optimizers_names:
        stats[optimizer_name] = {
            'qd_score': [],
            'coverage': [],
            'rmse': [],
            'qd_score_std': [],
            'coverage_std': [],
            'rmse_std': []
        }
        
        for path in optimizer_paths[optimizer_name]:
            archive = pickle.load(open(path, 'rb'))
            stats[optimizer_name]['qd_score'].append(compute_qd_score(archive))
            stats[optimizer_name]['coverage'].append(archive.stats.coverage)
            stats[optimizer_name]['rmse'].append(compute_archive_precision(archive))
        
        # Compute mean and std for each metric
        qd_scores = stats[optimizer_name]['qd_score']
        coverage_vals = stats[optimizer_name]['coverage']
        rmse_vals = stats[optimizer_name]['rmse']

        stats[optimizer_name]['qd_score'] = np.mean(qd_scores)
        stats[optimizer_name]['qd_score_std'] = np.std(qd_scores)

        stats[optimizer_name]['coverage'] = np.mean(coverage_vals)
        stats[optimizer_name]['coverage_std'] = np.std(coverage_vals)

        stats[optimizer_name]['rmse'] = np.mean(rmse_vals)
        stats[optimizer_name]['rmse_std'] = np.std(rmse_vals)
        
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats).T
    stats_df.columns = ['QD Score', 'Coverage', 'RMSE', 
                        'QD Score Std', 'Coverage Std', 'RMSE Std']
    stats_df.reset_index(inplace=True)
    stats_df.rename(columns={'index': 'Optimizer'}, inplace=True)
    
    return stats_df

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from itertools import combinations
import pickle

def compute_archives_stats_complete_new(list_of_paths):
    stats = {}
    raw_data = {}  # Store raw values for statistical tests
    
    optimizer_paths = get_optimizer_paths(list_of_paths)
    
    for optimizer_name in optimizers_names:
        stats[optimizer_name] = {
            'qd_score': [],
            'coverage': [],
            'rmse': [],
            'qd_score_std': [],
            'coverage_std': [],
            'rmse_std': []
        }
        
        # Store raw values for each metric
        raw_data[optimizer_name] = {
            'qd_score': [],
            'coverage': [],
            'rmse': []
        }
        
        for path in optimizer_paths[optimizer_name]:
            archive = pickle.load(open(path, 'rb'))
            qd_score = compute_qd_score(archive)
            coverage = archive.stats.coverage
            rmse = compute_archive_precision(archive)
            
            # Store for statistics
            stats[optimizer_name]['qd_score'].append(qd_score)
            stats[optimizer_name]['coverage'].append(coverage)
            stats[optimizer_name]['rmse'].append(rmse)
            
            # Store raw data for tests
            raw_data[optimizer_name]['qd_score'].append(qd_score)
            raw_data[optimizer_name]['coverage'].append(coverage)
            raw_data[optimizer_name]['rmse'].append(rmse)
        
        # Compute mean and std for each metric
        qd_scores = stats[optimizer_name]['qd_score']
        coverage_vals = stats[optimizer_name]['coverage']
        rmse_vals = stats[optimizer_name]['rmse']

        stats[optimizer_name]['qd_score'] = np.mean(qd_scores)
        stats[optimizer_name]['qd_score_std'] = np.std(qd_scores)

        stats[optimizer_name]['coverage'] = np.mean(coverage_vals)
        stats[optimizer_name]['coverage_std'] = np.std(coverage_vals)

        stats[optimizer_name]['rmse'] = np.mean(rmse_vals)
        stats[optimizer_name]['rmse_std'] = np.std(rmse_vals)
        
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats).T
    stats_df.columns = ['QD Score', 'Coverage', 'RMSE', 
                        'QD Score Std', 'Coverage Std', 'RMSE Std']
    stats_df.reset_index(inplace=True)
    stats_df.rename(columns={'index': 'Optimizer'}, inplace=True)
    
    return stats_df, raw_data

def perform_wilcoxon_tests(raw_data, alpha=0.05):
    """
    Perform pairwise Wilcoxon signed-rank tests between optimizers for each metric.
    
    Parameters:
    raw_data: dict with optimizer names as keys and metric arrays as values
    alpha: significance level (default 0.05)
    
    Returns:
    Dictionary with test results for each metric
    """
    optimizers = list(raw_data.keys())
    metrics = ['qd_score', 'coverage', 'rmse']
    
    results = {}
    
    for metric in metrics:
        results[metric] = {}
        results[metric]['comparisons'] = []
        results[metric]['p_values'] = []
        results[metric]['statistics'] = []
        results[metric]['significant'] = []
        results[metric]['better_optimizer'] = []
        
        # Get all pairwise combinations of optimizers
        for opt1, opt2 in combinations(optimizers, 2):
            data1 = np.array(raw_data[opt1][metric])
            data2 = np.array(raw_data[opt2][metric])
            
            # Check if we have enough data points
            if len(data1) < 2 or len(data2) < 2:
                print(f"Warning: Not enough data points for {opt1} vs {opt2} on {metric}")
                continue
            
            # Ensure same number of samples (paired test)
            min_samples = min(len(data1), len(data2))
            data1 = data1[:min_samples]
            data2 = data2[:min_samples]
            
            # Perform Wilcoxon signed-rank test
            try:
                statistic, p_value = wilcoxon(data1, data2, alternative='two-sided')
                
                # Determine which optimizer is better based on means
                mean1 = np.mean(data1)
                mean2 = np.mean(data2)
                
                # For QD Score and Coverage: higher is better
                # For RMSE: lower is better
                if metric in ['qd_score', 'coverage']:
                    better_opt = opt1 if mean1 > mean2 else opt2
                else:  # rmse
                    better_opt = opt1 if mean1 < mean2 else opt2
                
                # Store results
                results[metric]['comparisons'].append(f"{opt1} vs {opt2}")
                results[metric]['p_values'].append(p_value)
                results[metric]['statistics'].append(statistic)
                results[metric]['significant'].append(p_value < alpha)
                results[metric]['better_optimizer'].append(better_opt)
                
            except ValueError as e:
                print(f"Error in Wilcoxon test for {opt1} vs {opt2} on {metric}: {e}")
                continue
    
    return results

def print_wilcoxon_results(results, alpha=0.05):
    """
    Print formatted results of Wilcoxon tests
    """
    for metric in results.keys():
        print(f"\n{'='*50}")
        print(f"WILCOXON TEST RESULTS FOR {metric.upper()}")
        print(f"{'='*50}")
        
        if not results[metric]['comparisons']:
            print("No valid comparisons found.")
            continue
            
        for i, comparison in enumerate(results[metric]['comparisons']):
            p_val = results[metric]['p_values'][i]
            statistic = results[metric]['statistics'][i]
            significant = results[metric]['significant'][i]
            better_opt = results[metric]['better_optimizer'][i]
            
            print(f"\nComparison: {comparison}")
            print(f"Statistic: {statistic:.4f}")
            print(f"P-value: {p_val:.6f}")
            print(f"Significant (α={alpha}): {'Yes' if significant else 'No'}")
            
            if significant:
                print(f"Better optimizer: {better_opt}")
                print("Conclusion: There IS a significant difference")
            else:
                print("Conclusion: No significant difference detected")
    
    print(f"\n{'='*50}")

def create_wilcoxon_summary_table(results):
    """
    Create a summary DataFrame of all Wilcoxon test results
    """
    summary_data = []
    
    for metric in results.keys():
        for i, comparison in enumerate(results[metric]['comparisons']):
            summary_data.append({
                'Metric': metric,
                'Comparison': comparison,
                'Statistic': results[metric]['statistics'][i],
                'P-value': results[metric]['p_values'][i],
                'Significant': results[metric]['significant'][i],
                'Better_Optimizer': results[metric]['better_optimizer'][i]
            })
    
    return pd.DataFrame(summary_data)


def plot_archives_stats_columns(list_of_paths, path_dir):
    # for each optimizer, compute the average archive stats 
    # plot  qd score, coverage, rmse 
    # mean + std for each optimizer
    
    optimizer_paths = get_optimizer_paths(list_of_paths)

    stats_df = compute_archives_stats_complete(optimizer_paths)
    print("Archive Statistics:")
    print(stats_df)
    
    # Create a bar plot for each metric
    metrics = ['QD Score', 'Coverage', 'RMSE']
    
    # Create one figure with subplots for each metric
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
        
    colors = sns.color_palette("viridis", len(optimizers_names))
    color_map = dict(zip(optimizers_names, colors))  # Map optimizer name to color

    for ax, metric in zip(axes, metrics):
        bar_colors = [color_map[opt] for opt in stats_df['Optimizer']]
        ax.bar(stats_df['Optimizer'], stats_df[metric],
               yerr=stats_df[f'{metric} Std'],
               capsize=5, color=bar_colors, edgecolor='black')
        
        ax.set_title(metric)
        ax.set_xlabel('Optimizer')
        ax.set_ylabel(metric)    

    plt.tight_layout()
    extension = 'pdf' if SAVE_IN_PDF else 'png'
    plt.savefig(os.path.join(path_dir, f"archive_stats.{extension}"), dpi=300, bbox_inches='tight')
    
def compute_archives_stats(optimizer_paths):
    stats = {}
    for optimizer_name in optimizers_names:
        stats[optimizer_name] = {
            'qd_score': [],
            'coverage': [],
            'rmse': []
        }
        
        for path in optimizer_paths[optimizer_name]:
            archive = pickle.load(open(path, 'rb'))
            stats[optimizer_name]['qd_score'].append(compute_qd_score(archive))
            stats[optimizer_name]['coverage'].append(archive.stats.coverage)
            stats[optimizer_name]['rmse'].append(compute_archive_precision(archive))
    
    return stats

# Version using seaborn for more styling options
def plot_archives_stats_seaborn(list_of_paths, path_dir):
    
    optimizer_paths = get_optimizer_paths(list_of_paths)
    stats = compute_archives_stats(optimizer_paths)
    
    # Convert to long format DataFrame for seaborn
    data_rows = []
    for optimizer_name in optimizers_names:
        for i, (qd_score, coverage, rmse) in enumerate(zip(
            stats[optimizer_name]['qd_score'],
            stats[optimizer_name]['coverage'], 
            stats[optimizer_name]['rmse']
        )):
            data_rows.append({
                'Optimizer': optimizer_name,
                'Run': i,
                'QD Score': qd_score,
                'Coverage': coverage,
                'RMSE': rmse
            })
    
    df_long = pd.DataFrame(data_rows)
    
    # Melt the DataFrame to have metric as a variable
    df_melted = pd.melt(df_long, 
                       id_vars=['Optimizer', 'Run'], 
                       value_vars=['QD Score', 'Coverage', 'RMSE'],
                       var_name='Metric', 
                       value_name='Value')
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['QD Score', 'Coverage', 'RMSE']
    colors = sns.color_palette("viridis", len(optimizers_names))
    color_map = dict(zip(optimizers_names, colors))  # Map optimizer name to color
    
    for ax, metric in zip(axes, metrics):        
        metric_data = df_melted[df_melted['Metric'] == metric]
        
        metric_data['Optimizers'] = metric_data['Optimizer'].apply(convert_optimizer_name)
        
        bar_colors = [color_map[opt] for opt in metric_data['Optimizer']]
        
        sns.boxplot(data=metric_data, x='Optimizers', y='Value', ax=ax, palette=color_map, hue='Optimizer', legend=False)
        
        ax.set_title(metric)
        ax.set_xlabel('')
        # ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
    # set limits for y-axis
    axes[0].set_ylim(0, df_long['QD Score'].max() * 1.1)
    axes[1].set_ylim(0, df_long['Coverage'].max() * 1.1)
    axes[2].set_ylim(0, df_long['RMSE'].max() * 1.1)
    
    plt.tight_layout()
    extension = 'pdf' if SAVE_IN_PDF else 'png'
    plt.savefig(os.path.join(path_dir, f"archive_stats_boxplot_seaborn.{extension}"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Write summary statistics
    summary_stats = df_long.groupby('Optimizer')[['QD Score', 'Coverage', 'RMSE']].agg(['mean', 'std', 'median'])
    with open(os.path.join(path_dir, 'archive_stats_summary.txt'), 'w') as f:
        f.write("Summary Statistics:\n")
        f.write(summary_stats.to_string())

#

# -

# -------------------------------------
# -
import ast
def main():
    if len(sys.argv) != 4:
        print("Usage: python plot.py <path_file> <path_dir>")
        print("path_file: .txt file with paths to archives")
        print("path_dir: directory to save plots")
        print("nodes: list of nodes in the network (e.g. [8, 8, 4])")
        sys.exit(1)
        
    path_file = sys.argv[1]     # .txt file with paths
    path_dir = sys.argv[2]      # directory to save plots
    nodes = eval(sys.argv[3])  # list of nodes in the network, e.g. [8, 8, 4]
    os.makedirs(path_dir, exist_ok=True)  
        
    # Load paths from the file
    archive_paths = []  # list of paths to archives
    logs_paths = []     # list of paths to logs (history)
    
    with open(path_file, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            archive_paths.append(line + '/archive.pkl')
            logs_paths.append(line + '/log.txt')
            
    print(archive_paths)
    print("-")
    print(logs_paths)
    
    
    # - WILCOXON TEST
    # # Run your original function with modification
    # stats_df, raw_data = compute_archives_stats_complete_new(archive_paths)

    # # Perform Wilcoxon tests
    # wilcoxon_results = perform_wilcoxon_tests(raw_data, alpha=0.05)

    # # Print results
    # print_wilcoxon_results(wilcoxon_results)

    # # Create summary table
    # summary_df = create_wilcoxon_summary_table(wilcoxon_results)
    # # save to .csv file
    # summary_df.to_csv(os.path.join(path_dir, 'wilcoxon_summary.csv'), index=False)
    # print("\nWILCOXON TEST SUMMARY:")
    # print(summary_df.to_string(index=False))
    # -
    
    # - Plot
    print("Plotting...")
    
    # # Archives
    # visualize_avg_archives(archive_paths, path_dir)
    # visualize_all_archives_per_optimizer(archive_paths, path_dir)
    
    # # Archives stats
    # plot_archives_stats_columns(archive_paths, path_dir)
    # plot_archives_stats_seaborn(archive_paths, path_dir)

    # # Fitness history
    # plot_fitness_history(logs_paths, path_dir, complete=True, threshold=-110)
    # plot_fitness_history_compare(logs_paths, path_dir, complete=False, only_best=False)
    # plot_fitness_history_compare(logs_paths, path_dir, complete=False, only_best=True, threshold=-110)
    
    # # Descriptors 
    # plot_descriptors(archive_paths, path_dir)
    
    # # Hebbian parameters rules
    # plot_hebbian_distribution_per_params(archive_paths, path_dir, nodes)
    # plot_hebbian_distribution_per_nets(archive_paths, path_dir, nodes)
    # # plot_hebbian_distribution_per_params(archive_paths, path_dir, nodes, only_hidden=True)   # only hidden neurons
    # # plot_hebbian_distribution_per_nets(archive_paths, path_dir, nodes, only_hidden=True)     # only hidden neurons
    
    # PCA Rules 
    plot_combined_pca_rules(archive_paths, path_dir, nodes)
    plot_combined_pca_rules(archive_paths, path_dir, nodes, only_hidden=True)  # only hidden neurons

    # 
    print("Plots saved to:", path_dir)
    
if __name__ == "__main__":
    main()  
        