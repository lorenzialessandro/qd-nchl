import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pickle
from sklearn.decomposition import PCA

from network import NCHL, Neuron
from optimizer import MapElites

# -
# Archive

descriptor_names = ["Activation Diversity", "Weight Change Diversity"]

def visualize_archive(map_elites_path, path_dir, cmap="viridis"):
    map_elites = MapElites.load(map_elites_path)
    if map_elites.empty():
        print("Archive is empty, nothing to visualize.")
        return
    # Create a grid to store the fitness values
    grid = np.full(map_elites.map_size, np.nan)
    for pos, (_, fitness, _) in map_elites.archive.items():
        grid[pos[0], pos[1]] = fitness
        
    plt.figure(figsize=(12, 10))
    ax = plt.imshow(grid, cmap=cmap, origin='lower', interpolation='none')
    
    # Compute the tick positions and labels
    x_ticks = np.arange(map_elites.map_size[1])
    x_tick_labels = []
    for i in x_ticks:
        # Calculate true value for this tick position
        min_x, max_x = map_elites.bounds[1]
        true_value = min_x + i * (max_x - min_x) / (map_elites.map_size[1] - 1)
        # Format with appropriate precision
        if max_x - min_x < 0.1:
            # Use more decimal places for small ranges
            x_tick_labels.append(f"{true_value:.4f}")
        else:
            x_tick_labels.append(f"{true_value:.2f}")
    
    # For y-axis (first dimension)
    y_ticks = np.arange(map_elites.map_size[0])
    y_tick_labels = []
    for i in y_ticks:
        # Calculate true value for this tick position
        min_y, max_y = map_elites.bounds[0]
        true_value = min_y + i * (max_y - min_y) / (map_elites.map_size[0] - 1)
        # Format with appropriate precision
        if max_y - min_y < 0.1:
            # Use more decimal places for small ranges
            y_tick_labels.append(f"{true_value:.4f}")
        else:
            y_tick_labels.append(f"{true_value:.2f}")
            
    # Set the tick positions and labels
    plt.xticks(x_ticks, x_tick_labels, rotation=45)
    plt.yticks(y_ticks, y_tick_labels)
    
    # Add labels and title with descriptor names
    plt.xlabel(f'{descriptor_names[1]}')
    plt.ylabel(f'{descriptor_names[0]}')
    plt.title(f"Map-Elite Archive Fitness Values (coverage {map_elites.coverage:.2%})")
    
    # Add colorbar
    cbar = plt.colorbar(ax, label="Fitness")
    
    # Add grid lines
    # plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='black', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "archive.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Archive visualization saved to {os.path.join(path_dir, 'archive.png')}")
    

def visualize_average_archive(list_of_paths, output_dir, min_fitness=None, cmap="viridis"):
    """
    Create an average archive plot from multiple Map-Elites archives.
    
    Args:
        list_of_paths: List of paths to Map-Elites archive files
        output_dir: Directory to save the output plot
        min_fitness: Minimum possible fitness value to use for empty cells in averaging
                     If None, will try to determine from the data
        cmap: Colormap to use for visualization (default: "viridis")
    """
    if not list_of_paths:
        print("No paths provided, cannot visualize average archive.")
        return
    
    # Load the first archive to get map size and bounds
    first_archive = MapElites.load(list_of_paths[0])
    if first_archive.empty():
        print("First archive is empty, cannot determine dimensions.")
        return
    
    map_size = first_archive.map_size
    bounds = first_archive.bounds
    
    # Create a 3D array to store all grids
    all_grids = np.full((len(list_of_paths), map_size[0], map_size[1]), np.nan)
    
    # Valid archives counter and to track min fitness if needed
    valid_archives = 0
    observed_min_fitness = float('inf')
    
    # Process each archive
    for i, path in enumerate(list_of_paths):
        map_elites = MapElites.load(path)
        if map_elites.empty():
            print(f"Archive at {path} is empty, skipping.")
            continue
        
        # Ensure all archives have the same dimensions
        if map_elites.map_size != map_size:
            print(f"Archive at {path} has different dimensions, skipping.")
            continue
        
        valid_archives += 1
        
        # Fill the grid for this archive
        for pos, (_, fitness, _) in map_elites.archive.items():
            all_grids[i, pos[0], pos[1]] = fitness
            observed_min_fitness = min(observed_min_fitness, fitness)
    
    if valid_archives == 0:
        print("No valid archives found, cannot create average visualization.")
        return
    
    # If min_fitness is not provided, use the observed minimum minus a small margin
    if min_fitness is None:
        if observed_min_fitness == float('inf'):
            print("Warning: Could not determine minimum fitness, using 0 as default.")
            min_fitness = 0
        else:
            # Use observed minimum with a small margin (5% below)
            min_fitness = observed_min_fitness - abs(observed_min_fitness * 0.05)
            print(f"Using calculated minimum fitness value: {min_fitness}")
    
    # Create a mask for cells where all archives have NaN (completely empty cells)
    all_empty_mask = np.all(np.isnan(all_grids), axis=0)
    
    # Replace NaN with min_fitness value for averaging
    filled_grids = np.copy(all_grids)
    for i in range(len(list_of_paths)):
        filled_grids[i][np.isnan(filled_grids[i])] = min_fitness
    
    # Calculate average grid
    average_grid = np.mean(filled_grids, axis=0)
    
    # Set cells that were NaN in all archives back to NaN in the average
    average_grid[all_empty_mask] = np.nan
    
    # Calculate coverage (percentage of cells that have solutions in at least one archive)
    filled_cells = np.count_nonzero(~all_empty_mask)
    total_cells = np.prod(map_size)
    coverage = filled_cells / total_cells
    
    # Visualization
    plt.figure(figsize=(12, 10))
    ax = plt.imshow(average_grid, cmap=cmap, origin='lower', interpolation='none')
    
    # Compute the tick positions and labels for x-axis
    x_ticks = np.arange(map_size[1])
    x_tick_labels = []
    for i in x_ticks:
        min_x, max_x = bounds[1]
        true_value = min_x + i * (max_x - min_x) / (map_size[1] - 1)
        if max_x - min_x < 0.1:
            x_tick_labels.append(f"{true_value:.4f}")
        else:
            x_tick_labels.append(f"{true_value:.2f}")
    
    # Compute the tick positions and labels for y-axis
    y_ticks = np.arange(map_size[0])
    y_tick_labels = []
    for i in y_ticks:
        min_y, max_y = bounds[0]
        true_value = min_y + i * (max_y - min_y) / (map_size[0] - 1)
        if max_y - min_y < 0.1:
            y_tick_labels.append(f"{true_value:.4f}")
        else:
            y_tick_labels.append(f"{true_value:.2f}")
    
    # Set the tick positions and labels
    plt.xticks(x_ticks, x_tick_labels, rotation=45)
    plt.yticks(y_ticks, y_tick_labels)
    
    # Add labels and title
    plt.xlabel(f'{descriptor_names[1]}')
    plt.ylabel(f'{descriptor_names[0]}')
    plt.title(f"Average Map-Elite Archive Fitness Values\n(treating empty cells as min fitness {min_fitness:.2f}, coverage {coverage:.2%})")
    
    # Add colorbar
    cbar = plt.colorbar(ax, label="Average Fitness")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_archive.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Average archive visualization saved to {os.path.join(output_dir, 'average_archive.png')}")

def visualize_multiple_archives(list_of_paths, output_dir, cmap="viridis"):
    """
    Create multiple subplots, one for each Map-Elites archive, in a single figure.
    
    Args:
        list_of_paths: List of paths to Map-Elites archive files
        output_dir: Directory to save the output plot
        cmap: Colormap to use for visualization (default: "viridis")
    """
    if not list_of_paths:
        print("No paths provided, cannot visualize archives.")
        return
    
    # Determine number of rows and columns for subplots
    n_plots = len(list_of_paths)
    n_cols = min(3, n_plots)  # Maximum 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_plots == 1:
        axes = np.array([axes])  # Ensure axes is iterable for single plot
    axes = axes.flatten()
    
    # Process each archive
    for i, path in enumerate(list_of_paths):
        if i >= len(axes):
            break
            
        map_elites = MapElites.load(path)
        if map_elites.empty():
            axes[i].text(0.5, 0.5, "Empty Archive", horizontalalignment='center', verticalalignment='center')
            axes[i].set_title(f"Archive {i+1}: {os.path.basename(path)}")
            continue
        
        # Create grid for this archive
        grid = np.full(map_elites.map_size, np.nan)
        for pos, (_, fitness, _) in map_elites.archive.items():
            grid[pos[0], pos[1]] = fitness
        
        # Plot on the corresponding subplot
        im = axes[i].imshow(grid, cmap=cmap, origin='lower', interpolation='none')
        
        # Calculate coverage
        filled_cells = np.count_nonzero(~np.isnan(grid))
        total_cells = np.prod(map_elites.map_size)
        coverage = filled_cells / total_cells
        
        # Set title for this subplot
        axes[i].set_title(f"Archive {i+1}\nCoverage: {coverage:.2%}")
        
        # Compute tick positions and labels for x-axis (simplified for subplots)
        x_ticks = np.linspace(0, map_elites.map_size[1]-1, min(5, map_elites.map_size[1]))
        x_tick_labels = []
        for tick in x_ticks:
            min_x, max_x = map_elites.bounds[1]
            true_value = min_x + tick * (max_x - min_x) / (map_elites.map_size[1] - 1)
            x_tick_labels.append(f"{true_value:.2f}")
        
        # Compute tick positions and labels for y-axis (simplified for subplots)
        y_ticks = np.linspace(0, map_elites.map_size[0]-1, min(5, map_elites.map_size[0]))
        y_tick_labels = []
        for tick in y_ticks:
            min_y, max_y = map_elites.bounds[0]
            true_value = min_y + tick * (max_y - min_y) / (map_elites.map_size[0] - 1)
            y_tick_labels.append(f"{true_value:.2f}")
        
        # Set tick positions and labels
        axes[i].set_xticks(x_ticks)
        axes[i].set_xticklabels(x_tick_labels, rotation=45)
        axes[i].set_yticks(y_ticks)
        axes[i].set_yticklabels(y_tick_labels)
        
        # Add axis labels
        if i % n_cols == 0:  # leftmost column
            axes[i].set_ylabel(descriptor_names[0])
        if i >= (n_rows-1) * n_cols:  # bottom row
            axes[i].set_xlabel(descriptor_names[1])
        
        # Add colorbar for each subplot
        fig.colorbar(im, ax=axes[i], label="Fitness")
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Add overall title
    fig.suptitle("Multiple Map-Elite Archives Comparison", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for overall title
    plt.savefig(os.path.join(output_dir, "multiple_archives.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multiple archives visualization saved to {os.path.join(output_dir, 'multiple_archives.png')}")   

# -
# Fitness history 

def plot_fitness_history(logs_paths, path_dir, complete=False, threshold=None):
    "Plot fitness history of each run, their average, and cumulative global best."
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

    # Convert to numpy arrays
    all_best_fitnesses = np.array(all_best_fitnesses)
    all_avg_fitnesses = np.array(all_avg_fitnesses)
    all_cum_best_fitnesses = np.array(all_cum_best_fitnesses)

    # Calculate means and stds
    avg_best_fitnesses = np.mean(all_best_fitnesses, axis=0)
    avg_avg_fitnesses = np.mean(all_avg_fitnesses, axis=0)
    std_best_fitnesses = np.std(all_best_fitnesses, axis=0)
    std_avg_fitnesses = np.std(all_avg_fitnesses, axis=0)
    avg_cum_best_fitnesses = np.mean(all_cum_best_fitnesses, axis=0)
    std_cum_best_fitnesses = np.std(all_cum_best_fitnesses, axis=0)
    
    plt.figure(figsize=(10, 8))
    # Plot cumulative global best
    plt.plot(avg_cum_best_fitnesses, label="Avg Global Best")
    plt.fill_between(range(len(avg_cum_best_fitnesses)), avg_cum_best_fitnesses - std_cum_best_fitnesses,
                    avg_cum_best_fitnesses + std_cum_best_fitnesses, alpha=0.2)

    if complete:
        # Plot per-generation fitness
        plt.plot(avg_best_fitnesses, label="Avg Best Fitness")
        plt.fill_between(range(len(avg_best_fitnesses)), avg_best_fitnesses - std_best_fitnesses,
                        avg_best_fitnesses + std_best_fitnesses, alpha=0.2)
        plt.plot(avg_avg_fitnesses, label="Avg Avg Fitness")
        plt.fill_between(range(len(avg_avg_fitnesses)), avg_avg_fitnesses - std_avg_fitnesses,
                        avg_avg_fitnesses + std_avg_fitnesses, alpha=0.2)
        
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
    
    plt.legend()
    plt.title("Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig(os.path.join(path_dir, "fitness.png"), dpi=300, bbox_inches='tight')
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
    plt.savefig(os.path.join(path_dir, "fitness.png"), dpi=300, bbox_inches='tight')
    plt.close()

# -
# Rules Analysis

def get_layer_label(idx, total):
    if idx == 0:
        return "Input"
    elif idx == total - 1:
        return "Output"
    else:
        return "Hidden"

def plot_combined_pca_rules(list_of_paths, path_dir):
    """
    Plot PCA of rules with colors based on layer type (Input, Hidden, Output)
    and letters "I", "H", "O" indicating the layer type inside each dot
    """
    all_rules = []
    net_ids = []
    layer_ids = []

    max_layer_count = 0  # to determine layer roles (input, hidden, output)

    for net_id, path in enumerate(list_of_paths):
        net = NCHL.load(path)
        max_layer_count = max(max_layer_count, len(net.neurons))
        for layer_idx, layer in enumerate(net.neurons):
            for neuron in layer:
                rule = neuron.get_rule()
                rule_values = [float(p.item()) if hasattr(p, 'item') else float(p) for p in rule]
                all_rules.append(rule_values)
                net_ids.append(net_id)
                layer_ids.append(layer_idx)

    all_rules = np.array(all_rules)
    net_ids = np.array(net_ids)
    layer_ids = np.array(layer_ids)

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_rules)

    # Setup plot
    plt.figure(figsize=(12, 9))
    unique_nets = np.unique(net_ids)
    unique_layers = np.unique(layer_ids)

    # Define layer type information
    layer_types = {}
    for layer_idx in unique_layers:
        layer_types[layer_idx] = get_layer_label(layer_idx, max_layer_count)
    
    # Distinct colors for layer types (Input, Hidden, Output)
    layer_colors = {
        'Input': '#1f77b4',    # Blue
        'Hidden': '#ff7f0e',   # Orange
        'Output': '#2ca02c'    # Green
    }

    # Plot data points
    for net_id in unique_nets:
        for layer_idx in unique_layers:
            mask = (net_ids == net_id) & (layer_ids == layer_idx)
            if not np.any(mask):
                continue
                
            # Get layer type
            layer_type = layer_types[layer_idx]
            
            # Plot points colored by layer type
            scatter = plt.scatter(
                pca_result[mask, 0],
                pca_result[mask, 1],
                c=layer_colors[layer_type],
                marker='o',
                alpha=0.7,
                s=250,       # Large marker size
                edgecolors='k',
                linewidths=0.7
            )

    # Create layer type legend
    layer_type_handles = []
    for layer_type, color in layer_colors.items():
        layer_type_handles.append(
            Line2D([0], [0], marker='o', color='w', 
                   label=f'{layer_type}',
                   markerfacecolor=color, markersize=10, 
                   markeredgecolor='k')
        )

    # Add the legend for layer types (primary legend)
    plt.legend(handles=layer_type_handles, loc='best', title="Layer Types")

    # Plot formatting
    plt.xlabel(f"PCA 1 - {pca.explained_variance_ratio_[0]:.2f}")
    plt.ylabel(f"PCA 2 - {pca.explained_variance_ratio_[1]:.2f}")
    plt.title("PCA of Hebbian Rules by Layer Type")
    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "pca_multi_net_by_layer.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca_by_net(list_of_paths, path_dir):
    """
    Plot each rule PCA with subplot per net, colored by layer.
    """
    all_rules = []
    net_ids = []
    layer_ids = []
    max_layer_count = 0

    for net_id, path in enumerate(list_of_paths):
        net = NCHL.load(path)
        max_layer_count = max(max_layer_count, len(net.neurons))
        for layer_idx, layer in enumerate(net.neurons):
            for neuron in layer:
                rule = neuron.get_rule()
                rule_values = [float(p.item()) if hasattr(p, 'item') else float(p) for p in rule]
                all_rules.append(rule_values)
                net_ids.append(net_id)
                layer_ids.append(layer_idx)

    all_rules = np.array(all_rules)
    net_ids = np.array(net_ids)
    layer_ids = np.array(layer_ids)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_rules)

    unique_nets = np.unique(net_ids)
    unique_layers = np.unique(layer_ids)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_layers)))

    # Fixed layout for 10 nets
    n_rows, n_cols = 2, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8), sharex=True, sharey=True, constrained_layout=True)
    axes = axes.flatten()

    for ax, net_id in zip(axes, unique_nets):
        for layer_idx in unique_layers:
            mask = (net_ids == net_id) & (layer_ids == layer_idx)
            if np.any(mask):
                ax.scatter(
                    pca_result[mask, 0],
                    pca_result[mask, 1],
                    color=colors[layer_idx],
                    alpha=0.7,
                    s=100,
                )
        ax.set_title(f'Net {net_id}')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')

    # Hide any unused axes
    for j in range(len(unique_nets), len(axes)):
        axes[j].axis('off')

    # Legend
    labels = [get_layer_label(layer_idx, max_layer_count) for layer_idx in unique_layers]
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'{labels[layer_idx]}',
                   markerfacecolor=colors[layer_idx], markersize=10)
        for layer_idx in unique_layers
    ]
    fig.legend(handles=handles, title="Layers", loc='lower center', ncol=len(unique_layers), bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("PCA of Rules by Network (colored by Layer)", fontsize=16)

    # Save figure
    plt.savefig(os.path.join(path_dir, "pca_by_net.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca_by_layer(list_of_paths, path_dir):
    """
    Plot each rule PCA with subplot per layer, colored by net
    """
    all_rules = []
    net_ids = []
    layer_ids = []
    max_layer_count = 0

    for net_id, path in enumerate(list_of_paths):
        net = NCHL.load(path)
        max_layer_count = max(max_layer_count, len(net.neurons))
        for layer_idx, layer in enumerate(net.neurons):
            for neuron in layer:
                rule = neuron.get_rule()
                rule_values = [float(p.item()) if hasattr(p, 'item') else float(p) for p in rule]
                all_rules.append(rule_values)
                net_ids.append(net_id)
                layer_ids.append(layer_idx)

    all_rules = np.array(all_rules)
    net_ids = np.array(net_ids)
    layer_ids = np.array(layer_ids)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_rules)

    unique_nets = np.unique(net_ids)
    unique_layers = np.unique(layer_ids)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_nets)))

    fig, axes = plt.subplots(1, len(unique_layers), figsize=(6 * len(unique_layers), 6), sharex=True, sharey=True)
    if len(unique_layers) == 1:
        axes = [axes]

    for ax, layer_idx in zip(axes, unique_layers):
        label = get_layer_label(layer_idx, max_layer_count)
        for net_id in unique_nets:
            mask = (layer_ids == layer_idx) & (net_ids == net_id)
            if np.any(mask):
                ax.scatter(
                    pca_result[mask, 0],
                    pca_result[mask, 1],
                    color=colors[net_id],
                    alpha=0.7,
                    s=100,
                )
        ax.set_title(f'{label} layer')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        
    # legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Net {net_id}',
                            markerfacecolor=colors[net_id], markersize=10) for net_id in unique_nets]
    fig.legend(handles=handles, title="Networks", loc='lower center', ncol=len(unique_nets), bbox_to_anchor=(0.5, -0.06))
    
    fig.suptitle("PCA of Rules by Layer (colored by Network)", fontsize=16)
    plt.savefig(os.path.join(path_dir, "pca_by_layer.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_pca_rules(path_agent, path_dir):
    """
    Perform PCA on the rules of the network and plot the results.
    """
    net = NCHL.load(path_agent)
    rules = []
    layer_indices = []
    
    for layer_idx, layer in enumerate(net.neurons):
        for neuron in layer:
            rule = neuron.get_rule()
            rule_values = []
            for param in rule:
                if hasattr(param, 'item'): 
                    rule_values.append(param.item())
                else:
                    rule_values.append(float(param))
            
            rules.append(rule_values)
            layer_indices.append(layer_idx)
    
    rules = np.array(rules)
    layer_indices = np.array(layer_indices)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(rules)
    
    unique_layers = np.unique(layer_indices)
    cmap = plt.cm.get_cmap('viridis', len(unique_layers))
    
    plt.figure(figsize=(10, 8))
    for i, layer_idx in enumerate(unique_layers):
        mask = layer_indices == layer_idx
        
        plt.scatter(
            pca_result[mask, 0], 
            pca_result[mask, 1],
            c=[cmap(i)],  # Use the same color for all points in this layer
            label=f"Layer {layer_idx}",
            alpha=0.7,
            s=100
        )
    
    plt.title("PCA of Hebbian Rules by Layer")
    plt.xlabel(f"PCA 1 - {pca.explained_variance_ratio_[0]:.2f}")
    plt.ylabel(f"PCA 2 - {pca.explained_variance_ratio_[1]:.2f}")
    plt.legend()
    for i, neuron in enumerate(net.all_neurons):
        plt.annotate('N ' + neuron.neuron_id, (pca_result[i, 0], pca_result[i, 1]), fontsize=8)
    plt.savefig(os.path.join(path_dir, "pca_rules.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
# - 
# Descriptors

def plot_descriptors_by_net(list_of_paths, path_dir):
    """
    Plot the 2 descriptor in 2d space, colored by net
    """
    mapse = [MapElites.load(path) for path in list_of_paths]
    k = 10
    
    # scatter plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(mapse)))
    for i, map_elites in enumerate(mapse):
        bk = map_elites.get_best_k(k)
        for j in range(k):
            best_individual, best_fitness, best_descriptor = bk[j]
            plt.plot(best_descriptor[1], best_descriptor[0], 'o', markersize=11,
                     markeredgewidth=0, markerfacecolor=colors[i], alpha=0.7)

    plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label=f'Archive {i}',
                            markerfacecolor=colors[i], markersize=10) for i in range(len(mapse))],
               title="Runs", loc='upper left', bbox_to_anchor=(0, 1))
    
    min_x, max_x = mapse[0].bounds[1]
    min_y, max_y = mapse[0].bounds[0]
    x_margin = (max_x - min_x) * 0.05
    y_margin = (max_y - min_y) * 0.05
    plt.xlim(min_x - x_margin, max_x + x_margin)
    plt.ylim(min_y - y_margin, max_y + y_margin)
    
    plt.xticks(np.linspace(min_x, max_x, 5))
    plt.yticks(np.linspace(min_y, max_y, 5))
        
    
    plt.title("Descriptors by Network")
    plt.xlabel(f'{descriptor_names[1]}')
    plt.ylabel(f'{descriptor_names[0]}')
    # plt.legend()
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.savefig(os.path.join(path_dir, "descriptors_by_net.png"), dpi=300, bbox_inches='tight')
    plt.close()

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def plot_descriptors_by_net_zoom(list_of_paths, path_dir):
    """
    Plot the 2 descriptors in 2D space, colored by network, with a zoomed inset
    showing only the actual min-max area of the data points.
    """
    mapse = [MapElites.load(path) for path in list_of_paths]
    k = 5

    # Create main plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(mapse)))

    all_descriptors = []

    for i, map_elites in enumerate(mapse):
        bk = map_elites.get_best_k(k)
        for j in range(k):
            _, _, best_descriptor = bk[j]
            x, y = best_descriptor[1], best_descriptor[0]
            ax.plot(x, y, 'o', markersize=6,
                    markeredgewidth=0, markerfacecolor=colors[i], alpha=0.7)
            all_descriptors.append((x, y))

    # Main plot limits: [0,1] for both axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_yticks(np.linspace(0, 1, 5))

    ax.set_title(f'Best {k} Descriptors by Network')
    ax.set_xlabel(f'{descriptor_names[1]}')
    ax.set_ylabel(f'{descriptor_names[0]}')

    # Legend
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', label=f'Archive {i}',
               markerfacecolor=colors[i], markersize=11)
        for i in range(len(mapse))
    ], title="Runs", loc='upper left', bbox_to_anchor=(0, 1))

    # Compute tight limits for inset
    x_vals, y_vals = zip(*all_descriptors)
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    x_margin = (max_x - min_x) * 0.06
    y_margin = (max_y - min_y) * 0.06

    # Create zoomed-in inset showing only the true data bounds
    axins = inset_axes(ax, width="40%", height="40%", loc='upper right')
    for i, map_elites in enumerate(mapse):
        bk = map_elites.get_best_k(k)
        for j in range(k):
            _, _, best_descriptor = bk[j]
            axins.plot(best_descriptor[1], best_descriptor[0], 'o', markersize=9,
                       markeredgewidth=0, markerfacecolor=colors[i], alpha=0.7)

    axins.set_xlim(min_x - x_margin, max_x + x_margin)
    axins.set_ylim(min_y - y_margin, max_y + y_margin)
    # axins.set_xticks(np.linspace(min_x, max_x, 3))
    # axins.set_yticks(np.linspace(min_y, max_y, 3))
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # Save and close
    plt.savefig(os.path.join(path_dir, "descriptors_by_net.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
# -

# -------------------------------------
# -

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot.py <path_file> <path_dir>")
        print("path_file: .txt file with paths to archives")
        print("path_dir: directory to save plots")
        sys.exit(1)
        
    path_file = sys.argv[1]     # .txt file with paths
    path_dir = sys.argv[2]      # directory to save plots
    os.makedirs(path_dir, exist_ok=True)  
        
    # Load paths from the file
    archive_paths = []  # list of paths to archives
    agent_paths = []    # list of paths to agents
    logs_paths = []     # list of paths to logs (history)
    
    with open(sys.argv[1], 'r') as file:
        for line in file.readlines():
            line = line.strip()
            archive_paths.append(line + '/archive.pkl')
            agent_paths.append(line + '/best_nchl.pkl') 
            logs_paths.append(line + '/log.txt')
    
    # - Plot
    
    # Fitness history
    # w# plot_descriptors_by_net_zoom(archive_paths, path_dir)
    
    # Rules 
    plot_combined_pca_rules(agent_paths, path_dir)
    # plot_pca_by_net(agent_paths, path_dir)
    # plot_pca_by_layer(agent_paths, path_dir)
    
if __name__ == "__main__":
    main()  
        