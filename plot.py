import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.decomposition import PCA

from network import NCHL, Neuron
from optimizer import MapElites

# -
# Archive 

def visualize_archive(map_elites, path_dir, cmap="viridis", annot=False, high=False):
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
    ax = sns.heatmap(fitness_map, annot=annot, fmt=".2f", cmap=cmap, cbar=True, xticklabels=x_ticklabels, yticklabels=y_ticklabels)
    
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

def visualize_average_archive(list_of_paths, path_dir, cmap="viridis", annot=False, high=False):
    map_elites_list = [MapElites.load(path) for path in list_of_paths]
    
    map_shape = map_elites_list[0].map_size
    fitness_stack = np.full((len(map_elites_list), *map_shape), np.nan)

    for i, map_elites in enumerate(map_elites_list):
        if map_elites.empty():
            continue
        for pos, (_, fitness, _) in map_elites.archive.items():
            fitness_stack[i, pos[0], pos[1]] = fitness

    # Compute mean ignoring NaNs
    fitness_avg = np.nanmean(fitness_stack, axis=0)

    # Generate tick labels
    x_min, x_max = map_elites_list[0].bounds[0]
    y_min, y_max = map_elites_list[0].bounds[1]

    x_ticks = np.linspace(x_min, x_max, map_shape[0])
    y_ticks = np.linspace(y_min, y_max, map_shape[1])
    
    x_ticklabels = [f"{val:.2f}" for val in x_ticks]
    y_ticklabels = [f"{val:.2f}" for val in y_ticks]

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(fitness_avg, annot=annot, fmt=".2f", cmap=cmap, cbar=True,
                     xticklabels=x_ticklabels, yticklabels=y_ticklabels)
    # ax.invert_yaxis()

    plt.title("Average Map-Elite Archive Fitness")
    plt.xlabel("std mean neuron activations")
    plt.ylabel("std mean neuron weight changes")

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "average_archive.png"), dpi=300, bbox_inches='tight')
    plt.close()
    

# -
# Fitness history 

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
    Plot PCA of rules of each networks in unique plot colored by net and layer
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

    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_nets)))  # Use a colormap for colors
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>']  # Extend if needed

    for net_id in unique_nets:
        for layer_idx in unique_layers:
            mask = (net_ids == net_id) & (layer_ids == layer_idx)
            if not np.any(mask):
                continue
            plt.scatter(
                pca_result[mask, 0],
                pca_result[mask, 1],
                c=[colors[net_id]],
                marker=markers[layer_idx % len(markers)],
                alpha=0.7,
                s=80,
                edgecolors='k',
                linewidths=0.5
            )

    # Create network color legend
    color_handles = [
        Line2D([0], [0], marker='o', color='w', label=f'Net {net_id}',
               markerfacecolor=colors[net_id], markersize=10, markeredgecolor='k')
        for net_id in unique_nets
    ]

    # Create unique layer type handles (Input, Hidden, Output)
    layer_types = []
    marker_handles = []
    for layer_idx in unique_layers:
        label = get_layer_label(layer_idx, max_layer_count)
        if label not in layer_types:
            layer_types.append(label)
            marker_handles.append(
                Line2D([0], [0], marker=markers[layer_idx % len(markers)],
                       markerfacecolor='none', markeredgecolor='k', label=label, linestyle='None', markersize=10)
            )

    # Put the 2 legends lower but one side the other
    legend1 = plt.legend(handles=color_handles, loc='upper right', title="Networks")
    legend2 = plt.legend(handles=marker_handles, loc='lower right', title="Layers")
    plt.gca().add_artist(legend1)  # Add the first legend to the plot

    # Plot formatting
    plt.xlabel(f"PCA 1 - {pca.explained_variance_ratio_[0]:.2f}")
    plt.ylabel(f"PCA 2 - {pca.explained_variance_ratio_[1]:.2f}")
    plt.title("PCA of Hebbian Rules Across Multiple Networks")
    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "pca_multi_net.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca_by_net(list_of_paths, path_dir):
    """
    Plot each rule PCA with subplot per net, colored by color
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

    fig, axes = plt.subplots(1, len(unique_nets), figsize=(6 * len(unique_nets), 6), sharex=True, sharey=True)
    if len(unique_nets) == 1:
        axes = [axes]

    for ax, net_id in zip(axes, unique_nets):
        for layer_idx in unique_layers:
            mask = (net_ids == net_id) & (layer_ids == layer_idx)
            if np.any(mask):
                ax.scatter(
                    pca_result[mask, 0],
                    pca_result[mask, 1],
                    color=colors[layer_idx],
                    alpha=0.7
                )
        ax.set_title(f'Net {net_id}')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
    
    # legend
    labels = [get_layer_label(layer_idx, max_layer_count) for layer_idx in unique_layers]
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'{labels[layer_idx]}',
                            markerfacecolor=colors[layer_idx], markersize=10) for layer_idx in unique_layers]
    fig.legend(handles=handles, title="Layers", loc='lower center', ncol=len(unique_layers), bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("PCA of Rules by Network (colored by Layer)", fontsize=16)
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
                    alpha=0.7
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
    
    # scatter plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(mapse)))
    for i, net in enumerate(mapse):
        act_diversity, weight_diversity = net.get_best()[-1]
        plt.scatter(act_diversity, weight_diversity, label=f"Net {i}", color=colors[i], alpha=0.7, s=100)
        
    plt.title("Descriptors by Network")
    plt.xlabel("std mean neuron activations")
    plt.ylabel("std mean neuron weight changes")
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(os.path.join(path_dir, "descriptors_by_net.png"), dpi=300, bbox_inches='tight')
    plt.close()
        
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_descriptors_by_net_zoom(list_of_paths, path_dir):
    """
    Plot the 2 descriptors in 2D space with a zoomed-in inset.
    """
    mapse = [MapElites.load(path) for path in list_of_paths]
    
    # scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(mapse)))

    # Store points to replot in inset
    all_points = []
    k = 5

    for i, net in enumerate(mapse):
        dk = net.get_best_k(k)
        for d in dk:
            act_diversity, weight_diversity = d[-1]
            ax.scatter(act_diversity, weight_diversity, color=colors[i], alpha=0.7, s=50)
            all_points.append((act_diversity, weight_diversity, colors[i]))

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("Descriptors by Network (with Zoom)")
    ax.set_xlabel("std mean neuron activations")
    ax.set_ylabel("std mean neuron weight changes")
    # add unique net legends
    ax.legend(handles=[Line2D([0], [0], marker='o', color='w', label=f'Map {i}',
                            markerfacecolor=colors[i], markersize=10) for i in range(len(mapse))],
               title="Runs", loc='upper left', bbox_to_anchor=(0, 1))

    # Create zoomed-in inset
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    axins = inset_axes(ax, width="40%", height="40%", loc='upper right')
    
    # Plot same points in the zoom inset
    for act_div, weight_div, color in all_points:
        axins.scatter(act_div, weight_div, color=color, alpha=0.7, s=100)
        
    # Compute the limits for the inset
    min_x = min(act_div for act_div, _, _ in all_points)
    max_x = max(act_div for act_div, _, _ in all_points)
    min_y = min(weight_div for _, weight_div, _ in all_points)
    max_y = max(weight_div for _, weight_div, _ in all_points)
    # Add some padding
    padding = 0.01
    min_x = max(0, min_x - padding)
    max_x = min(1, max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(1, max_y + padding)
    
    # Set smaller limits for zoom
    axins.set_xlim(min_x, max_x)
    axins.set_ylim(min_y, max_y)
    # axins.set_xticks([])
    # axins.set_yticks([])

    # Draw box and lines to indicate zoom
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black")

    plt.savefig(os.path.join(path_dir, "descriptors_by_net_zoom.png"), dpi=300, bbox_inches='tight')
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
    
    with open(sys.argv[1], 'r') as file:
        for line in file.readlines():
            line = line.strip()
            archive_paths.append(line + '/archive.pkl')
            agent_paths.append(line + '/best_nchl.pkl') 
    
    # - Plot
    
    # Archive 
    visualize_average_archive(archive_paths, path_dir)
    
    # Descriptors
    plot_descriptors_by_net_zoom(archive_paths, path_dir)
    
    # Rules 
    # plot_combined_pca_rules(agent_paths, path_dir)
    plot_pca_by_net(agent_paths, path_dir)
    plot_pca_by_layer(agent_paths, path_dir)
    
if __name__ == "__main__":
    main()  
        