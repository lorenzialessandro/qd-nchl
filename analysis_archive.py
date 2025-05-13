import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os

from optimizer import MapElites

def visualize_descriptor_space(archive, path_dir):
    """Plot 3: Scatter plot of solutions in descriptor space"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if archive.empty():
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Descriptor Space', fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(path_dir, "descriptor_space.png"), dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    descriptors = []
    fitnesses = []
    
    for pos, (individual, fitness, descriptor) in archive.archive.items():
        descriptors.append(descriptor)
        fitnesses.append(fitness)
    
    descriptors = np.array(descriptors)
    fitnesses = np.array(fitnesses)
    
    # Create scatter plot
    scatter = ax.scatter(descriptors[:, 0], descriptors[:, 1], c=fitnesses, 
                        cmap='RdYlGn', s=100, alpha=1.0, edgecolors='black', linewidth=1.0)
    
    ax.set_xlabel('Activation Diversity')
    ax.set_ylabel('Weight Change Diversity')
    ax.set_title('Solutions in Descriptor Space', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Fitness', rotation=270, labelpad=15)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "descriptor_space.png"), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_best_solutions(archive, path_dir):
    """Plot 4: Characteristics of top solutions"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if archive.empty():
        return
    
    # Get top 10 solutions
    k = min(10, len(archive.archive))
    best_solutions = archive.get_best_k(k)
    
    # Extract data
    ranks = list(range(1, k + 1))
    fitnesses = [sol[1] for sol in best_solutions]
    act_divs = [sol[2][0] for sol in best_solutions]
    weight_divs = [sol[2][1] for sol in best_solutions]
    
    # Create bar plot
    x = np.arange(k)
    width = 0.25
    
    # Plot raw values without any scaling
    ax.bar(x, act_divs, width, label='Act. Div.', color='#2196F3', alpha=0.7)
    ax.bar(x + width, weight_divs, width, label='Weight Div.', color='#FF9800', alpha=0.7)
    
    ax.set_xlabel('Rank')
    ax.set_ylabel('Diversity Value')
    ax.set_title(f'Top {k} Solutions Characteristics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(ranks)
    # Add fitness values as text annotations
    for i in range(k):
        ax.text(i + width/2, act_divs[i], f'{fitnesses[i]:.2f}', ha='center', va='bottom')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Use subplots_adjust instead of tight_layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
    
    plt.savefig(os.path.join(path_dir, "best_solutions.png"), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_best_solutions_subplots(archive, path_dir):
    """Plot 4: Characteristics of top solutions with side-by-side subplots for each diversity measure"""
    if archive.empty():
        return
    
    # Get top 10 solutions
    k = min(10, len(archive.archive))
    best_solutions = archive.get_best_k(k)
    
    # Extract data
    ranks = list(range(1, k + 1))
    fitnesses = [sol[1] for sol in best_solutions]
    act_divs = [sol[2][0] for sol in best_solutions]
    weight_divs = [sol[2][1] for sol in best_solutions]
    
    max_lim = max(max(act_divs), max(weight_divs))
    
    # Create figure with side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Width of bars
    width = 0.7
    x = np.arange(k)
    
    # Subplot 1: Activation Diversity
    ax1.bar(x, act_divs, width, color='#2196F3', alpha=0.7)
    ax1.set_ylabel('Activation Diversity')
    ax1.set_xlabel('Rank')
    ax1.set_title('Top Solutions: Activation Diversity', fontsize=12, fontweight='bold')
    # ax1.set_ylim(0, 1.0)  # Set max y value to 1.0
    ax1.set_ylim(0, max_lim)  # Set max y value to the maximum of both measures
    ax1.set_xticks(x)
    ax1.set_xticklabels(ranks)
    
    # Add rank and fitness values as text annotations for subplot 1
    for i in range(k):
        ax1.text(i, act_divs[i], f'{fitnesses[i]:.2f}', ha='center', va='bottom')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Weight Diversity
    ax2.bar(x, weight_divs, width, color='#FF9800', alpha=0.7)
    ax2.set_ylabel('Weight Diversity')
    ax2.set_xlabel('Rank')
    ax2.set_title('Top Solutions: Weight Diversity', fontsize=12, fontweight='bold')
    # ax2.set_ylim(0, 1.0)  # Set max y value to 1.0
    ax2.set_ylim(0, max_lim)  # Set max y value to the maximum of both measures
    ax2.set_xticks(x)
    ax2.set_xticklabels(ranks)
    
    # Add rank and fitness values as text annotations for subplot 2
    for i in range(k):
        ax2.text(i, weight_divs[i], f'{fitnesses[i]:.2f}', ha='center', va='bottom')
    ax2.grid(True, alpha=0.3)
    
    # Add a common title
    plt.suptitle('Diversity Measures of Top Solutions', fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    
    # Save figure
    plt.savefig(os.path.join(path_dir, "best_solutions_subplots.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
def visualize_archive(map_elites, path_dir=None, cmap="viridis", annot=True, high=False):
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
    
# -
def plot_archive_analysis(archive, path_dir):
    visualize_archive(archive, cmap="Greens", annot=False, high=False)
    visualize_descriptor_space(archive, path_dir)
    visualize_best_solutions(archive, path_dir)
    visualize_best_solutions_subplots(archive, path_dir)

if __name__ == "__main__":
    path_dir = sys.argv[1]
    path_archive = os.path.join(path_dir, "archive.pkl")
    archive = MapElites.load(path_archive)
    
    path_out = os.path.join(path_dir, "a_archive")
    os.makedirs(path_out, exist_ok=True)
    
    plot_archive_analysis(archive, path_out)
