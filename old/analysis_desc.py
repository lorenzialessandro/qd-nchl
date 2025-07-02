import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from scipy.stats import gaussian_kde
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA

from network import NCHL, Neuron

# -
# activations
def plot_activations_per_neuron(nchl, path_dir):
    neuron_ids = [neuron.neuron_id for neuron in nchl.all_neurons]
    num_neurons = len(neuron_ids)
    cols = 5
    rows = (num_neurons + cols - 1) // cols  
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()  
    
    for idx, neuron in enumerate(nchl.all_neurons):
        activations = neuron.activations
        if activations is None:
            continue
        
        # Plot history of activations
        ax = axes[idx]
        ax.plot(activations, label=f'Neuron {neuron.neuron_id}')
        ax.set_title(f'Neuron {neuron.neuron_id}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Activation')
        ax.set_ylim([-1, 1]) 
        ax.legend()
        
    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "activations_per_neuron.png"), dpi=300, bbox_inches='tight')
    plt.close()

    
def plot_activations_time(nchl, path_dir):
    # plot the history of the net activations over time using average activation
    activations = []
    for neuron in nchl.all_neurons:
        if neuron.activations is not None:
            activations.append(neuron.activations)
    
    activations = np.array(activations)
    activations = np.mean(activations, axis=0)
    time = np.arange(len(activations))
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, activations)
    plt.title("Average Activations Over Time")
    plt.xlabel("Time")
    plt.ylabel("Activation")
    plt.ylim([-1, 1])
    plt.savefig(os.path.join(path_dir, "activations_time.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_activations_time_per_layer(nchl, path_dir):
    # create a unique plot displaying the activations of all neurons in a layer over time
    num_layers = len(nchl.neurons)
    max_neurons = max(len(layer) for layer in nchl.neurons)
    
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))

    for layer_idx, layer in enumerate(nchl.neurons):
        activations = []
        for neuron in layer:
            if neuron.activations is not None:
                activations.append(neuron.activations)
        activations = np.array(activations)
        activations = np.mean(activations, axis=0)
        time = np.arange(len(activations))
        
        ax = axes[layer_idx]
        ax.plot(time, activations)
        ax.set_title(f'Layer {layer_idx}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Activation')
        ax.set_ylim([-1, 1])
            
    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "activations_time_per_layer.png"), dpi=300, bbox_inches='tight')
    plt.close()
         
# -
# weight_changes
def plot_weight_changes_per_neuron(nchl, path_dir):
    neuron_ids = [neuron.neuron_id for neuron in nchl.all_neurons]
    num_neurons = len(neuron_ids)
    cols = 5
    rows = (num_neurons + cols - 1) // cols  
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()  
    
    for idx, neuron in enumerate(nchl.all_neurons):
        weight_changes = neuron.weight_changes
        if weight_changes is None:
            continue
        
        # Plot history of weight changes
        ax = axes[idx]
        ax.plot(weight_changes, label=f'Neuron {neuron.neuron_id}')
        ax.set_title(f'Neuron {neuron.neuron_id}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Weight Change')
        ax.set_ylim([-1, 1]) 
        ax.legend()
        
    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "weight_changes_per_neuron.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_weight_changes_time(nchl, path_dir):
    # plot the history of the net weight changes over time using average weight change
    weight_changes = []
    for neuron in nchl.all_neurons:
        if neuron.weight_changes is not None:
            weight_changes.append(neuron.weight_changes)
    
    weight_changes = np.array(weight_changes)
    weight_changes = np.mean(weight_changes, axis=0)
    time = np.arange(len(weight_changes))
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, weight_changes)
    plt.title("Average Weight Changes Over Time")
    plt.xlabel("Time")
    plt.ylabel("Weight Change")
    plt.ylim([-1, 1])
    plt.savefig(os.path.join(path_dir, "weight_changes_time.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_weight_changes_time_per_layer(nchl, path_dir):
    # create a unique plot displaying the weight changes of all neurons in a layer over time
    num_layers = len(nchl.neurons)
    max_neurons = max(len(layer) for layer in nchl.neurons)
    
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))

    for layer_idx, layer in enumerate(nchl.neurons):
        weight_changes = []
        for neuron in layer:
            if neuron.weight_changes is not None:
                weight_changes.append(neuron.weight_changes)
        weight_changes = np.array(weight_changes)
        weight_changes = np.mean(weight_changes, axis=0)
        time = np.arange(len(weight_changes))
        
        ax = axes[layer_idx]
        ax.plot(time, weight_changes)
        ax.set_title(f'Layer {layer_idx}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Weight Change')
        ax.set_ylim([-1, 1])
            
    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "weight_changes_time_per_layer.png"), dpi=300, bbox_inches='tight')
    plt.close()   
    
# -
# descriptors space
def plot_neurons_descriptors_space(nchl, path_dir):
    avg_activations = []
    avg_weight_changes = []
    neuron_ids = []

    for neuron in nchl.all_neurons:
        if neuron.activations is not None and neuron.weight_changes is not None:
            avg_act = np.mean(neuron.activations)
            avg_wchg = np.mean(np.abs(neuron.weight_changes))
            avg_activations.append(avg_act)
            avg_weight_changes.append(avg_wchg)
            neuron_ids.append(neuron.neuron_id)

    if not avg_activations:
        print("No neuron data available for descriptor plot.")
        return

    avg_activations = np.array(avg_activations)
    avg_weight_changes = np.array(avg_weight_changes)

    plt.figure(figsize=(10, 8))
    plt.scatter(avg_activations, avg_weight_changes, s=100)
    plt.xlabel("Average Activation")
    plt.ylabel("Average Absolute Weight Change")
    plt.title("Neuron Descriptor Space")

    for i, neuron_id in enumerate(neuron_ids):
        plt.text(avg_activations[i], avg_weight_changes[i], 'N ' + str(neuron_id), ha='right', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "neurons_descriptor_space.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_neurons_descriptor_space_pca(nchl, path_dir):
    descriptors = []
    neuron_ids = []

    for neuron in nchl.all_neurons:
        if neuron.activations is not None and neuron.weight_changes is not None:
            avg_act = np.mean(neuron.activations)
            avg_wchg = np.mean(np.abs(neuron.weight_changes))
            # Extend this list with more features if needed
            descriptors.append([avg_act, avg_wchg])
            neuron_ids.append(neuron.neuron_id)

    if not descriptors:
        print("No neuron data available for descriptor space.")
        return

    descriptors = np.array(descriptors)

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    descriptors_2d = pca.fit_transform(descriptors)

    plt.figure(figsize=(10, 8))
    plt.scatter(descriptors_2d[:, 0], descriptors_2d[:, 1], s=100)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Neuron Descriptor Space (PCA)")

    for i, neuron_id in enumerate(neuron_ids):
        plt.text(descriptors_2d[i, 0], descriptors_2d[i, 1], 'N ' + str(neuron_id), ha='right', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "neurons_descriptor_space_pca.png"), dpi=300, bbox_inches='tight')
    plt.close()


# Example usage:
if __name__ == "__main__":
    import sys
    path_dir = sys.argv[1]
    
    # search for the best model name (names are in the shape iteration_fitness.pkl)
    best_model_name = None
    best_fitness = -np.inf
    for file in os.listdir(path_dir):
        if file.endswith(".pkl"):
            try:
                iteration, fitness = map(float, file[:-4].split("_")) 
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_model_name = file
            except ValueError:
                continue
            
    print(f"Best model name: {best_model_name}")
    nchl = NCHL.load(path_dir, best_model_name)
    
    path_out = os.path.join(path_dir, os.pardir, "a_desc")
    os.makedirs(path_out, exist_ok=True)
    
    # Plot activations
    plot_activations_per_neuron(nchl, path_out)
    plot_activations_time(nchl, path_out)
    plot_activations_time_per_layer(nchl, path_out)
    # Plot weight changes
    plot_weight_changes_per_neuron(nchl, path_out)
    plot_weight_changes_time(nchl, path_out)
    plot_weight_changes_time_per_layer(nchl, path_out)
    # Plot descriptors space
    plot_neurons_descriptors_space(nchl, path_out)
    plot_neurons_descriptor_space_pca(nchl, path_out)
    