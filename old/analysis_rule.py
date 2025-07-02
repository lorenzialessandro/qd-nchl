import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
import os

from network import NCHL, Neuron

def plot_distribution_rule_per_neuron(nchl, path_dir):
    param_names = ['pre_factor', 'post_factor', 'correlation', 'decorrelation', 'eta']
    neuron_ids = [neuron.neuron_id for neuron in nchl.all_neurons]
    neuron_rules = [neuron.get_rule() for neuron in nchl.all_neurons]
    
    # Plot the 5 parameters for each neuron
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, param in enumerate(param_names):
        axes[i].scatter(neuron_ids, [getattr(neuron, param) for neuron in nchl.all_neurons])
        axes[i].set_title(param)
        axes[i].set_xlabel('Neuron ID')
        axes[i].set_ylabel(param)
        axes[i].set_xticks(neuron_ids)
        axes[i].set_xticklabels(neuron_ids, rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "distribution_rule_per_neuron.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_rule_per_neuron(nchl, path_dir):
    param_names = ['pre_factor', 'post_factor', 'correlation', 'decorrelation', 'eta']
    neuron_ids = [neuron.neuron_id for neuron in nchl.all_neurons]

    num_neurons = len(neuron_ids)
    cols = 5  # or set dynamically based on num_neurons
    rows = (num_neurons + cols - 1) // cols  # ensures all neurons get a subplot

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()  # make it easier to index

    for idx, neuron in enumerate(nchl.all_neurons):
        rule_values = [getattr(neuron, param) for param in param_names]
        ax = axes[idx]
        ax.bar(param_names, rule_values)
        ax.set_xticks(param_names)
        ax.set_xticklabels(param_names, rotation=45)
        ax.set_title(f'Neuron {neuron.neuron_id}')
        ax.set_ylabel('Value')
        ax.set_ylim([min(0, min(rule_values)), max(rule_values)*1.1])  # better visualization

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "rules_per_neuron.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_rule_per_neuron_per_layer(network, path_dir):
    param_names = ['pre_factor', 'post_factor', 'correlation', 'decorrelation', 'eta']

    for layer_idx, layer in enumerate(network.neurons):
        neuron_ids = [neuron.neuron_id for neuron in layer]
        num_neurons = len(neuron_ids)

        cols = 5
        rows = (num_neurons + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten()

        for idx, neuron in enumerate(layer):
            rule_values = [getattr(neuron, param) for param in param_names]
            ax = axes[idx]
            ax.bar(param_names, rule_values)
            ax.set_xticks(range(len(param_names)))
            ax.set_xticklabels(param_names, rotation=45)
            ax.set_title(f'Neuron {neuron.neuron_id}')
            ax.set_ylabel('Value')
            ax.set_ylim([min(0, min(rule_values)), max(rule_values) * 1.1])

        # Hide any unused subplots
        for j in range(num_neurons, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        filename = f"rules_per_neuron_layer_{layer_idx}.png"
        plt.savefig(os.path.join(path_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()


def plot_rules_per_layer(network, path_dir):
    param_names = ['pre_factor', 'post_factor', 'correlation', 'decorrelation', 'eta']
    num_params = len(param_names)
    num_layers = len(network.neurons)

    fig, axes = plt.subplots(1, num_params, figsize=(5 * num_params, 5))

    for i, param in enumerate(param_names):
        param_data_per_layer = []
        for layer_idx, layer in enumerate(network.neurons):
            values = [getattr(neuron, param) for neuron in layer]
            param_data_per_layer.append(values)

        ax = axes[i]
        ax.boxplot(param_data_per_layer, labels=[f"L{idx}" for idx in range(num_layers)])
        ax.set_title(param)
        ax.set_xlabel("Layer")
        ax.set_ylabel(param)

    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "rules_distribution_per_layer.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_rule_per_neuron_per_layer_grouped(network, path_dir):
    param_names = ['pre_factor', 'post_factor', 'correlation', 'decorrelation', 'eta']
    num_layers = len(network.neurons)

    # Determine the max number of neurons in any layer for figure width
    max_neurons = max(len(layer) for layer in network.neurons)

    fig, axes = plt.subplots(num_layers, max_neurons, figsize=(max_neurons * 3, num_layers * 3), squeeze=False)

    for layer_idx, layer in enumerate(network.neurons):
        for neuron_idx, neuron in enumerate(layer):
            ax = axes[layer_idx][neuron_idx]
            rule_values = [getattr(neuron, param) for param in param_names]
            ax.bar(param_names, rule_values)
            ax.set_title(f'Neuron {neuron.neuron_id}', fontsize=9)
            ax.set_ylim([min(0, min(rule_values)), max(rule_values) * 1.1])
            ax.set_xticks(range(len(param_names)))
            ax.set_xticklabels(param_names, rotation=45, fontsize=8)
            ax.tick_params(axis='y', labelsize=8)

        # Hide unused axes in this row (layer)
        for neuron_idx in range(len(layer), max_neurons):
            fig.delaxes(axes[layer_idx][neuron_idx])

        # Optional: label the layer at the start of the row
        axes[layer_idx][0].set_ylabel(f'Layer {layer_idx}', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(path_dir, "rules_per_neuron_layer_grouped.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_pca_rules(nchl, path_dir):
    """
    Plot PCA analysis for NCHL.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # Extract the rules from the neurons
    rules = np.array([neuron.get_rule() for neuron in nchl.all_neurons])
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(rules)

    # Plot the PCA result
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.title('PCA of NCHL Rules')
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
    # add labels
    for i, neuron in enumerate(nchl.all_neurons):
        plt.annotate(neuron.neuron_id, (pca_result[i, 0], pca_result[i, 1]), fontsize=8)
    
    plt.savefig(os.path.join(path_dir, "pca_rules.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_pca_rules_3(nchl, path_dir):
    """
    Plot 3D PCA analysis for NCHL.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Extract the rules from the neurons
    rules = np.array([neuron.get_rule() for neuron in nchl.all_neurons])
    
    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(rules)
    # Plot the PCA result
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2])
    ax.set_title('3D PCA of NCHL Rules')
    ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_zlabel(f'PCA Component 3 ({pca.explained_variance_ratio_[2]:.2%})')
    # add labels
    for i, neuron in enumerate(nchl.all_neurons):
        ax.text(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], neuron.neuron_id, fontsize=8)
    plt.savefig(os.path.join(path_dir, "pca_rules_3d.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    