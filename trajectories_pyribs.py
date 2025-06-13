import numpy as np
import torch
import os
import pickle
import gymnasium as gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
import argparse
from pathlib import Path
from matplotlib import cm
import json

from network import NCHL

# Optimizer names
optimizers_names = ["MAPElites", "CMAME", "CMAMAE"]

# Utility function
def get_optimizer_paths(list_of_paths):
    optimizer_paths = {name: [] for name in optimizers_names}
    for path in list_of_paths:
        for name in optimizers_names:
            if name in path:
                optimizer_paths[name].append(path)
                break
    return optimizer_paths

def load_model(model_path):
    """Load a saved NCHL model"""
    print(f"Loading model from {model_path}...")
    return NCHL.load(model_path)

def run_test_rollouts(model, env, num_rollouts=3, steps=1000, debug=False):
    """
    Run test rollouts and collect trajectories.
    For each rollout, store:
    - Input values A_i
    - Pre-synaptic A_j (before activation)
    - Post-synaptic activations A_j' (after activation)
    """
    trajectories = []
    action_space = env.action_space
    is_continuous = isinstance(action_space, gym.spaces.Box)
    
    for rollout_idx in range(num_rollouts):
        print(f"Running rollout {rollout_idx+1}/{num_rollouts}...")
        obs, _ = env.reset()
            
        trajectory = {
            'inputs': [],       # Input values A_i
            'pre_synaptic': [], # Pre-synaptic values A_j (before activation)
            'post_synaptic': [], # Post-synaptic values A_j' (after activation)
            'actions': [],      # Actions taken
            'rewards': [],      # Rewards received
            'termination_reason': None  # Why the episode ended
        }
        
        total_reward = 0
        
        for step_idx in range(steps):
            # Store input values
            trajectory['inputs'].append(obs.copy())
            
            # Forward pass through the model
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32).to(model.device)
                
                # Track the values after each layer
                layer_inputs = [x]
                for i, layer in enumerate(model.network):
                    # Get pre-activation (pre-synaptic) values
                    pre_activation = layer(layer_inputs[-1])
                    
                    # If this is the last layer, store the pre-synaptic values
                    if i == len(model.network) - 1:
                        trajectory['pre_synaptic'].append(pre_activation.cpu().numpy().flatten())
                    
                    # Apply activation function (tanh)
                    post_activation = torch.tanh(pre_activation)
                    layer_inputs.append(post_activation)
                
                # Store post-synaptic values (final output after activation)
                trajectory['post_synaptic'].append(post_activation.cpu().numpy().flatten())
                
                # Use the model output to take an action
                action_values = post_activation.cpu().numpy()
                
                if is_continuous:
                    # For continuous action spaces (like Ant-v5)
                    # Actions are already in the range [-1, 1] due to tanh activation
                    action = action_values
                    
                    # Ensure the action has the right dimension
                    if len(action) != action_space.shape[0]:
                        # If dimensions don't match, reshape or pad/truncate as needed
                        if len(action) > action_space.shape[0]:
                            action = action[:action_space.shape[0]]
                        else:
                            # If output is smaller, pad with zeros (or another strategy)
                            action = np.pad(action, (0, action_space.shape[0] - len(action)), 'constant')
                else:
                    # For discrete action spaces (like CartPole, MountainCar, LunarLander)
                    action = np.argmax(action_values)
                  
            # Store the action
            trajectory['actions'].append(action)
            
            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Store termination reason for debugging
            if terminated:
                trajectory['termination_reason'] = "Terminated"
            elif truncated:
                trajectory['termination_reason'] = "Truncated (max steps)"
        
            # Store reward
            trajectory['rewards'].append(reward)
            total_reward += reward
            
            if done:
                if debug:
                    print(f"Rollout {rollout_idx+1} ended after {step_idx+1} steps with total reward {total_reward}")
                    print(f"Termination reason: {trajectory['termination_reason']}")
                    if isinstance(env.action_space, gym.spaces.Discrete):
                        actions_taken = trajectory['actions']
                        action_counts = {a: actions_taken.count(a) for a in set(actions_taken)}
                        print(f"Action distribution: {action_counts}")
                break
        
        # If we didn't break (episode didn't end), note that we reached max steps
        else:
            trajectory['termination_reason'] = "Reached max steps"
            if debug:
                print(f"Rollout {rollout_idx+1} completed full {steps} steps with total reward {total_reward}")
        
        trajectories.append(trajectory)
    
    return trajectories

def perform_pca_analysis(trajectories, n_components=3, debug=False):
    """
    Perform PCA analysis on the trajectories.
    For each step, perform PCA on:
    - Input values
    - Pre-synaptic values
    - Post-synaptic values
    """
    if debug:
        print("Starting PCA analysis...")
    
    # Find minimum trajectory length
    steps = min([len(traj['inputs']) for traj in trajectories])
    if debug:
        print(f"Minimum trajectory length: {steps} steps")
    
    # Initialize result structures
    input_pcas = []
    pre_pcas = []
    post_pcas = []
    
    for step in range(steps):
        if debug and step % 100 == 0:
            print(f"Analyzing step {step}/{steps}...")
            
        # Collect data for this step across all trajectories
        step_inputs = np.array([np.array(traj['inputs'][step]).flatten() for traj in trajectories])
        step_pre = np.array([np.array(traj['pre_synaptic'][step]).flatten() for traj in trajectories])
        step_post = np.array([np.array(traj['post_synaptic'][step]).flatten() for traj in trajectories])
        
        # Ensure data shapes are correct
        if step_inputs.ndim == 1:
            step_inputs = step_inputs.reshape(-1, 1)
        if step_pre.ndim == 1:
            step_pre = step_pre.reshape(-1, 1)
        if step_post.ndim == 1:
            step_post = step_post.reshape(-1, 1)
        
        # Perform PCA for each type of data
        # Only apply PCA if we have enough samples and dimensions
        if len(step_inputs) > 1:
            if step_inputs.shape[1] > 1:
                # More than one feature, can do real PCA
                n_comps = min(n_components, step_inputs.shape[1], step_inputs.shape[0])
                input_pca = PCA(n_components=n_comps)
                input_pca.fit(step_inputs)
                input_pcas.append(input_pca)
        else:
            raise ValueError("Not enough data points for PCA analysis")
        
        # Same for pre-synaptic values
        if len(step_pre) > 1:
            if step_pre.shape[1] > 1:
                n_comps = min(n_components, step_pre.shape[1], step_pre.shape[0])
                pre_pca = PCA(n_components=n_comps)
                pre_pca.fit(step_pre)
                pre_pcas.append(pre_pca)
        else:
            raise ValueError("Not enough data points for PCA analysis")
        
        # Same for post-synaptic values
        if len(step_post) > 1:
            if step_post.shape[1] > 1:
                n_comps = min(n_components, step_post.shape[1], step_post.shape[0])
                post_pca = PCA(n_components=n_comps)
                post_pca.fit(step_post)
                post_pcas.append(post_pca)
        else:
            raise ValueError("Not enough data points for PCA analysis")
    
    # Collect results
    results = {
        'input': {
            'explained_variance': np.array([pca.explained_variance_ratio_ for pca in input_pcas]),
            'components': np.array([pca.components_ for pca in input_pcas])
        },
        'pre_synaptic': {
            'explained_variance': np.array([pca.explained_variance_ratio_ for pca in pre_pcas]),
            'components': np.array([pca.components_ for pca in pre_pcas])
        },
        'post_synaptic': {
            'explained_variance': np.array([pca.explained_variance_ratio_ for pca in post_pcas]),
            'components': np.array([pca.components_ for pca in post_pcas])
        }
    }
    
    return results, input_pcas, pre_pcas, post_pcas

def project_trajectories_on_pca(trajectories, pcas, n_components=3, debug=False):
    """
    Project trajectories onto PCA components for visualization.
    For each trajectory, project each step onto the PCA components.
    """
    if debug:
        print("Projecting trajectories onto PCA space...")
        
    projected_trajectories = []
    
    for traj_idx, traj in enumerate(trajectories):
        if debug:
            print(f"Projecting trajectory {traj_idx+1}/{len(trajectories)}...")
            
        projected_traj = {
            'input': [],
            'pre_synaptic': [],
            'post_synaptic': [],
            'rewards': traj['rewards'].copy(),  # Copy rewards for visualization
            'steps': []  # Store step indices for 3D plotting
        }
        
        # Find the minimum number of steps (trajectory length vs available PCAs)
        steps = min(len(traj['inputs']), len(pcas['input']))
        if steps == 0:
            if debug:
                print(f"Warning: Trajectory {traj_idx+1} has no valid steps to project")
            continue
            
        for step in range(steps):
            projected_traj['steps'].append(step)  # Store step for 3D plotting
            
            try:
                # Project input values
                input_data = np.array(traj['inputs'][step]).flatten().reshape(1, -1)
                projection = pcas['input'][step].transform(input_data)[0]
                # Pad with zeros if fewer components than requested
                if len(projection) < n_components:
                    projection = np.pad(projection, (0, n_components - len(projection)))
                # Or truncate if more components than requested
                elif len(projection) > n_components:
                    projection = projection[:n_components]
                projected_traj['input'].append(projection)
                
                # Project pre-synaptic values
                pre_data = np.array(traj['pre_synaptic'][step]).flatten().reshape(1, -1)
                projection = pcas['pre_synaptic'][step].transform(pre_data)[0]
                if len(projection) < n_components:
                    projection = np.pad(projection, (0, n_components - len(projection)))
                elif len(projection) > n_components:
                    projection = projection[:n_components]
                projected_traj['pre_synaptic'].append(projection)
                
                # Project post-synaptic values
                post_data = np.array(traj['post_synaptic'][step]).flatten().reshape(1, -1)
                projection = pcas['post_synaptic'][step].transform(post_data)[0]
                if len(projection) < n_components:
                    projection = np.pad(projection, (0, n_components - len(projection)))
                elif len(projection) > n_components:
                    projection = projection[:n_components]
                projected_traj['post_synaptic'].append(projection)
                
            except Exception as e:
                if debug:
                    print(f"Error projecting step {step} of trajectory {traj_idx+1}: {e}")
                continue
        
        # Only add trajectories with valid data
        if projected_traj['input'] and projected_traj['pre_synaptic'] and projected_traj['post_synaptic']:
            projected_trajectories.append(projected_traj)
        elif debug:
            print(f"Warning: Trajectory {traj_idx+1} has no valid projections")
    
    return projected_trajectories

def visualize_optimizer_pca_trajectories_3d(all_projected_trajectories_by_optimizer, output_dir=None):
    """
    Visualize PCA trajectories for all optimizers in 3D, with separate subplots for each optimizer.
    """
    type_labels = ['Input Values', 'Pre-synaptic Values', 'Post-synaptic Values']
    data_types = ['input', 'pre_synaptic', 'post_synaptic']
    
    # Create a figure with subplots: 3 columns (data types) x num_optimizers rows
    n_optimizers = len(all_projected_trajectories_by_optimizer)
    fig = plt.figure(figsize=(18, 6 * n_optimizers))
    
    # Color maps for different optimizers
    optimizer_colors = {'MAPElites': 'tab:blue', 'CMAME': 'tab:orange', 'CMAMAE': 'tab:green'}
    
    for opt_idx, (optimizer_name, optimizer_trajectories) in enumerate(all_projected_trajectories_by_optimizer.items()):
        for i, (data_type, type_label) in enumerate(zip(data_types, type_labels)):
            ax = fig.add_subplot(n_optimizers, 3, opt_idx * 3 + i + 1, projection='3d')
            
            # Get base color for this optimizer
            base_color = optimizer_colors.get(optimizer_name, 'tab:gray')
            
            for model_idx, projected_trajectories in enumerate(optimizer_trajectories):
                for traj_idx, traj in enumerate(projected_trajectories):
                    if data_type not in traj or not traj[data_type]:
                        continue
                    
                    # Extract data
                    data = np.array(traj[data_type])
                    steps = traj['steps']
                    
                    # Check if we have enough dimensions for a proper 3D plot
                    if data.shape[1] >= 2:
                        # Use first two PCA components for x, y
                        x, y = data[:, 0], data[:, 1]
                        
                        # Plot the trajectory with some transparency
                        ax.scatter3D(x, y, steps, alpha=0.6, s=30, 
                                   color=base_color, 
                                   label=f'{optimizer_name} M{model_idx+1} R{traj_idx+1}' if i == 0 and model_idx == 0 and traj_idx == 0 else "")
                    
                    else:
                        # If only one dimension, use time as second dimension
                        x = data[:, 0]
                        ax.plot3D(x, steps, steps, alpha=0.7, linewidth=2, color=base_color,
                                label=f'{optimizer_name} M{model_idx+1} R{traj_idx+1}' if i == 0 and model_idx == 0 and traj_idx == 0 else "")
                        
            # Set labels
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2' if data.shape[1] >= 2 else 'Time Step')
            ax.set_zlabel('Time Step')
            ax.set_title(f'{optimizer_name} - {type_label}')
            
            # Add legend only to the first column
            if i == 0:
                ax.legend(loc="upper left", fontsize='small')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, '3d_pca_trajectories_by_optimizer.png'), dpi=300, bbox_inches='tight')
        print(f"Optimizer PCA visualization saved to {os.path.join(output_dir, '3d_pca_trajectories_by_optimizer.png')}")
    plt.close(fig)

def visualize_single_optimizer_pca_trajectories_3d(projected_trajectories, optimizer_name, output_dir=None):
    """
    Visualize PCA trajectories for a single optimizer with subplots for each model.
    Similar to the original visualize_all_pca_trajectories_combined but for one optimizer.
    """
    # Data types to visualize
    data_types = ['input', 'pre_synaptic', 'post_synaptic']
    type_labels = ['Input Values', 'Pre-synaptic Values', 'Post-synaptic Values']
    
    # Number of models for this optimizer
    n_models = len(projected_trajectories)
    if n_models == 0:
        print(f"No models to visualize for {optimizer_name}")
        return
    
    # Create figure with subplots arranged by model (rows) and data type (columns)
    fig = plt.figure(figsize=(18, 5 * n_models))
    gs = gridspec.GridSpec(n_models, 3, figure=fig)
    
    # Custom colormap for different rollouts
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot for each model and data type
    for model_idx, model_trajectories in enumerate(projected_trajectories):
        for type_idx, (data_type, type_label) in enumerate(zip(data_types, type_labels)):
            # Create 3D subplot
            ax = fig.add_subplot(gs[model_idx, type_idx], projection='3d')
            
            # Skip if no trajectories
            if not model_trajectories:
                ax.text2D(0.5, 0.5, "No trajectory data available", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax.transAxes)
                continue
            
            # Plot each trajectory for this model
            for traj_idx, traj in enumerate(model_trajectories):
                if data_type not in traj or not traj[data_type]:
                    continue
                    
                # Extract data
                data = np.array(traj[data_type])
                steps = traj['steps']
                
                # Check if we have enough dimensions for a proper 3D plot
                if data.shape[1] >= 2:
                    # Use first two PCA components for x, y
                    x, y = data[:, 0], data[:, 1]
                    
                    # Plot the trajectory - use a different color for each rollout
                    color_idx = traj_idx % len(colors)
                    ax.plot3D(x, y, steps, alpha=0.7, linewidth=2,
                            color=colors[color_idx], label=f'Rollout {traj_idx+1}')
                    
                    # Mark start point (green circle)
                    ax.scatter3D(x[0], y[0], steps[0], color='green', s=80, marker='o',
                               label='Start' if traj_idx == 0 else "")
                    
                    # Mark end point (red cross)
                    ax.scatter3D(x[-1], y[-1], steps[-1], color='red', s=80, marker='x',
                               label='End' if traj_idx == 0 else "")
                else:
                    # If only one dimension, use time as second dimension
                    x = data[:, 0]
                    color_idx = traj_idx % len(colors)
                    ax.plot3D(x, steps, steps, alpha=0.7, linewidth=2,
                            color=colors[color_idx], label=f'Rollout {traj_idx+1}')
            
            # Set labels and title
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2' if data.shape[1] >= 2 else 'Time Step')
            ax.set_zlabel('Time Step')
            
            # Create title with model name and data type
            ax.set_title(f'{optimizer_name} Model {model_idx+1} - {type_label}')
            
            # Add legend only to the first plot of each row
            if type_idx == 0:
                ax.legend(loc='upper left', fontsize='small')
                
            # Set consistent view angle for better comparison
            ax.view_init(elev=30, azim=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        filename = f'{optimizer_name.lower()}_models_pca_comparison.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"{optimizer_name} PCA comparison saved to {os.path.join(output_dir, filename)}")
    
    plt.close(fig)

def visualize_single_optimizer_reward_landscape_combined(trajectories, optimizer_name, output_dir=None):
    """
    Visualize reward landscape for a single optimizer with subplots for each model.
    """
    n_models = len(trajectories)
    if n_models == 0:
        print(f"No models to visualize for {optimizer_name}")
        return
        
    # Create figure with subplots arranged by model
    n_cols = 3  # number of subplots per row
    n_rows = (n_models + n_cols - 1) // n_cols  # ceil division
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    
    # Determine appropriate dimensions to visualize based on input size
    sample_input = trajectories[0][0]['inputs'][0]
    input_dims = len(sample_input)

    # Select dimensions to plot based on environment
    if input_dims == 4:  # Likely CartPole
        dim1, dim2 = 0, 1  
        x_label, y_label = 'Cart Position', 'Cart Velocity'
    elif input_dims == 2:  # Likely MountainCar
        dim1, dim2 = 0, 1
        x_label, y_label = 'Position', 'Velocity'
    elif input_dims == 8:  # Likely LunarLander
        dim1, dim2 = 0, 1
        x_label, y_label = 'X Position', 'Y Position'
    else:
        # For unknown environments, just use the first two dimensions
        dim1, dim2 = 0, min(1, input_dims-1)
        x_label, y_label = 'Dimension 1', 'Dimension 2'
    
    # Plot for each model
    for model_idx, model_trajectories in enumerate(trajectories):
        # Extract state-reward pairs from all trajectories
        states_dim1 = []
        states_dim2 = []
        rewards = []
        for traj in model_trajectories:
            for i, (state, reward) in enumerate(zip(traj['inputs'], traj['rewards'])):
                states_dim1.append(state[dim1])
                states_dim2.append(state[dim2])
                rewards.append(reward)
        
        # If we have enough data points, create a scatter plot
        if states_dim1:
            ax = fig.add_subplot(gs[model_idx // n_cols, model_idx % n_cols])
            sc = ax.scatter(states_dim1, states_dim2, c=rewards, cmap='viridis', 
                          alpha=0.6, s=30, edgecolors='none')
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f'{optimizer_name} Model {model_idx+1} Reward Landscape')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add colorbar
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.01)
            cbar.set_label('Reward')
            
    plt.tight_layout()
    
    if output_dir:
        filename = f'{optimizer_name.lower()}_reward_landscape_combined.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"{optimizer_name} reward landscape saved to {os.path.join(output_dir, filename)}")
        
    plt.close(fig)

def visualize_single_optimizer_reward_landscape_3d(trajectories, optimizer_name, output_dir=None):
    """
    Visualize 3D reward landscape for a single optimizer with subplots for each model.
    """
    n_models = len(trajectories)
    if n_models == 0:
        print(f"No models to visualize for {optimizer_name}")
        return
    
    # Create figure with subplots arranged by model
    n_cols = 3  # number of subplots per row
    n_rows = (n_models + n_cols - 1) // n_cols  # ceil division
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    
    # Determine appropriate dimensions to visualize based on input size
    sample_input = trajectories[0][0]['inputs'][0]
    input_dims = len(sample_input)

    # Select dimensions to plot based on environment
    if input_dims == 4:  # Likely CartPole
        dim1, dim2 = 0, 1  
        x_label, y_label = 'Cart Position', 'Cart Velocity'
    elif input_dims == 2:  # Likely MountainCar
        dim1, dim2 = 0, 1
        x_label, y_label = 'Position', 'Velocity'
    elif input_dims == 8:  # Likely LunarLander
        dim1, dim2 = 0, 1
        x_label, y_label = 'X Position', 'Y Position'
    else:
        # For unknown environments, just use the first two dimensions
        dim1, dim2 = 0, min(1, input_dims-1)
        x_label, y_label = 'Dimension 1', 'Dimension 2'
    
    # Plot for each model
    for model_idx, model_trajectories in enumerate(trajectories):
        # Extract state-reward-time tuples from all trajectories
        states_dim1 = []
        states_dim2 = []
        rewards = []
        time_steps = []
        
        for traj_idx, traj in enumerate(model_trajectories):
            for step_idx, (state, reward) in enumerate(zip(traj['inputs'], traj['rewards'])):
                states_dim1.append(state[dim1])
                states_dim2.append(state[dim2])
                rewards.append(reward)
                time_steps.append(step_idx)
        
        # If we have enough data points, create a 3D scatter plot
        if states_dim1:
            ax = fig.add_subplot(gs[model_idx // n_cols, model_idx % n_cols], projection='3d')
            
            # Create 3D scatter plot with rewards as color
            sc = ax.scatter3D(states_dim1, states_dim2, time_steps, 
                            c=rewards, cmap='viridis', alpha=0.6, s=30)
            
            # Set labels
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel('Time Step')
            ax.set_title(f'{optimizer_name} Model {model_idx+1} 3D Reward Landscape')
            
            # Add colorbar
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.1, shrink=0.8)
            cbar.set_label('Reward')
            
            # Set consistent view angle for better comparison
            ax.view_init(elev=20, azim=45)
            
    plt.tight_layout()
    
    if output_dir:
        filename = f'{optimizer_name.lower()}_reward_landscape_3d_combined.png'
        plt.savefig(os.path.join(output_dir, filename), 
                   dpi=300, bbox_inches='tight')
        print(f"{optimizer_name} 3D reward landscape saved to {os.path.join(output_dir, filename)}")
        
    plt.close(fig)

def visualize_optimizer_reward_landscape_combined(all_trajectories_by_optimizer, output_dir=None):
    """
    Visualize reward landscape for all optimizers in a single figure with subplots.
    """
    
    # Determine appropriate dimensions to visualize based on input size
    # Get sample from first available trajectory
    sample_input = None
    for optimizer_trajectories in all_trajectories_by_optimizer.values():
        if optimizer_trajectories and optimizer_trajectories[0]:
            sample_input = optimizer_trajectories[0][0]['inputs'][0]
            break
    
    if sample_input is None:
        print("No trajectory data available for reward landscape visualization")
        return
        
    input_dims = len(sample_input)

    # Select dimensions to plot based on environment
    if input_dims == 4:  # Likely CartPole
        dim1, dim2 = 0, 1  
        x_label, y_label = 'Cart Position', 'Cart Velocity'
    elif input_dims == 2:  # Likely MountainCar
        dim1, dim2 = 0, 1
        x_label, y_label = 'Position', 'Velocity'
    elif input_dims == 8:  # Likely LunarLander
        dim1, dim2 = 0, 1
        x_label, y_label = 'X Position', 'Y Position'
    else:
        # For unknown environments, just use the first two dimensions
        dim1, dim2 = 0, min(1, input_dims-1)
        x_label, y_label = 'Dimension 1', 'Dimension 2'
    
    # Create figure with subplots for each optimizer
    n_optimizers = len(all_trajectories_by_optimizer)
    fig = plt.figure(figsize=(6 * n_optimizers, 5))
    
    for opt_idx, (optimizer_name, optimizer_trajectories) in enumerate(all_trajectories_by_optimizer.items()):
        ax = fig.add_subplot(1, n_optimizers, opt_idx + 1)
        
        # Extract state-reward pairs from all trajectories for this optimizer
        states_dim1 = []
        states_dim2 = []
        rewards = []
        
        for model_trajectories in optimizer_trajectories:
            for traj in model_trajectories:
                for i, (state, reward) in enumerate(zip(traj['inputs'], traj['rewards'])):
                    states_dim1.append(state[dim1])
                    states_dim2.append(state[dim2])
                    rewards.append(reward)
        
        # If we have enough data points, create a scatter plot
        if states_dim1:
            sc = ax.scatter(states_dim1, states_dim2, c=rewards, cmap='viridis', 
                          alpha=0.6, s=30, edgecolors='none')
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f'{optimizer_name} Reward Landscape')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add colorbar
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.01)
            cbar.set_label('Reward')
            
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'reward_landscape_by_optimizer.png'), dpi=300, bbox_inches='tight')
        print(f"Optimizer reward landscape visualization saved to {os.path.join(output_dir, 'reward_landscape_by_optimizer.png')}")
        
    plt.close(fig)

def visualize_optimizer_reward_landscape_3d(all_trajectories_by_optimizer, output_dir=None):
    """
    Visualize 3D reward landscape for all optimizers with time on z-axis.
    """
    
    # Determine appropriate dimensions to visualize based on input size
    sample_input = None
    for optimizer_trajectories in all_trajectories_by_optimizer.values():
        if optimizer_trajectories and optimizer_trajectories[0]:
            sample_input = optimizer_trajectories[0][0]['inputs'][0]
            break
    
    if sample_input is None:
        print("No trajectory data available for 3D reward landscape visualization")
        return
        
    input_dims = len(sample_input)

    # Select dimensions to plot based on environment
    if input_dims == 4:  # Likely CartPole
        dim1, dim2 = 0, 1  
        x_label, y_label = 'Cart Position', 'Cart Velocity'
    elif input_dims == 2:  # Likely MountainCar
        dim1, dim2 = 0, 1
        x_label, y_label = 'Position', 'Velocity'
    elif input_dims == 8:  # Likely LunarLander
        dim1, dim2 = 0, 1
        x_label, y_label = 'X Position', 'Y Position'
    else:
        # For unknown environments, just use the first two dimensions
        dim1, dim2 = 0, min(1, input_dims-1)
        x_label, y_label = 'Dimension 1', 'Dimension 2'
    
    # Create figure with 3D subplots for each optimizer
    n_optimizers = len(all_trajectories_by_optimizer)
    fig = plt.figure(figsize=(6 * n_optimizers, 5))
    
    for opt_idx, (optimizer_name, optimizer_trajectories) in enumerate(all_trajectories_by_optimizer.items()):
        ax = fig.add_subplot(1, n_optimizers, opt_idx + 1, projection='3d')
        
        # Extract state-reward-time tuples from all trajectories for this optimizer
        states_dim1 = []
        states_dim2 = []
        rewards = []
        time_steps = []
        
        for model_trajectories in optimizer_trajectories:
            for traj_idx, traj in enumerate(model_trajectories):
                for step_idx, (state, reward) in enumerate(zip(traj['inputs'], traj['rewards'])):
                    states_dim1.append(state[dim1])
                    states_dim2.append(state[dim2])
                    rewards.append(reward)
                    time_steps.append(step_idx)
        
        # If we have enough data points, create a 3D scatter plot
        if states_dim1:
            # Create 3D scatter plot with rewards as color
            sc = ax.scatter3D(states_dim1, states_dim2, time_steps, 
                            c=rewards, cmap='viridis', alpha=0.6, s=30)
            
            # Set labels
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel('Time Step')
            ax.set_title(f'{optimizer_name} 3D Reward Landscape')
            
            # Add colorbar
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.1, shrink=0.8)
            cbar.set_label('Reward')
            
            # Set consistent view angle for better comparison
            ax.view_init(elev=20, azim=45)
            
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'reward_landscape_3d_by_optimizer.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"3D optimizer reward landscape visualization saved to {os.path.join(output_dir, 'reward_landscape_3d_by_optimizer.png')}")
        
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='NCHL Behavioral Analysis with Multiple Optimizers')
    parser.add_argument('path_file', type=str, 
                      help='Path to file containing paths to trained NCHL model files')
    parser.add_argument('nodes', type=json.loads)
    parser.add_argument('task', type=str, help='Task name (e.g., CartPole-v1, Ant-v5)')
    parser.add_argument('output_dir', type=str, help='Output directory for results')
    parser.add_argument('--num_rollouts', type=int, default=3, help='Number of test rollouts per model')
    parser.add_argument('--n_components', type=int, default=3, help='Number of PCA components to use')
    parser.add_argument('--steps', type=int, default=1000, help='Maximum number of steps per rollout')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read all archive paths
    all_archive_paths = []
    with open(args.path_file, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            all_archive_paths.append(line + '/archive.pkl')
    
    # Group paths by optimizer
    optimizer_paths = get_optimizer_paths(all_archive_paths)
    
    print("Processing trained models by optimizer...")
    print(f"Found optimizers: {list(optimizer_paths.keys())}")
    for opt_name, paths in optimizer_paths.items():
        print(f"  {opt_name}: {len(paths)} models")
    
    # Process models for each optimizer
    all_trajectories_by_optimizer = {}
    all_projected_trajectories_by_optimizer = {}
    
    for optimizer_name, archive_paths in optimizer_paths.items():
        if not archive_paths:  # Skip if no paths for this optimizer
            continue
            
        print(f"\nAnalyzing {optimizer_name} models...")
        optimizer_trajectories = []
        optimizer_projected_trajectories = []
        
        for i, path in enumerate(archive_paths):
            print(f"  Analyzing model {i+1}/{len(archive_paths)}: {path}")
            
            try:
                archive = pickle.load(open(path, 'rb'))
                best_elite = archive.best_elite
                
                # Load model
                model = NCHL(nodes=args.nodes)
                model.set_params(best_elite['solution'])
                
                env = gym.make(args.task)

                # Run test rollouts
                trajectories = run_test_rollouts(model, env, num_rollouts=args.num_rollouts, 
                                               steps=args.steps, debug=args.debug)
                
                # Perform PCA analysis
                results, input_pcas, pre_pcas, post_pcas = perform_pca_analysis(
                    trajectories, n_components=args.n_components, debug=args.debug)
                
                # Store PCAs for projection
                pcas = {
                    'input': input_pcas,
                    'pre_synaptic': pre_pcas,
                    'post_synaptic': post_pcas
                }
                
                # Project trajectories onto PCA space
                projected_trajectories = project_trajectories_on_pca(
                    trajectories, pcas, n_components=args.n_components, debug=args.debug)
                
                # Store results for this model
                optimizer_trajectories.append(trajectories)
                optimizer_projected_trajectories.append(projected_trajectories)
                
                # Close environment
                env.close()
                
            except Exception as e:
                print(f"    Error processing {path}: {e}")
                continue
        
        # Store results for this optimizer
        if optimizer_trajectories:  # Only store if we have valid data
            all_trajectories_by_optimizer[optimizer_name] = optimizer_trajectories
            all_projected_trajectories_by_optimizer[optimizer_name] = optimizer_projected_trajectories
    
    # Generate visualizations comparing optimizers
    print("\nGenerating optimizer comparison visualizations...")
    
    # Visualize PCA trajectories by optimizer (comparison plots)
    visualize_optimizer_pca_trajectories_3d(all_projected_trajectories_by_optimizer, args.output_dir)
    
    # Visualize reward landscapes by optimizer (comparison plots)
    visualize_optimizer_reward_landscape_combined(all_trajectories_by_optimizer, args.output_dir)
    visualize_optimizer_reward_landscape_3d(all_trajectories_by_optimizer, args.output_dir)
    
    # Generate individual plots for each optimizer
    print("\nGenerating individual optimizer visualizations...")
    for optimizer_name, optimizer_trajectories in all_trajectories_by_optimizer.items():
        print(f"  Generating plots for {optimizer_name}...")
        
        # Individual PCA trajectories plot for this optimizer
        if optimizer_name in all_projected_trajectories_by_optimizer:
            visualize_single_optimizer_pca_trajectories_3d(
                all_projected_trajectories_by_optimizer[optimizer_name], 
                optimizer_name, 
                args.output_dir
            )
        
        # Individual reward landscape plots for this optimizer
        visualize_single_optimizer_reward_landscape_combined(
            optimizer_trajectories, 
            optimizer_name, 
            args.output_dir
        )
        visualize_single_optimizer_reward_landscape_3d(
            optimizer_trajectories, 
            optimizer_name, 
            args.output_dir
        )
    
    # Generate summary statistics
    print("\nSummary Statistics by Optimizer:")
    for optimizer_name, optimizer_trajectories in all_trajectories_by_optimizer.items():
        total_trajectories = sum(len(model_trajs) for model_trajs in optimizer_trajectories)
        avg_rewards = []
        
        for model_trajs in optimizer_trajectories:
            for traj in model_trajs:
                avg_rewards.append(sum(traj['rewards']))
        
        if avg_rewards:
            print(f"  {optimizer_name}:")
            print(f"    Models: {len(optimizer_trajectories)}")
            print(f"    Total trajectories: {total_trajectories}")
            print(f"    Mean episode reward: {np.mean(avg_rewards):.2f} ± {np.std(avg_rewards):.2f}")
            print(f"    Max episode reward: {np.max(avg_rewards):.2f}")
            print(f"    Min episode reward: {np.min(avg_rewards):.2f}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()