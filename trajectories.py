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

from network import NCHL

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
                
                #
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
    
def visualize_all_pca_trajectories_3d(all_projected_trajectories, output_dir=None):
    """
    Visualize PCA trajectories for all models in 3D.
    """
    fig = plt.figure(figsize=(18, 6))
    
    type_labels = ['Input Values', 'Pre-synaptic Values', 'Post-synaptic Values']
    data_types = ['input', 'pre_synaptic', 'post_synaptic']
    
    for i, (data_type, type_label) in enumerate(zip(data_types, type_labels)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        for model_idx, projected_trajectories in enumerate(all_projected_trajectories):
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
                    
                    # Plot the trajectory
                    # ax.plot3D(x, y, steps, alpha=0.7, linewidth=2,
                    #         label=f'Model {model_idx+1} Traj {traj_idx+1}' if i == 0 else "")
                    ax.scatter3D(x, y, steps, alpha=0.5, s=50, marker='o',
                                label=f'Model {model_idx+1} R. {traj_idx+1}' if i == 0 else "")
                
                    # # Mark start point (green circle)
                    # ax.scatter3D(x[0], y[0], steps[0], color='green', s=80, marker='o',
                    #            label='Start' if traj_idx == 0 else "")
                    
                    # # Mark end point (red cross)
                    # ax.scatter3D(x[-1], y[-1], steps[-1], color='red', s=80, marker='x',
                    #            label='End' if traj_idx == 0 else "")
                
                else:
                    # If only one dimension, use time as second dimension
                    x = data[:, 0]
                    ax.plot3D(x, steps, steps, alpha=0.8, linewidth=2,
                            label=f'Model {model_idx+1} Traj {traj_idx+1}' if i == 0 else "")
                    
        # Set labels
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2' if data.shape[1] >= 2 else 'Time Step')
        ax.set_zlabel('Time Step')
        ax.set_title(f'{type_label} in PCA Space')
        # Add legend to first plot only
        # if i == 0:
        #     ax.legend(loc="lower left", ncol=len(all_projected_trajectories), bbox_to_anchor=(0, -0.3))
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, '3d_pca_trajectories.png'), dpi=300, bbox_inches='tight')
        print(f"PCA visualization saved to {os.path.join(output_dir, '3d_pca_trajectories.png')}")
    plt.close(fig)
    
def visualize_all_pca_trajectories_means_3d(all_projected_trajectories, output_dir=None):
    """
    Visualize mean PCA trajectories for all models in 3D.
    """
    fig = plt.figure(figsize=(18, 6))
    
    type_labels = ['Input Values', 'Pre-synaptic Values', 'Post-synaptic Values']
    data_types = ['input', 'pre_synaptic', 'post_synaptic']
    
    # compute mean trajectories for each model
    mean_trajectories = []
    for model_idx, projected_trajectories in enumerate(all_projected_trajectories):
        # Collect trajectories by data_type across all sequences
        collected = {dtype: [] for dtype in data_types}

        for traj in projected_trajectories:
            for dtype in data_types:
                if dtype in traj and traj[dtype] is not None:
                    collected[dtype].append(traj[dtype])  # shape: (time, components)

        # Compute mean trajectory over models (axis=0)
        mean_traj = {}
        for dtype in data_types:
            if collected[dtype]:
                stacked = np.stack(collected[dtype], axis=0)  # shape: (num_models, time, components)
                mean_traj[dtype] = np.mean(stacked, axis=0)   # shape: (time, components)
            else:
                mean_traj[dtype] = None

        mean_trajectories.append(mean_traj)

    for i, (data_type, type_label) in enumerate(zip(data_types, type_labels)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        for model_idx, mean_traj in enumerate(mean_trajectories):
            if data_type not in mean_traj or mean_traj[data_type] is None:
                continue
            
            # Extract data
            data = np.array(mean_traj[data_type])
            steps = np.arange(data.shape[0])
            
            # Check if we have enough dimensions for a proper 3D plot
            if data.shape[1] >= 2:
                # Use first two PCA components for x, y
                x, y = data[:, 0], data[:, 1]
                
                # Plot the trajectory
                ax.plot3D(x, y, steps, alpha=0.7, linewidth=2,
                        label=f'Model {model_idx+1}' if i == 0 else "")
                ax.scatter3D(x, y, steps, alpha=0.5, s=100, marker='o',
                            label=f'Model {model_idx+1}' if i == 0 else "")
                
            else:
                # If only one dimension, use time as second dimension
                x = data[:, 0]
                ax.plot3D(x, steps, steps, alpha=0.8, linewidth=2,
                        label=f'Model {model_idx+1}' if i == 0 else "")
        
        # Set labels
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2' if data.shape[1] >= 2 else 'Time Step')
        ax.set_zlabel('Time Step')
        ax.set_title(f'{type_label} in PCA Space')
        
        # Add legend to first plot only
        if i == 0:
            ax.legend(loc='upper left')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, '3d_pca_trajectories_means.png'), dpi=300, bbox_inches='tight')
        print(f"Mean PCA visualization saved to {os.path.join(output_dir, '3d_pca_trajectories_means.png')}")
    
    plt.close(fig)

def visualize_all_pca_trajectories_combined(all_projected_trajectories, output_dir=None):
    """
    Visualize PCA trajectories for all models in a single figure with subplots.
    """

    # Data types to visualize
    data_types = ['input', 'pre_synaptic', 'post_synaptic']
    type_labels = ['Input Values', 'Pre-synaptic Values', 'Post-synaptic Values']
    
    # Number of models
    n_models = len(all_projected_trajectories)
    if n_models == 0:
        print("No models to visualize")
        return
    
    # Create figure with subplots arranged by model (rows) and data type (columns)
    fig = plt.figure(figsize=(18, 5 * n_models))
    gs = gridspec.GridSpec(n_models, 3, figure=fig)
    
    # Custom colormap for different rollouts
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot for each model and data type
    for model_idx, model_trajectories in enumerate(all_projected_trajectories):
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
            ax.set_title(f'Model {model_idx+1} - {type_label}')
            
            # Add legend only to the first plot of each row
            if type_idx == 0:
                ax.legend(loc='upper left', fontsize='small')
                
            # Set consistent view angle for better comparison
            ax.view_init(elev=30, azim=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'all_models_pca_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Combined PCA visualization saved to {os.path.join(output_dir, 'all_models_pca_comparison.png')}")
    
    plt.close(fig)
    
def visualize_reward_landscape_combined(all_trajectories, output_dir=None):
    """
    Visualize reward landscape for all models in a single figure with subplots.
    """
    
    n_models = len(all_trajectories)
    if n_models == 0:
        print("No models to visualize")
        return
    # Create figure with subplots arranged by model (rows) and data type (columns)
    n_cols = 3  # number of subplots per row
    n_rows = (n_models + n_cols - 1) // n_cols  # ceil division
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    
    # Determine appropriate dimensions to visualize based on input size
    sample_input = all_trajectories[0][0]['inputs'][0]
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
    
    # Plot for each model and data type
    for model_idx, model_trajectories in enumerate(all_trajectories):
        # Extract state-reward pairs from all trajectories
        states_dim1 = []
        states_dim2 = []
        rewards = []
        for traj in model_trajectories:
            for i, (state, reward) in enumerate(zip(traj['inputs'], traj['rewards'])):
                # For some environments rewards come at the next step
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
            ax.set_title(f'Model {model_idx+1} Reward Landscape')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add colorbar
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.01)
            cbar.set_label('Reward')
            
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'reward_landscape_combined.png'), dpi=300, bbox_inches='tight')
        print(f"Combined reward landscape visualization saved to {os.path.join(output_dir, 'reward_landscape_combined.png')}")
        
    plt.close(fig)
    

def main():
    parser = argparse.ArgumentParser(description='NCHL Behavioral Analysis')
    parser.add_argument('path_file', type=str, 
                      help='Path to file containing paths to trained NCHL model files')
    parser.add_argument('output_dir', type=str, help='Output directory for results')
    parser.add_argument('--num_rollouts', type=int, default=3, help='Number of test rollouts per model')
    parser.add_argument('--n_components', type=int, default=3, help='Number of PCA components to use')
    parser.add_argument('--steps', type=int, default=1000, help='Maximum number of steps per rollout')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read trained model paths
    trained_model_paths = []
    with open(args.path_file, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            trained_model_paths.append(line + '/best_nchl.pkl')
    
    task = trained_model_paths[0].split('/')[-2].split('_')[0]
    
    # Process trained models
    print("Processing trained models...")
    all_trajectories = []
    all_projected_trajectories = []
    
    for i, model_path in enumerate(trained_model_paths):
        print(f"Analyzing model: {model_path}")
        
        # Load model
        model = load_model(model_path)
        
        # Set up environment
        env = gym.make(task)

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
        
        # Store results
        all_trajectories.append(trajectories)
        all_projected_trajectories.append(projected_trajectories)
        
        # Close environment
        env.close()
    
    # -
    # Visualize PCA trajectories for models
    visualize_all_pca_trajectories_3d(all_projected_trajectories, args.output_dir)
    
    visualize_all_pca_trajectories_means_3d(all_projected_trajectories, args.output_dir)
    
    # Visualize combined PCA trajectories
    visualize_all_pca_trajectories_combined(all_projected_trajectories, args.output_dir)
    
    # Visualize reward landscapes for models
    visualize_reward_landscape_combined(all_trajectories, args.output_dir)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()