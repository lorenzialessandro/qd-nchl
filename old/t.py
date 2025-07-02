import numpy as np
import torch
import os
import pickle
import gymnasium as gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import argparse
from pathlib import Path
from matplotlib import cm
import matplotlib.animation as animation

from network import NCHL

def load_model(model_path):
    """Load a saved NCHL model"""
    print(f"Loading model from {model_path}...")
    return NCHL.load(model_path)

def setup_environment(env_name):
    """Set up the gym environment"""
    env = gym.make(env_name)
    return env

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

def visualize_activations_3d(trajectories, output_dir=None, anim=False):
    """
    Create 3D visualizations of activations over time.
    """
    fig = plt.figure(figsize=(18, 6))
    
    titles = ['Input Values', 'Pre-synaptic Values', 'Post-synaptic Values']
    data_types = ['inputs', 'pre_synaptic', 'post_synaptic']
    
    # For input space, check what kind of environment we might be dealing with
    # and select appropriate dimensions to visualize
    sample_input = trajectories[0]['inputs'][0]
    input_dims = len(sample_input)
    
    # Determine which dimensions to visualize for input space
    if input_dims == 4:  # Likely CartPole
        # For CartPole: position, velocity, angle, angular velocity
        dim_labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Angular Velocity']
        # Visualize position vs angle vs time
        input_dims_to_plot = [0, 2]
    elif input_dims == 2:  # Likely MountainCar
        # For MountainCar: position, velocity
        dim_labels = ['Position', 'Velocity']
        input_dims_to_plot = [0, 1]
    elif input_dims == 8:  # Likely LunarLander
        # For LunarLander: x, y, vx, vy, angle, angular vel, leg1, leg2
        dim_labels = ['X', 'Y', 'VX', 'VY', 'Angle', 'Angular Vel', 'Leg1', 'Leg2']
        # Visualize x, y, angle
        input_dims_to_plot = [0, 1, 4]
    else:
        # For unknown environments, just use the first two dimensions
        dim_labels = [f'Dim{i}' for i in range(input_dims)]
        input_dims_to_plot = [0, 1] if input_dims > 1 else [0, 0]
    
    # Create 3D plots
    for i, (title, data_type) in enumerate(zip(titles, data_types)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        for traj_idx, traj in enumerate(trajectories):
            data = traj[data_type]
            steps = range(len(data))
            
            if data_type == 'inputs':
                # For input space, use the determined dimensions
                if len(input_dims_to_plot) >= 2 and input_dims > 1:
                    dim1, dim2 = input_dims_to_plot[:2]
                    x = [d[dim1] for d in data]
                    y = [d[dim2] for d in data]
                else:
                    # Fallback for 1D input
                    x = [d[0] for d in data]
                    y = [0] * len(data)  # Use zeros for second dimension
            else:
                # For pre/post-synaptic, use first two neurons if available
                if isinstance(data[0], np.ndarray) and data[0].size > 1:
                    x = [d[0] for d in data]
                    y = [d[1] if d.size > 1 else 0 for d in data]
                else:
                    x = [float(d) for d in data]
                    y = [0] * len(data)
            
            # Plot 3D trajectory
            ax.plot3D(x, y, steps, alpha=0.8, linewidth=2, 
                    label=f'Traj {traj_idx+1}' if i == 0 else "")
            
            # Add markers for start and end points
            ax.scatter3D(x[0], y[0], 0, color='green', s=100, marker='o', 
                       label='Start' if traj_idx == 0 and i == 0 else "")
            ax.scatter3D(x[-1], y[-1], len(steps)-1, color='red', s=100, marker='x',
                       label='End' if traj_idx == 0 and i == 0 else "")
            
        # Set labels
        if data_type == 'inputs':
            ax.set_xlabel(dim_labels[input_dims_to_plot[0]] if input_dims > input_dims_to_plot[0] else 'Dim 0')
            ax.set_ylabel(dim_labels[input_dims_to_plot[1]] if input_dims > input_dims_to_plot[1] else 'Dim 1')
        else:
            ax.set_xlabel('Neuron 1')
            ax.set_ylabel('Neuron 2')
        ax.set_zlabel('Time Step')
        ax.set_title(title)
        
        # Add legend to first plot only
        if i == 0:
            ax.legend(loc='upper left')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, '3d_activations.png'), dpi=300, bbox_inches='tight')
    
    # Create animation if requested
    if anim and output_dir:
        for i, (title, data_type) in enumerate(zip(titles, data_types)):
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            ax.clear()
            
            # Set viewing angle for animation
            ax.view_init(elev=30, azim=0)
            
            # Function to update the plot
            def update(frame):
                ax.view_init(elev=30, azim=frame)
                return fig,
            
            # Create the animation
            ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 5), 
                                         interval=100, blit=True)
            ani.save(os.path.join(output_dir, f'3d_{data_type}_animation.gif'), 
                   writer='pillow', fps=10, dpi=100)
    
    plt.close(fig)
    
def visualize_all_pca_trajectories_3d(all_projected_trajectories, output_dir=None):
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
    plt.close(fig)
    
    
def visualize_all_pca_trajectories_means_3d(all_projected_trajectories, output_dir=None):
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
    plt.close(fig)

def visualize_pca_trajectories_3d(projected_trajectories, output_dir=None, anim=False):
    """
    Visualize the trajectories in 3D PCA space with time as the third dimension.
    """
    fig = plt.figure(figsize=(18, 6))
    
    labels = ['Input Values', 'Pre-synaptic Values', 'Post-synaptic Values']
    data_types = ['input', 'pre_synaptic', 'post_synaptic']
    
    for i, (label, data_type) in enumerate(zip(labels, data_types)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Skip if no data
        if not projected_trajectories:
            ax.text2D(0.5, 0.5, "No trajectory data available", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax.transAxes)
            continue
        
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
                # ax.plot3D(x, y, steps, alpha=0.8, linewidth=2,
                #         label=f'Traj {traj_idx+1}' if i == 0 else "")
                ax.scatter3D(x, y, steps, alpha=0.5, s=100, marker='o',
                            label=f'Traj {traj_idx+1}' if i == 0 else "")
                
                # # Mark start and end points
                # # Start point: a black triangle (pointing up)
                # ax.scatter3D(x[0], y[0], steps[0], color='black', s=120, marker='^',
                #             label='Start' if traj_idx == 0 and i == 0 else "")

                # # End point: a red cross (X)
                # ax.scatter3D(x[-1], y[-1], steps[-1], color='red', s=120, marker='X',
                #             label='End' if traj_idx == 0 and i == 0 else "")

            else:
                # If only one dimension, use time as second dimension
                x = data[:, 0]
                ax.scatter3D(x, steps, steps, alpha=0.1, linewidth=2,
                        label=f'Traj {traj_idx+1}' if i == 0 else "") 
                
                ax.scatter3D(x[0], steps[0], steps[0], color='black', s=100, marker='o',
                           label='Start' if traj_idx == 0 and i == 0 else "")
                ax.scatter3D(x[-1], steps[-1], steps[-1], color='black', s=100, marker='x',
                           label='End' if traj_idx == 0 and i == 0 else "")
        
        # Set labels
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2' if data.shape[1] >= 2 else 'Time Step')
        ax.set_zlabel('Time Step')
        ax.set_title(f'{label} in PCA Space')
        
        # Add legend to first plot only
        if i == 0:
            ax.legend(loc='upper left')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, '3d_pca_trajectories.png'), dpi=300, bbox_inches='tight')
    
    # Create animation if requested
    if anim and output_dir:
        for i, (label, data_type) in enumerate(zip(labels, data_types)):
            # Create a separate figure for animation to avoid interference
            ani_fig = plt.figure(figsize=(8, 6))
            ax = ani_fig.add_subplot(111, projection='3d')
            
            # Skip if no data
            if not projected_trajectories or not projected_trajectories[0][data_type]:
                plt.close(ani_fig)
                continue
            
            # Get the data for the first trajectory
            traj = projected_trajectories[0]
            data = np.array(traj[data_type])
            steps = traj['steps']
            
            # Check if we have enough dimensions
            if data.shape[1] >= 2:
                x, y = data[:, 0], data[:, 1]
                
                # Function to update the plot for animation
                def update(frame):
                    ax.clear()
                    ax.view_init(elev=30, azim=frame)
                    ax.plot3D(x, y, steps, alpha=0.8, linewidth=2)
                    ax.scatter3D(x[0], y[0], steps[0], color='green', s=100, marker='o')
                    ax.scatter3D(x[-1], y[-1], steps[-1], color='red', s=100, marker='x')
                    ax.set_xlabel('PC1')
                    ax.set_ylabel('PC2')
                    ax.set_zlabel('Time Step')
                    ax.set_title(f'{label} in PCA Space')
                    return ax,
                
                # Create the animation
                ani = animation.FuncAnimation(ani_fig, update, frames=np.arange(0, 360, 5),
                                           interval=100, blit=False)
                ani.save(os.path.join(output_dir, f'3d_pca_{data_type}_animation.gif'),
                       writer='pillow', fps=10, dpi=100)
            
            plt.close(ani_fig)
    
    plt.close(fig)

def visualize_reward_landscape(trajectories, output_dir=None):
    """
    Visualize the reward landscape over state space.
    """
    # Check if we have trajectories with rewards
    if not trajectories or 'rewards' not in trajectories[0]:
        return
    
    fig = plt.figure(figsize=(12, 6))
    
    # Determine appropriate dimensions to visualize based on input size
    sample_input = trajectories[0]['inputs'][0]
    input_dims = len(sample_input)
    
    # Select dimensions to plot based on environment
    if input_dims == 4:  # Likely CartPole
        dim1, dim2 = 0, 2  # Position vs Angle
        x_label, y_label = 'Cart Position', 'Pole Angle'
    elif input_dims == 2:  # Likely MountainCar
        dim1, dim2 = 0, 1  # Position vs Velocity
        x_label, y_label = 'Position', 'Velocity'
    elif input_dims == 8:  # Likely LunarLander
        dim1, dim2 = 0, 1  # X vs Y position
        x_label, y_label = 'X Position', 'Y Position'
    else:
        # For unknown environments, just use the first two dimensions
        dim1, dim2 = 0, min(1, input_dims-1)
        x_label, y_label = 'Dimension 1', 'Dimension 2'
    
    # Extract state-reward pairs from all trajectories
    states_dim1 = []
    states_dim2 = []
    rewards = []
    
    for traj in trajectories:
        for i, (state, reward) in enumerate(zip(traj['inputs'], traj['rewards'])):
            # For some environments rewards come at the next step
            if input_dims > max(dim1, dim2):
                states_dim1.append(state[dim1])
                states_dim2.append(state[dim2])
                rewards.append(reward)
    
    # If we have enough data points, create a scatter plot
    if states_dim1:
        ax = fig.add_subplot(111)
        sc = ax.scatter(states_dim1, states_dim2, c=rewards, cmap='viridis', 
                      alpha=0.6, s=30, edgecolors='none')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title('Reward Landscape')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add colorbar
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.01)
        cbar.set_label('Reward')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'reward_landscape.png'), dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def analyze_model(model_path, env_name, output_dir, args):
    """
    Analyze a single model and generate visualizations.
    """
    # Create model-specific output directory
    model_name = Path(model_path).stem
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    print(f"Analyzing model: {model_path}")
    
    # Load model
    model = load_model(model_path)
    
    # Set up environment
    env = setup_environment(env_name)
    
    # Run test rollouts
    print("Running test rollouts...")
    trajectories = run_test_rollouts(model, env, num_rollouts=args.num_rollouts, 
                                   steps=args.steps, debug=args.debug)
    
    # Visualize reward landscape
    print("Visualizing reward landscape...")
    visualize_reward_landscape(trajectories, model_output_dir)
    
    # Perform PCA analysis
    print("Performing PCA analysis...")
    results, input_pcas, pre_pcas, post_pcas = perform_pca_analysis(
        trajectories, n_components=args.n_components, debug=args.debug)
    
    # Store PCAs for projection
    pcas = {
        'input': input_pcas,
        'pre_synaptic': pre_pcas,
        'post_synaptic': post_pcas
    }
    
    # Visualize activations in 3D
    print("Visualizing 3D activations...")
    visualize_activations_3d(trajectories, model_output_dir, anim=args.animate)
    
    # Project trajectories onto PCA space
    print("Projecting trajectories onto PCA space...")
    projected_trajectories = project_trajectories_on_pca(
        trajectories, pcas, n_components=args.n_components, debug=args.debug)
    
    # Visualize PCA trajectories in 3D
    print("Visualizing 3D PCA trajectories...")
    visualize_pca_trajectories_3d(projected_trajectories, model_output_dir, anim=args.animate)
    
    # Close environment
    env.close()
    
    print(f"Analysis of {model_path} complete. Results saved to {model_output_dir}")

def calculate_and_visualize_average_pca(all_projected_trajectories, output_dir=None, anim=False):
    """
    Calculate and visualize the average PCA trajectories across all models and rollouts.
    
    Parameters:
    -----------
    all_projected_trajectories : list of lists
        A list where each element is the list of projected_trajectories for a model
    output_dir : str, optional
        Directory to save visualizations
    anim : bool, optional
        Whether to create animations
        
    Returns:
    --------
    dict
        Dictionary containing average trajectories for each data type
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    
    print("Calculating average PCA trajectories across all models and rollouts...")
    
    # Step 1: Find the maximum common trajectory length
    # We need to align all trajectories to calculate proper averages
    max_common_steps = float('inf')
    all_flattened_trajectories = []
    
    # Flatten the nested list and find the minimum trajectory length
    for model_trajectories in all_projected_trajectories:
        for traj in model_trajectories:
            if traj['steps']:
                max_common_steps = min(max_common_steps, len(traj['steps']))
                all_flattened_trajectories.append(traj)
    
    if max_common_steps == float('inf') or not all_flattened_trajectories:
        print("No valid trajectories found for averaging.")
        return None
    
    print(f"Found {len(all_flattened_trajectories)} total trajectories across all models")
    print(f"Using common trajectory length of {max_common_steps} steps")
    
    # Step 2: Initialize structures for average trajectories
    average_trajectories = {
        'input': np.zeros((max_common_steps, 3)),  # 3 for the 3 PCA components
        'pre_synaptic': np.zeros((max_common_steps, 3)),
        'post_synaptic': np.zeros((max_common_steps, 3)),
        'steps': list(range(max_common_steps)),
        'num_trajectories': np.zeros(max_common_steps)  # Count trajectories per step for proper averaging
    }
    
    # Step 3: Accumulate all trajectory data
    for traj in all_flattened_trajectories:
        # Only process up to the common number of steps
        valid_steps = min(len(traj['steps']), max_common_steps)
        
        for step_idx in range(valid_steps):
            # Update counter for proper averaging
            average_trajectories['num_trajectories'][step_idx] += 1
            
            # Accumulate PCA values for each data type
            for data_type in ['input', 'pre_synaptic', 'post_synaptic']:
                if data_type in traj and len(traj[data_type]) > step_idx:
                    # Get the PCA components (up to 3)
                    components = np.array(traj[data_type][step_idx])
                    
                    # Ensure we have at most 3 components
                    if len(components) > 3:
                        components = components[:3]
                    elif len(components) < 3:
                        # Pad with zeros if fewer than 3 components
                        components = np.pad(components, (0, 3 - len(components)))
                    
                    # Add to the accumulator
                    average_trajectories[data_type][step_idx] += components
    
    # Step 4: Calculate averages
    for step_idx in range(max_common_steps):
        if average_trajectories['num_trajectories'][step_idx] > 0:
            count = average_trajectories['num_trajectories'][step_idx]
            for data_type in ['input', 'pre_synaptic', 'post_synaptic']:
                average_trajectories[data_type][step_idx] /= count
    
    # Step 5: Visualize average trajectories
    if output_dir:
        fig = plt.figure(figsize=(18, 6))
        labels = ['Input Values', 'Pre-synaptic Values', 'Post-synaptic Values']
        data_types = ['input', 'pre_synaptic', 'post_synaptic']
        
        for i, (label, data_type) in enumerate(zip(labels, data_types)):
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            
            # Extract data
            data = average_trajectories[data_type]
            steps = average_trajectories['steps']
            
            # Plot the trajectory in 3D space (PC1, PC2, Time)
            x, y = data[:, 0], data[:, 1]
            # ax.scatter3D(x, y, steps, alpha=0.8, s=100, marker='o',
            #               label=f'Average {label}' if i == 0 else "")
            ax.plot3D(x, y, steps, alpha=0.8, linewidth=2, color='blue', 
                     label=f'Average Trajectory')
            
            # Mark start and end points
            ax.scatter3D(x[0], y[0], steps[0], color='green', s=100, marker='o',
                       label='Start')
            ax.scatter3D(x[-1], y[-1], steps[-1], color='red', s=100, marker='x',
                       label='End')
            
            # Set labels
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('Time Step')
            ax.set_title(f'Average {label} in PCA Space')
            ax.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_pca_trajectories.png'), dpi=300, bbox_inches='tight')
        
        # Create 2D projection of average trajectories
        fig2 = plt.figure(figsize=(18, 6))
        for i, (label, data_type) in enumerate(zip(labels, data_types)):
            ax = fig2.add_subplot(1, 3, i+1)
            
            # Extract data
            data = average_trajectories[data_type]
            
            # Plot PC1 vs PC2
            x, y = data[:, 0], data[:, 1]
            
            # Use colormap to indicate time progression
            points = ax.scatter(x, y, c=range(len(x)), cmap='viridis', 
                              alpha=0.8, s=30, edgecolors='none')
            
            # Plot trajectory line
            ax.plot(x, y, alpha=0.5, linewidth=1, color='gray')
            
            # Mark start and end points
            ax.scatter(x[0], y[0], color='green', s=100, marker='o', label='Start')
            ax.scatter(x[-1], y[-1], color='red', s=100, marker='x', label='End')
            
            # Set labels
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(f'Average {label} in PCA Space (PC1 vs PC2)')
            ax.legend(loc='upper left')
            
            # Add colorbar to show time progression
            cbar = plt.colorbar(points, ax=ax)
            cbar.set_label('Time Step')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_pca_trajectories_2d.png'), dpi=300, bbox_inches='tight')
        
        # Create animation if requested
        if anim:
            for i, (label, data_type) in enumerate(zip(labels, data_types)):
                # Create a separate figure for animation
                ani_fig = plt.figure(figsize=(8, 6))
                ax = ani_fig.add_subplot(111, projection='3d')
                
                # Extract data
                data = average_trajectories[data_type]
                steps = average_trajectories['steps']
                x, y = data[:, 0], data[:, 1]
                
                # Function to update the plot for animation
                def update(frame):
                    ax.clear()
                    ax.view_init(elev=30, azim=frame)
                    ax.plot3D(x, y, steps, alpha=0.8, linewidth=2, color='blue')
                    ax.scatter3D(x[0], y[0], steps[0], color='green', s=100, marker='o')
                    ax.scatter3D(x[-1], y[-1], steps[-1], color='red', s=100, marker='x')
                    ax.set_xlabel('PC1')
                    ax.set_ylabel('PC2')
                    ax.set_zlabel('Time Step')
                    ax.set_title(f'Average {label} in PCA Space')
                    return ax,
                
                # Create the animation
                ani = animation.FuncAnimation(ani_fig, update, frames=np.arange(0, 360, 5),
                                           interval=100, blit=False)
                ani.save(os.path.join(output_dir, f'average_pca_{data_type}_animation.gif'),
                       writer='pillow', fps=10, dpi=100)
                
                plt.close(ani_fig)
        
        plt.close(fig)
        plt.close(fig2)
    
    return average_trajectories

def visualize_all_pca_trajectories_combined(all_projected_trajectories, model_names, output_dir=None, n_components=3):
    """
    Visualize PCA trajectories for all models in a single figure with subplots.
    
    Parameters:
    -----------
    all_projected_trajectories : list
        List of projected trajectories for each model
    model_names : list
        List of model names/identifiers
    output_dir : str, optional
        Directory to save visualizations
    n_components : int, optional
        Number of PCA components to visualize (default: 3)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec
    
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
    for model_idx, (model_trajectories, model_name) in enumerate(zip(all_projected_trajectories, model_names)):
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
            ax.set_title(f'{model_name} - {type_label}')
            
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
    
    # Create a 2D version (PC1 vs PC2) for easier comparison
    fig2 = plt.figure(figsize=(18, 5 * n_models))
    gs2 = gridspec.GridSpec(n_models, 3, figure=fig2)
    
    for model_idx, (model_trajectories, model_name) in enumerate(zip(all_projected_trajectories, model_names)):
        for type_idx, (data_type, type_label) in enumerate(zip(data_types, type_labels)):
            # Create 2D subplot
            ax = fig2.add_subplot(gs2[model_idx, type_idx])
            
            # Skip if no trajectories
            if not model_trajectories:
                ax.text(0.5, 0.5, "No trajectory data available", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax.transAxes)
                continue
            
            # Plot each trajectory for this model
            for traj_idx, traj in enumerate(model_trajectories):
                if data_type not in traj or not traj[data_type]:
                    continue
                    
                # Extract data
                data = np.array(traj[data_type])
                
                # Check if we have enough dimensions
                if data.shape[1] >= 2:
                    # Use first two PCA components
                    x, y = data[:, 0], data[:, 1]
                    
                    # Plot the trajectory - use a different color for each rollout
                    color_idx = traj_idx % len(colors)
                    ax.plot(x, y, alpha=0.7, linewidth=2,
                          color=colors[color_idx], label=f'Rollout {traj_idx+1}')
                    
                    # Mark start point (green circle)
                    ax.scatter(x[0], y[0], color='green', s=80, marker='o',
                             label='Start' if traj_idx == 0 else "")
                    
                    # Mark end point (red cross)
                    ax.scatter(x[-1], y[-1], color='red', s=80, marker='x',
                             label='End' if traj_idx == 0 else "")
                else:
                    # If only one dimension, use step index as second dimension
                    x = data[:, 0]
                    y = np.arange(len(x))
                    color_idx = traj_idx % len(colors)
                    ax.plot(x, y, alpha=0.7, linewidth=2,
                          color=colors[color_idx], label=f'Rollout {traj_idx+1}')
            
            # Set labels and title
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2' if data.shape[1] >= 2 else 'Time Step')
            
            # Create title with model name and data type
            ax.set_title(f'{model_name} - {type_label}')
            
            # Add legend only to the first plot of each row
            if type_idx == 0:
                ax.legend(loc='upper left', fontsize='small')
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'all_models_pca_comparison_2d.png'), dpi=300, bbox_inches='tight')
        print(f"Combined 2D PCA visualization saved to {os.path.join(output_dir, 'all_models_pca_comparison_2d.png')}")
    
    plt.close(fig2)

def main_with_average_pca():
    parser = argparse.ArgumentParser(description='NCHL Behavioral Analysis with 3D Visualization')
    parser.add_argument('--path_file', type=str, required=True, help='Paths to NCHL model files')
    parser.add_argument('--env', type=str, required=True, help='Gym environment name')
    parser.add_argument('--num_rollouts', type=int, default=3, help='Number of test rollouts per model')
    parser.add_argument('--steps', type=int, default=1000, help='Maximum number of steps per rollout')
    parser.add_argument('--n_components', type=int, default=2, help='Number of PCA components')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--animate', action='store_true', help='Create animations of 3D visualizations')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_paths = []
    with open(args.path_file, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            model_paths.append(line + '/best_nchl.pkl')
    
    # List to collect all projected trajectories across models
    all_projected_trajectories = []
    model_names = []
    
    # Process each model
    for i, model_path in enumerate(model_paths):
        model_names.append(os.path.basename(model_path))
        # output_dir = os.path.join(args.output_dir, f'model_{i+1}')
        # os.makedirs(output_dir, exist_ok=True)
        
        print(f"Analyzing model: {model_path}")
        
        # Load model
        model = load_model(model_path)
        
        # Set up environment
        env = setup_environment(args.env)
        
        # Run test rollouts
        print("Running test rollouts...")
        trajectories = run_test_rollouts(model, env, num_rollouts=args.num_rollouts, 
                                       steps=args.steps, debug=args.debug)
        
        # Visualize reward landscape
        # print("Visualizing reward landscape...")
        # visualize_reward_landscape(trajectories, output_dir)
        
        # Perform PCA analysis
        print("Performing PCA analysis...")
        results, input_pcas, pre_pcas, post_pcas = perform_pca_analysis(
            trajectories, n_components=args.n_components, debug=args.debug)
        
        # Store PCAs for projection
        pcas = {
            'input': input_pcas,
            'pre_synaptic': pre_pcas,
            'post_synaptic': post_pcas
        }
        
        # Visualize activations in 3D
        # print("Visualizing 3D activations...")
        #  visualize_activations_3d(trajectories, output_dir, anim=args.animate)
        
        # Project trajectories onto PCA space
        print("Projecting trajectories onto PCA space...")
        projected_trajectories = project_trajectories_on_pca(
            trajectories, pcas, n_components=args.n_components, debug=args.debug)
        
        # Visualize PCA trajectories in 3D
        # print("Visualizing 3D PCA trajectories...")
        # visualize_pca_trajectories_3d(projected_trajectories, output_dir, anim=args.animate)
        
        # Add projected trajectories to the collection
        all_projected_trajectories.append(projected_trajectories)
        
        # Close environment
        env.close()
        
        # print(f"Analysis of {model_path} complete. Results saved to {output_dir}")
        
    # Combine all projected trajectories for visualization
    print("Creating combined PCA visualization for all models...")
    visualize_all_pca_trajectories_combined(all_projected_trajectories, model_names, args.output_dir, args.n_components)
    
    # Calculate and visualize average PCA trajectories across all models
    print("Calculating and visualizing average PCA trajectories...")
    average_trajectories = calculate_and_visualize_average_pca(
        all_projected_trajectories, args.output_dir, anim=args.animate)
    
    print("Behavioral analysis complete!")
    
def visualize_reward_landscape_combined(all_trajectories, model_names, output_dir=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec
    
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
        dim1, dim2 = 0, 2  # Position vs Angle
        x_label, y_label = 'Cart Position', 'Pole Angle'
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
    for model_idx, (model_trajectories, model_name) in enumerate(zip(all_trajectories, model_names)):
        # Extract state-reward pairs from all trajectories
        states_dim1 = []
        states_dim2 = []
        rewards = []
        for traj in model_trajectories:
            for i, (state, reward) in enumerate(zip(traj['inputs'], traj['rewards'])):
                # For some environments rewards come at the next step
                states_dim1.append(state[0])
                states_dim2.append(state[1])
                rewards.append(reward)
        # If we have enough data points, create a scatter plot
        if states_dim1:
            ax = fig.add_subplot(gs[model_idx // n_cols, model_idx % n_cols])
            sc = ax.scatter(states_dim1, states_dim2, c=rewards, cmap='viridis', 
                          alpha=0.6, s=30, edgecolors='none')
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f'{model_name} - Reward Landscape')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add colorbar
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.01)
            cbar.set_label('Reward')
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'reward_landscape_combined.png'), dpi=300, bbox_inches='tight')
        print(f"Combined reward landscape visualization saved to {os.path.join(output_dir, 'reward_landscape_combined.png')}")
        
    plt.close(fig)
    

def visualize_model_vs_random_comparison(trained_trajectories, random_trajectories, 
                               trained_names, random_names, output_dir=None):
    """
    Visualize comparison between trained models and random models in the same plot.
    
    Parameters:
    -----------
    trained_trajectories : list
        List of projected trajectories for trained models
    random_trajectories : list
        List of projected trajectories for random models
    trained_names : list
        Names of trained models
    random_names : list
        Names of random models
    output_dir : str, optional
        Directory to save visualizations
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from mpl_toolkits.mplot3d import Axes3D
    
    # Data types to visualize
    data_types = ['input', 'pre_synaptic', 'post_synaptic']
    type_labels = ['Input Values', 'Pre-synaptic Values', 'Post-synaptic Values']
    
    # Create figure with 3 subplots (one for each data type)
    fig = plt.figure(figsize=(18, 6))
    
    # Calculate average trajectories for trained models
    trained_avg = calculate_average_trajectories(trained_trajectories)
    
    # Calculate average trajectories for random models
    random_avg = calculate_average_trajectories(random_trajectories)
    
    # Plot each data type
    for i, (data_type, type_label) in enumerate(zip(data_types, type_labels)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Skip if no data available
        # if trained_avg is None or random_avg is None:
        #     ax.text2D(0.5, 0.5, "Insufficient data for comparison",
        #              horizontalalignment='center', verticalalignment='center',
        #              transform=ax.transAxes)
        #     continue
        
        # Plot average trajectory for trained models
        if data_type in trained_avg and len(trained_avg[data_type]) > 0:
            data = trained_avg[data_type]
            steps = trained_avg['steps']
            
            if data.shape[1] >= 2:  # If we have at least 2 PCA components
                x, y = data[:, 0], data[:, 1]
                # ax.plot3D(x, y, steps, alpha=0.8, linewidth=2, color='blue',
                #         label='Trained Models (Avg)')
                ax.scatter3D(x, y, steps, alpha=0.5, s=100, marker='o',
                            label='Trained Models (Avg)')
                
                # Mark start and end points
                # ax.scatter3D(x[0], y[0], steps[0], color='blue', s=100, marker='o',
                #            label='Trained Start')
                # ax.scatter3D(x[-1], y[-1], steps[-1], color='blue', s=100, marker='x',
                #            label='Trained End')
        
        # Plot average trajectory for random models
        if data_type in random_avg and len(random_avg[data_type]) > 0:
            data = random_avg[data_type]
            steps = random_avg['steps']
            
            if data.shape[1] >= 2:  # If we have at least 2 PCA components
                x, y = data[:, 0], data[:, 1]
                # ax.plot3D(x, y, steps, alpha=0.8, linewidth=2, color='red',
                #         label='Random Models (Avg)')
                ax.scatter3D(x, y, steps, alpha=0.5, s=100, marker='o',
                            label='Random Models (Avg)')
                
                # Mark start and end points
                # ax.scatter3D(x[0], y[0], steps[0], color='red', s=100, marker='o',
                #            label='Random Start')
                # ax.scatter3D(x[-1], y[-1], steps[-1], color='red', s=100, marker='x',
                #            label='Random End')
        
        # Set labels
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('Time Step')
        ax.set_title(f'Average {type_label} in PCA Space\nTrained vs Random Models')
        ax.legend(loc='upper left')
        
        # Set consistent view angle
        ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'trained_vs_random_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Trained vs random comparison saved to {os.path.join(output_dir, 'trained_vs_random_comparison.png')}")
    
    plt.close(fig)
    
    
    # Create 2D version (PC1 vs PC2) for easier interpretation
    fig2 = plt.figure(figsize=(18, 6))
    
    for i, (data_type, type_label) in enumerate(zip(data_types, type_labels)):
        ax = fig2.add_subplot(1, 3, i+1)
        
        # Skip if no data available
        if trained_avg is None or random_avg is None:
            ax.text(0.5, 0.5, "Insufficient data for comparison",
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
            continue
        
        # Plot average trajectory for trained models
        if data_type in trained_avg and len(trained_avg[data_type]) > 0:
            data = trained_avg[data_type]
            
            if data.shape[1] >= 2:  # If we have at least 2 PCA components
                x, y = data[:, 0], data[:, 1]
                
                # Use colormap to show time progression
                points = ax.scatter(x, y, c=range(len(x)), cmap='Blues', 
                                 alpha=0.8, s=30, edgecolors='none')
                
                # Plot trajectory line
                ax.plot(x, y, alpha=0.8, linewidth=2, color='blue',
                      label='Trained Models (Avg)')
                
                # Mark start and end points
                ax.scatter(x[0], y[0], color='blue', s=100, marker='o',
                         label='Trained Start')
                ax.scatter(x[-1], y[-1], color='blue', s=100, marker='x',
                         label='Trained End')
        
        # Plot average trajectory for random models
        if data_type in random_avg and len(random_avg[data_type]) > 0:
            data = random_avg[data_type]
            
            if data.shape[1] >= 2:  # If we have at least 2 PCA components
                x, y = data[:, 0], data[:, 1]
                
                # Use colormap to show time progression
                points = ax.scatter(x, y, c=range(len(x)), cmap='Reds', 
                                 alpha=0.8, s=30, edgecolors='none')
                
                # Plot trajectory line
                ax.plot(x, y, alpha=0.8, linewidth=2, color='red',
                      label='Random Models (Avg)')
                
                # Mark start and end points
                ax.scatter(x[0], y[0], color='red', s=100, marker='o',
                         label='Random Start')
                ax.scatter(x[-1], y[-1], color='red', s=100, marker='x',
                         label='Random End')
        
        # Set labels
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Average {type_label} in PCA Space (PC1 vs PC2)\nTrained vs Random Models')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'trained_vs_random_comparison_2d.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"2D trained vs random comparison saved to {os.path.join(output_dir, 'trained_vs_random_comparison_2d.png')}")
    
    plt.close(fig2)


def calculate_average_trajectories(all_trajectories_list):
    """
    Calculate average trajectories across all models and rollouts.
    
    Parameters:
    -----------
    all_trajectories_list : list of lists
        A list where each element is the list of projected_trajectories for a model
        
    Returns:
    --------
    dict
        Dictionary containing average trajectories for each data type
    """
    import numpy as np
    
    # Flatten the nested list of trajectories
    all_flattened_trajectories = []
    for model_trajectories in all_trajectories_list:
        for traj in model_trajectories:
            if 'steps' in traj and traj['steps']:
                all_flattened_trajectories.append(traj)
    
    if not all_flattened_trajectories:
        print("No valid trajectories found for averaging.")
        return None
    
    # Find the maximum common trajectory length
    max_common_steps = float('inf')
    for traj in all_flattened_trajectories:
        if traj['steps']:
            max_common_steps = min(max_common_steps, len(traj['steps']))
    
    if max_common_steps == float('inf'):
        print("No valid steps found in trajectories.")
        return None
    
    # Initialize structures for average trajectories
    average_trajectories = {
        'input': np.zeros((max_common_steps, 3)),
        'pre_synaptic': np.zeros((max_common_steps, 3)),
        'post_synaptic': np.zeros((max_common_steps, 3)),
        'steps': list(range(max_common_steps)),
        'num_trajectories': np.zeros(max_common_steps)
    }
    
    # Accumulate all trajectory data
    for traj in all_flattened_trajectories:
        valid_steps = min(len(traj['steps']), max_common_steps)
        
        for step_idx in range(valid_steps):
            # Update counter for proper averaging
            average_trajectories['num_trajectories'][step_idx] += 1
            
            # Accumulate PCA values for each data type
            for data_type in ['input', 'pre_synaptic', 'post_synaptic']:
                if data_type in traj and len(traj[data_type]) > step_idx:
                    # Get the PCA components (up to 3)
                    components = np.array(traj[data_type][step_idx])
                    
                    # Ensure we have at most 3 components
                    if len(components) > 3:
                        components = components[:3]
                    elif len(components) < 3:
                        # Pad with zeros if fewer than 3 components
                        components = np.pad(components, (0, 3 - len(components)))
                    
                    # Add to the accumulator
                    average_trajectories[data_type][step_idx] += components
    
    # Calculate averages
    for step_idx in range(max_common_steps):
        if average_trajectories['num_trajectories'][step_idx] > 0:
            count = average_trajectories['num_trajectories'][step_idx]
            for data_type in ['input', 'pre_synaptic', 'post_synaptic']:
                average_trajectories[data_type][step_idx] /= count
    
    return average_trajectories


def modified_main_function():
    parser = argparse.ArgumentParser(description='NCHL Behavioral Analysis with Model vs Random Comparison')
    parser.add_argument('--trained_path_file', type=str, required=True, 
                      help='Path to file containing paths to trained NCHL model files')
    parser.add_argument('--random_path_file', type=str, required=True, 
                      help='Path to file containing paths to random NCHL model files')
    parser.add_argument('--env', type=str, required=True, help='Gym environment name')
    parser.add_argument('--num_rollouts', type=int, default=3, help='Number of test rollouts per model')
    parser.add_argument('--steps', type=int, default=1000, help='Maximum number of steps per rollout')
    parser.add_argument('--n_components', type=int, default=2, help='Number of PCA components')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--animate', action='store_true', help='Create animations of 3D visualizations')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read trained model paths
    trained_model_paths = []
    with open(args.trained_path_file, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            trained_model_paths.append(line + '/best_nchl.pkl')
    
    # Read random model paths
    random_model_paths = []
    with open(args.random_path_file, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            random_model_paths.append(line + '/best_nchl.pkl')
    
    # Process trained models
    print("Processing trained models...")
    trained_projected_trajectories = []
    trained_model_names = []
    roll_t = []
    
    for i, model_path in enumerate(trained_model_paths):
        trained_model_names.append(os.path.basename(model_path))
        print(f"Analyzing trained model: {model_path}")
        
        # Load model
        model = load_model(model_path)
        
        # Set up environment
        env = setup_environment(args.env)
        
        # Run test rollouts
        trajectories = run_test_rollouts(model, env, num_rollouts=args.num_rollouts, 
                                       steps=args.steps, debug=args.debug)
        
        visualize_activations_3d(trajectories, args.output_dir, anim=args.animate)
        exit()
        roll_t.append(trajectories)
        
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
        
        # Add to collection
        trained_projected_trajectories.append(projected_trajectories)
        
        # Close environment
        env.close()
    
    # Process random models
    print("Processing random models...")
    random_projected_trajectories = []
    random_model_names = []
    
    for i, model_path in enumerate(random_model_paths):
        random_model_names.append(os.path.basename(model_path))
        print(f"Analyzing random model: {model_path}")
        
        # Load model
        model = load_model(model_path)
        
        # Set up environment
        env = setup_environment(args.env)
        
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
        
        # Add to collection
        random_projected_trajectories.append(projected_trajectories)
        
        # Close environment
        env.close()
        
    visualize_all_pca_trajectories_3d(trained_projected_trajectories,  
                                      args.output_dir)
    
    visualize_all_pca_trajectories_means_3d(trained_projected_trajectories,  
                                      args.output_dir)
    
    exit()
    
    # Visualize reward landscapes for trained and random models
    print("Creating combined reward landscape visualization...")
    visualize_reward_landscape_combined(roll_t, trained_model_names, args.output_dir)
    
    # # Visualize trained models
    # print("Creating trained models visualization...")
    # # os.makedirs(os.path.join(args.output_dir, 'trained_models'), exist_ok=True)
    visualize_all_pca_trajectories_combined(trained_projected_trajectories, 
                                         trained_model_names, 
                                         args.output_dir,
                                         args.n_components)
    exit()
    
    # # Visualize random models
    # print("Creating random models visualization...")
    # os.makedirs(os.path.join(args.output_dir, 'random_models'), exist_ok=True)
    # visualize_all_pca_trajectories_combined(random_projected_trajectories, 
    #                                      random_model_names, 
    #                                      os.path.join(args.output_dir, 'random_models'),
    #                                      args.n_components)
    
    # Compare trained vs random models
    print("Creating trained vs random models comparison...")
    os.makedirs(os.path.join(args.output_dir, 'comparison'), exist_ok=True)
    visualize_model_vs_random_comparison(trained_projected_trajectories, 
                                      random_projected_trajectories,
                                      trained_model_names,
                                      random_model_names,
                                      args.output_dir)
    
    print("Analysis complete!")


if __name__ == "__main__":
    modified_main_function()