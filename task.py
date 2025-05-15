import sys
import os
import gymnasium as gym
import numpy as np
import torch
import random
import json
import wandb
from multiprocessing import Pool

from network import NCHL, Neuron
from optimizer import MapElites
from utils import *
from analysis_archive import plot_archive_analysis, visualize_archive 

def evaluate(data):
    params = data[0]
    args = data[1]
    nodes = args["nodes"]
    task = args["task"]

    env = gym.make(task)
    agent = NCHL(nodes) # grad=False
    agent.set_params(params)
    
    # Get environment action space information
    action_space = env.action_space
    is_continuous = isinstance(action_space, gym.spaces.Box)
    
    rews = []
    state, _ = env.reset(seed=args.get("seed", None))
    done = False
    truncated = False
    rew_ep = 0
    step_count = 0
    max_steps = args.get("max_steps", 1000)  # Default max steps
    
    # Early stopping conditions for Ant environment
    # consecutive_negative_rewards = 0
    # negative_reward_threshold = args.get("negative_reward_threshold", 30)

    while not (done or truncated) and step_count < max_steps:
        input_tensor = torch.tensor(state, dtype=torch.float32)
        output = agent.forward(input_tensor)
        agent.update_weights()

        if is_continuous:
            # For continuous action spaces (like Ant-v5)
            # Actions are already in the range [-1, 1] due to tanh activation
            action = output.detach().cpu().numpy().flatten()
            
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
            action = np.argmax(output.detach().cpu().numpy())
        
        state, reward, done, truncated, info = env.step(action)

        rew_ep += reward
        rews.append(reward)
        step_count += 1
        
        # Early stopping for Ant to prevent wasting computation on poor solutions
        # if "Ant" in task:
        #     if reward < 0:
        #         consecutive_negative_rewards += 1
        #         if consecutive_negative_rewards > negative_reward_threshold:
        #             break
        #     else:
        #         consecutive_negative_rewards = 0
    
    env.close()
    
    # Compute descriptors
    act_diversity, weight_diversity = agent.get_descriptors()
    
    return rew_ep, (act_diversity, weight_diversity), agent

def parallel_val(candidates, args):
    with Pool() as p:
        return p.map(evaluate, [[c, json.loads(json.dumps(args))] for c in candidates])
    
def launcher(config):
    print(config)
    if config["wandb"]:
        wb = wandb.init(project="qd-nchl-runs", name=f"{config['task']}_{config['seed']}", config=config)
        
    os.makedirs(config["path_dir"], exist_ok=True)
    # path_pkl = f"{config['path_dir']}/pkl"
    # os.makedirs(path_pkl, exist_ok=True)
    
    # Extract MapElites parameters from the config
    map_elites_params = {
        'seed': config['seed'],
        'map_size': config['map_size'],
        'length': config['length'],
        'pop_size': config['pop_size'],
        'bounds': config['bounds'],
        'threshold': config['threshold'],
        'path_dir': config['path_dir'],
        'sigma': config['sigma'],
        'use_elitism': config['use_elitism'],
        'use_crossover': config['use_crossover'],
        'use_adaptive_mutation': config['use_adaptive_mutation'],
        'use_prob_sampling': config['use_prob_sampling'],
    }

    archive = MapElites(**map_elites_params)
    
    history_best_fitnesses = []
    history_avg_fitnesses = []
    logs = []
    
    global_best_fitness = float("-inf")
    
    for i in range(config["iterations"]):
        candidates = archive.ask() # ask for new candidates
        
        res = parallel_val(candidates, config)
        fitnesses = [r[0] for r in res]
        descriptors = [r[1] for r in res]
        agents = [r[2] for r in res]
        
        archive.tell(candidates, fitnesses, descriptors) # tell the archive about the new candidates
        
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > global_best_fitness:
            # save the best agent 
            best_agent = agents[best_idx]
            best_agent.save(path_dir=config["path_dir"])
        
        global_best_fitness = max(global_best_fitness, archive.best_fitness)
        
        log = "iteration " + str(i) + "  " + str(max(fitnesses)) + "  " + str(np.mean(fitnesses))
        if config["wandb"]:
            wandb.log({"iteration": i, "max_fitness": max(fitnesses), "avg_fitness": np.mean(fitnesses), "global_best_fitness": global_best_fitness, "coverage": archive.coverage})
        history_best_fitnesses.append(max(fitnesses))
        history_avg_fitnesses.append(np.mean(fitnesses))
        print(log)
        logs.append(log)
        
        
        # if i % 50 == 0:
        #     path_it = f"{config['path_dir']}/{i}"
        #     os.makedirs(path_it, exist_ok=True)
        #     visualize_archive(archive, path_dir=path_it, cmap="Greens", annot=False, high=False)
            # plot_history(history_avg_fitnesses, history_best_fitnesses, path_dir=path_it)
            
      
    # Get the best individual from the archive  
    best_ind, best_fit, best_desc = archive.get_best()
    print("Best fitness: ", best_fit)
    print("Best descriptor: ", best_desc)
    print("Best individual: ", best_ind)
    log = "Best fitness: " + str(best_fit) + "   Best descriptor: " + str(best_desc) + "   Best individual: " + str(best_ind)
    logs.append(log)
    # best_net = NCHL(nodes=config["nodes"])
    # best_net.set_params(best_ind)
    # best_net.save(path_dir=config["path_dir"]) # save the best individual
    
    archive.save() # save the archive
    save_log(logs, path_dir=config["path_dir"]) # save the logs
    
    # Create final visualizations
    visualize_archive(archive, path_dir=config["path_dir"], cmap="Greens", annot=False, high=False)
    plot_history(history_avg_fitnesses, history_best_fitnesses, path_dir=config["path_dir"]) 
    
    # # Search for the best model in the pkl directory
    # pkl_path = config["path_dir"] + "/pkl"
    # best_model_name = None
    # best_fitness = -np.inf
    # for file in os.listdir(pkl_path):
    #     if file.endswith(".pkl"):
    #         try:
    #             iteration, fitness = map(float, file[:-4].split("_")) 
    #             if fitness > best_fitness:
    #                 best_fitness = fitness
    #                 best_model_name = file
    #         except ValueError:
    #             continue
    # print(f"Best model name: {best_model_name}")
    
    # # Remove all other models in the pkl directory
    # for file in os.listdir(pkl_path):
    #     if file.endswith(".pkl") and file != best_model_name:
    #         os.remove(os.path.join(pkl_path, file))
    
    # # Save the best model in the main directory
    # best_net.save(path_dir=config["path_dir"], model_name=best_model_name)
    
    if config["wandb"]:
        wandb.finish()
        
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    # Load config
    task = sys.argv[1]
    seed = int(sys.argv[2])
    config = load_config("config.yaml", task, seed)
    
    # Set the random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Run the launcher function
    launcher(config)
    