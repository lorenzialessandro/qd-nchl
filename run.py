import sys
import os
import gymnasium as gym
import numpy as np
import torch
import random
import json
import pickle
import wandb
from multiprocessing import Pool

from network import NCHL, Neuron
from pyribs import MAPElites, CMAME, CMAMAE
from utils import *

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
    
    while not (done or truncated) and step_count < max_steps:
        input_tensor = torch.tensor(state, dtype=torch.float32)
        output = agent.forward(input_tensor)
        agent.update_weights()

        if is_continuous:
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
            action = np.argmax(output.detach().cpu().numpy())
        
        state, reward, done, truncated, info = env.step(action)

        rew_ep += reward
        rews.append(reward)
        step_count += 1
    
    env.close()
    
    # Compute descriptors
    act_diversity, weight_diversity = agent.get_descriptors()
    # print(f"Episode reward: {rew_ep}, Action diversity: {act_diversity}, Weight diversity: {weight_diversity}")
    
    return rew_ep, (act_diversity, weight_diversity)

def parallel_val(candidates, args):
    with Pool() as p:
        return p.map(evaluate, [[c, json.loads(json.dumps(args))] for c in candidates])
    
def launcher(config):
    # print(config)
    
    if config["optimizer"] == "MAPElites":
        optimizer = MAPElites(config)
    elif config["optimizer"] == "CMAME":
        optimizer = CMAME(config)
    elif config["optimizer"] == "CMAMAE":
        optimizer = CMAMAE(config)
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    if config["wandb"]:
        wb = wandb.init(project="qd-nchl-pyribs", name=f"{config['path_dir']}", config=config)
        
    os.makedirs(config["path_dir"], exist_ok=True)
    
    history_best_fitnesses = []
    history_avg_fitnesses = []
    logs = []
    
    global_best_fitness = float("-inf")
    
    
    for i in range(config["iterations"]):
        candidates = optimizer.ask()
        
        res = parallel_val(candidates, config)
        
        fitnesses = [r[0] for r in res]
        descriptors = [r[1] for r in res]
        
        # for fitness, desc in zip(fitnesses, descriptors):
        #     print(f"Fitness: {fitness}, Descriptors: {desc}")
        # print()
        
        optimizer.tell(fitnesses, descriptors)
        
        global_best_fitness = max(global_best_fitness, max(fitnesses))
        
        stats = optimizer.get_stats()
        log = "iteration "+ str(i) + "  " + str(max(fitnesses)) + "  " + str(np.mean(fitnesses)) + "  " + str(stats["num_elites"]) + "  " + str(stats["coverage"]) + "  " + str(stats["qd_score"]) + "  " + str(stats["obj_max"]) + "  " + str(stats["obj_mean"])
        if config["wandb"]:
            wandb.log({"iteration": i, "max_fitness": max(fitnesses), "avg_fitness": np.mean(fitnesses), "global_best_fitness": global_best_fitness, "num_elites": stats["num_elites"], "coverage": stats["coverage"], "qd_score": stats["qd_score"], "obj_max": stats["obj_max"], "obj_mean": stats["obj_mean"]})
        history_best_fitnesses.append(max(fitnesses))
        history_avg_fitnesses.append(np.mean(fitnesses))
        print(log)
        logs.append(log)     
        
    # Save archive
    optimizer.save_archive()
        
    # Save the best elite
    best_elite = optimizer.get_best()
    best_fit = best_elite['objective']
    best_desc = best_elite['measures']
    best_ind = best_elite['solution']
    
    # Save the best elite to a file
    with open(os.path.join(config["path_dir"], 'best_elite.pkl'), 'wb') as f:
        pickle.dump(best_elite, f)
    
    # print("Best fitness: ", best_fit)
    # print("Best descriptor: ", best_desc)
    # print("Best individual: ", best_ind)
    log = "Best fitness: " + str(best_fit) + "   Best descriptor: " + str(best_desc) + "   Best individual: " + str(best_ind)
    logs.append(log)
    
    save_log(logs, path_dir=config["path_dir"])
    
    # Log config and final stats
    logs = []
    logs.append("Configuration:")
    logs.append(json.dumps(config, indent=4))
    logs.append("Final Stats:")
    stats = optimizer.get_stats()
    # precision = optimizer.compute_archive_precision()
    logs.append(f"Number of elites: {stats['num_elites']}")
    logs.append(f"Coverage: {stats['coverage']}")
    logs.append(f"QD Score: {stats['qd_score']}")
    logs.append(f"Objective Max: {stats['obj_max']}")
    logs.append(f"Objective Mean: {stats['obj_mean']}")
    # logs.append(f"Archive Precision (RMSE): {precision}")
    save_log(logs, path_dir=config["path_dir"], filename="final_stats.txt")
    

    # Log best 3 elites
    logs = []
    data = optimizer.archive.data()
    objectives = data['objective']
    best_indices = sorted(range(len(objectives)), key=lambda i: objectives[i], reverse=True)[:3]
    for i, idx in enumerate(best_indices):
        log = f"Elite {i+1}: Fitness: {objectives[idx]}, Solution: {data['solution'][idx]}, Measures: {data['measures'][idx]}"
        print(log)
        logs.append(log)
        
    save_log(logs, path_dir=config["path_dir"], filename="best_elites.txt")
    
    optimizer.visualize_archive()
    plot_history(history_avg_fitnesses, history_best_fitnesses, path_dir=config["path_dir"]) 
    
    if config["wandb"]:
        wandb.finish()
        
def run(n_run=1):
    optimizers = ["MAPElites", "CMAME", "CMAMAE"]
    
    # CartPole-v1 
    config = {
        "task": "CartPole-v1",
        "nodes": [4, 4, 2],
        "dims": [10, 10],  
        "ranges": [[0, 1], [0, 1]], 
        "sigma": 0.1,
        "iterations": 300, #300
        "batch_size": 100, #100
        "num_emitters": 2,
        "wandb": False,
    }
    
    agent = NCHL(config["nodes"])
    config["solution_dim"] = agent.nparams
    
    for seed in seeds:
        
        config["seed"] = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        for optimizer in optimizers:
            config["optimizer"] = optimizer
                
            config["path_dir"] = f"expruns_map/{config['optimizer']}_{config['task']}_{config['seed']}"
            
            print(f"Running with seed: {seed} and optimizer: {optimizer}")
            launcher(config)
    
if __name__ == "__main__":
    n_run = 1
    if len(sys.argv) > 1:
        n_run = int(sys.argv[1])
        
    run(n_run)