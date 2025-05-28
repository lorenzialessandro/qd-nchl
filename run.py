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
    
    env.close()
    
    # Compute descriptors
    act_diversity, weight_diversity = agent.get_descriptors()
    
    return rew_ep, (act_diversity, weight_diversity)

def parallel_val(candidates, args):
    with Pool() as p:
        return p.map(evaluate, [[c, json.loads(json.dumps(args))] for c in candidates])
    
def launcher(config):
    print(config)
    
    optimizer = MAPElites(config)
    
    if config["wandb"]:
        wb = wandb.init(project="qd-nchl-runs", name=f"{optimizer.__class__.__name__}_{config['task']}_{config['seed']}", config=config)
        
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
        
        log = "iteration "+ str(i) + "  " + str(max(fitnesses)) + "  " + str(np.mean(fitnesses)) + "  " + optimizer.get_stats()
        if config["wandb"]:
            wandb.log({"iteration": i, "max_fitness": max(fitnesses), "avg_fitness": np.mean(fitnesses), "global_best_fitness": global_best_fitness, "coverage": archive.coverage})
        history_best_fitnesses.append(max(fitnesses))
        history_avg_fitnesses.append(np.mean(fitnesses))
        print(log)
        logs.append(log)
        optimizer.visualize_archive()
        
    # Save the best elite
    best_elite = optimizer.get_best()
    best_fit = best_elite['objective']
    best_desc = best_elite['measures']
    best_ind = best_elite['solution']
    
    print("Best fitness: ", best_fit)
    print("Best descriptor: ", best_desc)
    print("Best individual: ", best_ind)
    log = "Best fitness: " + str(best_fit) + "   Best descriptor: " + str(best_desc) + "   Best individual: " + str(best_ind)
    logs.append(log)
    
    save_log(logs, path_dir=config["path_dir"])
    
    optimizer.visualize_archive()
    plot_history(history_avg_fitnesses, history_best_fitnesses, path_dir=config["path_dir"]) 
    
    if config["wandb"]:
        wandb.finish()
    
if __name__ == "__main__":
    seed = random.randint(0, 10000)
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"Using seed: {seed}")
    config = {
        "task": "LunarLander-v3",
        "nodes": [8, 8, 4],
        "seed": seed,
        "dims": [10, 10],  
        "ranges": [[0, 1], [0, 0.5]], 
        "sigma": 0.1,
        "iterations": 100,
        "batch_size": 100,
        "num_emitters": 1,
        "wandb": False,
    }
    
    agent = NCHL(config["nodes"])
    config["solution_dim"] = agent.nparams
    config["path_dir"] = f"exp/{config['task']}_{config['seed']}_map_elites"
    
    launcher(config)