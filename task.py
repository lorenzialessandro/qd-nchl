import sys
import os
import gymnasium as gym
import numpy as np
import torch
import random
import json
from multiprocessing import Pool

from network import NCHL, Neuron
from optimizer import MapElites
from utils import *

def evaluate(data):
    params = data[0]
    args = data[1]
    nodes = args["nodes"]
    task = args["task"]

    env = gym.make(task)
    agent = NCHL(nodes) # grad=False
    agent.set_params(params)
    
    rews = []
    state, _ = env.reset()
    done = False
    truncated = False
    rew_ep = 0

    while not (done or truncated):
        input = torch.tensor(state)
        output = agent.forward(input)
        agent.update_weights()

        if "Ant" in task:
            # continuous action spaces in Ant-v5
            action = output.detach().cpu().numpy().flatten()
        else:
            action = np.argmax(output.tolist())  
        
        state, reward, done, truncated, _ = env.step(action)

        rew_ep += reward
        rews.append(reward)
        
    env.close()
    
    # Compute descriptors
    d = agent.get_descriptors()      
    
    return rew_ep, d

def parallel_val(candidates, args):
    with Pool() as p:
        return p.map(evaluate, [[c, json.loads(json.dumps(args))] for c in candidates])
    
def launcher(config):
    print(config)
    os.makedirs(config["path_dir"], exist_ok=True)
    
    # Extract MapElites parameters from the config
    map_elites_params = {
        'seed': config['seed'],
        'map_size': config['map_size'],
        'length': config['length'],
        'pop_size': config['pop_size'],
        'bounds': config['bounds'],
        'threshold': config['threshold'],
        'path_dir': config['path_dir'],
        'sigma': config['sigma']
    }

    archive = MapElites(**map_elites_params)
    
    history_best_fitnesses = []
    history_avg_fitnesses = []
    
    for i in range(config["iterations"]):
        candidates = archive.ask() # ask for new candidates
        
        res = parallel_val(candidates, config)
        fitnesses = [r[0] for r in res]
        descriptors = [r[1] for r in res]
        
        log = "iteration " + str(i) + "  " + str(max(fitnesses)) + "  " + str(np.mean(fitnesses))
        history_best_fitnesses.append(max(fitnesses))
        history_avg_fitnesses.append(np.mean(fitnesses))
        print(log)
        
        archive.tell(candidates, fitnesses, descriptors) # tell the archive about the new candidates
        
        if i % 10 == 0:
            visualize_archive(archive, cmap="Greens", annot=False, high=False)
        
    best_ind, best_fit, best_desc = archive.get_best()
    print("Best fitness: ", best_fit)
    print("Best descriptor: ", best_desc)
    print("Best individual: ", best_ind)
    best_net = NCHL(nodes=config["nodes"])
    best_net.set_params(best_ind)
    best_net.save(path_dir=config["path_dir"]) # save the best network
    
    archive.save()
    visualize_archive(archive, cmap="Greens", annot=False, high=False)
    plot_history(history_avg_fitnesses, history_best_fitnesses, path_dir=config["path_dir"])
    
    # load the best network
    # load_best = NCHL.load(path_dir=config["path_dir"]) 

        
if __name__ == "__main__":
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