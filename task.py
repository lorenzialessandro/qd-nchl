import sys
import os
import gymnasium as gym
import numpy as np
import torch
import random
import pickle
import json
from multiprocessing import Pool

from network import NCHL, Neuron
from optimizer import MapElite
from utils import *

def evaluate(data):
    params = data[0]
    args = data[1]
    nodes = args["nodes"]
    task = args["task"]

    env = gym.make(task)
    agent = NCHL(nodes, grad=False)
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
    os.makedirs(config["dir"], exist_ok=True)

    archive = MapElite(config["seed"], config["map_size"], config["length"], config["pop_size"], config["bounds"], config["dir"], config["sigma"])
    
    history_best_fitnesses = []
    history_avg_fitnesses = []
    
    for i in range(config["iterations"]):
        candidates = archive.ask(best=config["best"]) # ask for new candidates
        
        res = parallel_val(candidates, config)
        fitnesses = [r[0] for r in res]
        descriptors = [r[1] for r in res]
        
        log = "iteration " + str(i) + "  " + str(max(fitnesses)) + "  " + str(np.mean(fitnesses))
        history_best_fitnesses.append(max(fitnesses))
        history_avg_fitnesses.append(np.mean(fitnesses))
        print(log)
        
        archive.tell(candidates, fitnesses, descriptors) # tell the archive about the new candidates
        
    archive.visualize()
    
    from matplotlib import pyplot as plt
    plt.plot(history_avg_fitnesses, label="avg fitness")
    plt.plot(history_best_fitnesses, label="best fitness")
    plt.legend()
    plt.title("Fitness over generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig(os.path.join(config["dir"], "fitness.png"))
    plt.show()
        
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