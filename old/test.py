import sys
import os
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from network import NCHL


def evaluate(agent, task, episodes=100):
    env = gym.make(task)
    rews = []
    
    for _ in tqdm(range(episodes)):
        state, _ = env.reset(seed=567)
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
        rews.append(rew_ep)        
    
    env.close()
    return np.mean(rews), rews

def main():
    path_model = sys.argv[1]
    task = sys.argv[2]
    
    # Load the network
    agent = NCHL.load(path_model)
    
    avg_reward, rewards = evaluate(agent, task)
    
    print(f"Average reward: {avg_reward}")
    
    # Plot the rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards over Episodes')
    plt.savefig(f"{path_model}_rewards.png")
    
if __name__ == "__main__":
    main()