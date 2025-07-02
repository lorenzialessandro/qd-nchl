import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from network import NCHL
from ribs.archives import GridArchive


# search all the "archive.pkl" files in the "exp3" directory
# def find_archive_files(directory):
#     archive_files = []
    
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file == "archive.pkl":
#                 archive_files.append(os.path.join(root, file))
#     return archive_files

# archive_files = find_archive_files("exp")
# for archive_file in archive_files:
#     # load the archive
#     archive = pickle.load(open(archive_file, "rb"))
#     # compute the RMSE
#     objectives = archive.data()['objective']
#     max_fitness = np.max(objectives)
#     rmse = np.sqrt(np.mean((objectives - max_fitness) ** 2))
#     print(f"RMSE for {archive_file}: {rmse}")
    
    
# read all the dir name of a new directory and save them in a txt file sorted (txt file name is the task)
# create one file for each task in the directory
def save_directory_names(directory):
    tasks = ['CartPole-v1', 'MountainCar-v0', 'LunarLander-v3']
    for task in tasks:
        # check if subdirectory contains the task as part of the name
        task_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and task in d]
        # sort the directories
        task_dirs.sort()
        # save the directories to a txt file
        # save the complete path of the directories
        if task_dirs:
            os.makedirs(os.path.join(directory, 'path'), exist_ok=True)
            with open(os.path.join(directory, f"path/{task}.txt"), 'w') as f:
                for task_dir in task_dirs:
                    f.write(os.path.join(directory, task_dir) + '\n')
            
def screen_task(task="CartPole-v1"):
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np

    # Create the CartPole environment with render_mode="rgb_array" to get frames
    env = gym.make(task, render_mode="rgb_array")
    observation, info = env.reset()

    # Run one episode and capture a screenshot at a specific step
    screenshot_step = 50  # Capture image at this step

    for step in range(100):
        # Take a random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        # Render the current frame
        frame = env.render()

        # Save screenshot at a specific step
        if step == screenshot_step:
            plt.imshow(frame)
            plt.axis('off')
            # plt.title(f"CartPole Screenshot at Step {step}")
            plt.savefig(f"{task}_screenshot_step_10.pdf", bbox_inches='tight', dpi=300, format='pdf')
            print(f"Screenshot saved at step {step}.")

        if terminated or truncated:
            break

    env.close()
    

import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_single_archive(archive_path, title=None, cmap="viridis", save_path=None):
    archive = pickle.load(open(archive_path, 'rb'))
    map_size = archive.dims
    bounds = []
    for i in range(len(map_size)):
        bounds.append((archive.lower_bounds[i], archive.upper_bounds[i]))
    data = archive.data()
    objectives = data['objective']
    indexes = data['index']
    
    objective_grid = np.full(map_size, np.nan)
    for i, idx in enumerate(indexes):
        coords = np.unravel_index(idx, map_size)
        objective_grid[coords] = objectives[i]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(objective_grid, cmap=cmap, aspect='auto',
                   extent=[bounds[1][0], bounds[1][1], bounds[0][0], bounds[0][1]])

    # if title is None:
    #     title = f"Archive: {archive_path.split('/')[-1]}"
    # ax.set_title(title, fontsize=14, fontweight='bold')
    # ax.set_xlabel("Descriptor 2", fontsize=12)
    # ax.set_ylabel("Descriptor 1", fontsize=12)
    
    # cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label("Objective Value", fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300, format='pdf')
    plt.close()


save_directory_names("expruns")
# plot_single_archive(
#     archive_path='expruns/CMAMAE_LunarLander-v3_4724/archive.pkl',
#     save_path="presentation_plot.pdf"
# )
# screen_task("MountainCar-v0")