

import numpy as np
import torch
import torch.nn as nn
import os

class MapElite():
    """
    """

    def __init__(self, seed, map_size, length, pop_size, bounds, dir, sigma=0.1):

        self.map_size = map_size
        self.length = length
        self.pop_size = pop_size
        self.bounds = bounds
        self.sigma = sigma
        self.dir = dir

        self.rng = np.random.default_rng(seed)

        # keys are grid positions (i, j), values are (individual, fitness, descriptors)
        self.archive = {}

    def _empty(self):
        """
        Check if the archive is empty.
        """
        return len(self.archive) == 0

    def ask(self, best=False):
        """
        Generate a new population of individuals.
        """

        # If the archive is empty, generate random individuals
        if self._empty():
            return self.rng.uniform(-1.0, 1.0, (self.pop_size, self.length)).astype(np.float32)

        # If the archive is not empty, sample from the archive
        else:
            if best:
                positions = [max(self.archive.keys(), key=lambda x: self.archive[x][1])]
            
            else:
                positions = list(self.archive.keys())
            
            # Sample random positions from the archive
            selected_indexes = self.rng.choice(len(positions), size=self.pop_size, replace=True)
            selected_positions = [positions[i] for i in selected_indexes]

            # Get the corresponding individuals from the archive
            parents = [self.archive[pos][0] for pos in selected_positions]
            # Create offspring applying mutation
            offspring = self._mutate(parents)

            return offspring

    def _mutate(self, parents):
        """
        Apply mutation to the parents to create offspring.
        """
        # Add Gaussian noise to the parents
        offspring = []

        for parent in parents:
            mutation = self.rng.normal(0, self.sigma, size=parent.shape)
            child = np.clip(parent + mutation, -1.0, 1.0).astype(np.float32)
            offspring.append(child)

        return np.array(offspring)

    def tell(self, individuals, fitnesses, descriptors):
        """
        Store the individuals in the archive.
        """
        for individual, fitness, descriptor in zip(individuals, fitnesses, descriptors):
            # Get the grid position of the individual
            pos = self._get_grid_position(descriptor)

            # If the position is empty or the fitness is better, update the archive
            if pos not in self.archive or fitness > self.archive[pos][1]:
                self.archive[pos] = (individual, fitness, descriptor)
                
            # print(f"Position: {pos}, Fitness: {fitness}, Descriptor: {descriptor}")

    def _get_grid_position(self, descriptor):
        """
        Get the grid position of the descriptor.
        """
        position = []
        for i, desc in enumerate(descriptor):
            # Normalize the descriptor to the range [0, 1]
            min_v, max_v = self.bounds[i]
            desc_norm = (desc - min_v) / (max_v - min_v)
            desc_norm = np.clip(desc_norm, 0, 1)
   
            # Get the grid position
            pos = int(desc_norm * self.map_size[i])
            # Ensure the position is within bounds
            pos = min(pos, self.map_size[i] - 1)

            position.append(pos) 
        
        return tuple(position)

    def visualize(self, save=False):
        """
        Visualize the archive using heatmap.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create a 2D array to store the fitness values
        fitness_map = np.full(self.map_size, -200.0)

        for pos, (individual, fitness, descriptor) in self.archive.items():
            fitness_map[pos] = fitness

        plt.figure(figsize=(10, 8))
        sns.heatmap(fitness_map, annot=False, fmt=".2f", cmap="viridis", cbar=True)
        plt.title("Map-Elite Archive Fitness Values")
        plt.xlabel("Grid X")
        plt.ylabel("Grid Y")
        if save:
            plt.savefig(os.path.join(self.dir, "archive.png"))
        plt.show()

    
