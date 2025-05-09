

import numpy as np
import torch
import torch.nn as nn
import os
import pickle

class MapElites():
    """
    Map Elites algorithm for generating a diverse set of solutions in a given search space.
    """

    def __init__(self, seed, map_size, length, pop_size, bounds, threshold, path_dir, sigma):
        self.map_size = map_size
        self.length = length
        self.pop_size = pop_size
        self.bounds = bounds
        self.sigma = sigma
        self.threshold = threshold 
        self.path_dir = path_dir

        self.rng = np.random.default_rng(seed)

        # keys are grid positions (i, j), values are (individual, fitness, descriptors)
        self.archive = {}

    def empty(self):
        """
        Check if the archive is empty.
        """
        return len(self.archive) == 0

    def ask(self):
        """
        Generate a new population of individuals.
        """

        # If the archive is empty, generate random individuals
        if self.empty():
            return self.rng.uniform(-1.0, 1.0, (self.pop_size, self.length)).astype(np.float32)

        # If the archive is not empty, sample from the archive
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
    
    def get_best(self):
        """
        Get the best individual in the archive.
        """
        if self.empty():
            return None, None, None
        
        best_pos = max(self.archive, key=lambda pos: self.archive[pos][1])
        return self.archive[best_pos] # individual, fitness, descriptor
    
    def get_best_k(self, k):
        """
        Get the top k individuals in the archive.
        """
        if self.empty():
            return None, None, None
        
        best_positions = sorted(self.archive.keys(), key=lambda pos: self.archive[pos][1], reverse=True)[:k]
        return [self.archive[pos] for pos in best_positions] # list of (individual, fitness, descriptor)
    
    def save(self):
        """
        Save the archive to a file.
        """
        with open(os.path.join(self.path_dir, "archive.pkl"), "wb") as f:
            pickle.dump(self.archive, f)
            
    @classmethod
    def load(cls, path_dir=None):
        """
        Load the archive from a file.
        """
        with open(os.path.join(self.path_dir, "archive.pkl"), "rb") as f:
            return pickle.load(f)