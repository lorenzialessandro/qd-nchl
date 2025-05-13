import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import random

class MapElites():
    """
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
        
        # Store the best individual 
        self.best_individual = None
        self.best_fitness = float('-inf')

        # Archive parameters
        self.use_elitism = True
        self.use_crossover = True
        self.crossover_rate = 0.3
        
        # Initialization strategy
        self.init_strategy = 'uniform'  # 'uniform', 'normal', 'xavier'
        
        # Adaptive mutation parameters
        self.min_sigma = 0.01
        self.max_sigma = 0.1
        self.sigma_decay = 0.99
        self.use_adaptive_mutation = True
        
        # keys are grid positions (i, j), values are (individual, fitness, descriptors)
        self.archive = {}
        self.coverage = 0.0

    def empty(self):
        """
        Check if the archive is empty.
        """
        return len(self.archive) == 0
    
    def initialize_individual(self):
        """
        Initialize a single individual with the chosen strategy.
        """
        if self.init_strategy == 'uniform':
            return self.rng.uniform(-1.0, 1.0, self.length).astype(np.float32)
        elif self.init_strategy == 'normal':
            return self.rng.normal(0.0, 0.1, self.length).astype(np.float32)
        elif self.init_strategy == 'xavier':
            # Xavier initialization for neural networks
            scale = np.sqrt(6.0 / self.length)
            return self.rng.uniform(-scale, scale, self.length).astype(np.float32)
        else:
            # Default to uniform
            return self.rng.uniform(-1.0, 1.0, self.length).astype(np.float32)

    def ask(self):
        """
        Generate a new population of individuals.
        """
        # If the archive is empty, generate random individuals
        if self.empty():
            return np.array([self.initialize_individual() for _ in range(self.pop_size)])

        # If the archive is not empty, sample from the archive
        else:
            positions = list(self.archive.keys())
            
            # Determine how many individuals to generate from each method
            num_elites = int(self.pop_size * 0.3) if self.use_elitism else 0
            num_crossover = int(self.pop_size * self.crossover_rate) if self.use_crossover else 0 
            num_mutation = self.pop_size - num_elites - num_crossover
            
            offspring = []
            
            # 1. Elite individuals (top performers) - direct copies
            if num_elites > 0:
                # Sort positions by fitness
                sorted_positions = sorted(positions, 
                                         key=lambda pos: self.archive[pos][1], 
                                         reverse=True)
                
                # Select top elite individuals
                elites = [self.archive[pos][0] for pos in sorted_positions[:num_elites]]
                offspring.extend(elites)
            
            # 2. Crossover individuals
            if num_crossover > 0:
                for _ in range(num_crossover):
                    # Select two parents via tournament selection
                    parent1_pos = self._tournament_selection(positions, 3)
                    parent2_pos = self._tournament_selection(positions, 3)
                    
                    parent1 = self.archive[parent1_pos][0]
                    parent2 = self.archive[parent2_pos][0]
                    
                    # Apply crossover
                    child = self._crossover(parent1, parent2)
                    offspring.append(child)
            
            # 3. Mutation individuals
            if num_mutation > 0:
                # Sample random positions from the archive with fitness-proportionate selection
                selected_positions = self._sample_archive_positions(positions, num_mutation)
                
                # Get the corresponding individuals from the archive
                parents = [self.archive[pos][0] for pos in selected_positions]
                
                # Create offspring applying mutation
                mutated = self._mutate(parents)
                offspring.extend(mutated)
            
            # 4. Always include the best individual found so far
            if self.best_individual is not None and self.use_elitism:
                # Replace a random non-elite individual
                if len(offspring) > 0:
                    replace_idx = num_elites + self.rng.integers(0, len(offspring) - num_elites)
                    offspring[replace_idx] = self.best_individual
            
            return np.array(offspring)
    
    def _tournament_selection(self, positions, tournament_size):
        """
        Select an individual using tournament selection.
        """
        # Select random positions for the tournament
        tournament = random.sample(positions, min(tournament_size, len(positions)))
        
        # Return the position with the highest fitness
        return max(tournament, key=lambda pos: self.archive[pos][1])
    
    def _sample_archive_positions(self, positions, num_samples):
        """
        Sample positions from the archive with probability proportional to fitness.
        """
        if len(positions) <= num_samples:
            return positions
        
        # Get fitnesses for all positions
        fitnesses = np.array([self.archive[pos][1] for pos in positions])
        
        # Shift fitnesses to be positive for probability calculation
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            fitnesses = fitnesses - min_fitness + 1e-6
        
        # Calculate selection probabilities
        probs = fitnesses / np.sum(fitnesses)
        
        # Sample positions based on probabilities
        selected_indices = self.rng.choice(
            len(positions), size=num_samples, replace=True, p=probs
        )
        
        return [positions[i] for i in selected_indices]
    
    def _crossover(self, parent1, parent2):
        """
        Apply crossover between two parents to create a child.
        """
        # Implementation of two-point crossover
        if self.rng.random() < 0.5:
            # Two-point crossover
            points = sorted(self.rng.choice(self.length, size=2, replace=False))
            child = np.copy(parent1)
            child[points[0]:points[1]] = parent2[points[0]:points[1]]
        else:
            # Uniform crossover
            mask = self.rng.random(self.length) < 0.5
            child = np.where(mask, parent1, parent2)
        
        return child.astype(np.float32)

    def _mutate(self, parents):
        """
        Apply mutation to the parents to create offspring.
        """
        # Add Gaussian noise to the parents
        offspring = []

        # Adjust sigma based on archive size and best fitness
        if self.use_adaptive_mutation:
            # More individuals in archive = lower sigma (more exploitation)
            archive_ratio = min(1.0, len(self.archive) / (self.map_size[0] * self.map_size[1]))
            self.sigma = max(self.min_sigma, 
                            min(self.max_sigma, 
                                self.sigma * (1.0 - 0.1 * archive_ratio)))

        for parent in parents:
            # Occasionally apply stronger mutation
            current_sigma = self.sigma * 3.0 if self.rng.random() < 0.1 else self.sigma
            
            # Apply mutation: Gaussian noise 
            mutation = self.rng.normal(0, current_sigma, size=parent.shape)
            child = np.clip(parent + mutation, -1.0, 1.0).astype(np.float32)
            offspring.append(child)

        return offspring

    def tell(self, individuals, fitnesses, descriptors):
        """
        Store the individuals in the archive.
        """
        # Track best individual across all evaluations
        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > self.best_fitness:
            self.best_fitness = fitnesses[max_idx]
            self.best_individual = individuals[max_idx].copy()
        
        # Filter out NaN fitnesses
        valid_indices = ~np.isnan(fitnesses)
        individuals = [ind for i, ind in enumerate(individuals) if valid_indices[i]]
        fitnesses = [fit for i, fit in enumerate(fitnesses) if valid_indices[i]]
        descriptors = [desc for i, desc in enumerate(descriptors) if valid_indices[i]]
        
        for individual, fitness, descriptor in zip(individuals, fitnesses, descriptors):
            # Get the grid position of the individual
            pos = self._get_grid_position(descriptor)

            # If the position is empty or the fitness is better, update the archive
            if pos not in self.archive or fitness > self.archive[pos][1]:
                self.archive[pos] = (individual.copy(), fitness, descriptor)
                
            # print(f"Stored individual at position {pos} with fitness {fitness} and descriptor {descriptor}")

        # Update coverage
        self.coverage = len(self.archive) / (self.map_size[0] * self.map_size[1])

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
        Save the archive and parameters to a file.
        """
        save_data = {
            'seed': self.rng.bit_generator._seed_seq.entropy,  # save the original seed
            'map_size': self.map_size,
            'length': self.length,
            'pop_size': self.pop_size,
            'bounds': self.bounds,
            'threshold': self.threshold,
            'path_dir': self.path_dir,
            'sigma': self.sigma,
            'archive': self.archive,
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness
        }

        with open(os.path.join(self.path_dir, "archive.pkl"), "wb") as f:
            pickle.dump(save_data, f)

    
    @classmethod        
    def load(cls, path_file):
        """
        Load the archive from a file and return a MapElites instance.
        """
        with open(path_file, "rb") as f:
            data = pickle.load(f)

        instance = cls(
            seed=data['seed'],
            map_size=data['map_size'],
            length=data['length'],
            pop_size=data['pop_size'],
            bounds=data['bounds'],
            threshold=data['threshold'],
            path_dir=data['path_dir'],
            sigma=data['sigma']
        )

        instance.archive = data['archive']
        instance.best_individual = data['best_individual']
        instance.best_fitness = data['best_fitness']
        instance.coverage = len(instance.archive) / (instance.map_size[0] * instance.map_size[1])

        return instance
