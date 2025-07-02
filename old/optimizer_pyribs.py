import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import random
import matplotlib.pyplot as plt

from ribs.archives import GridArchive
from ribs.visualize import grid_archive_heatmap

class MapElites():
    """
    """

    def __init__(self, seed, map_size, length, pop_size, bounds, threshold, path_dir, sigma, archive_type='grid',
                 use_elitism=False, use_crossover=False, use_adaptive_mutation=False, use_prob_sampling=False):
        
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
        
        # Ask strategy
        self.use_prob_sampling = use_prob_sampling # if True, use probability sampling for mutation, else use uniform sampling

        # Archive parameters
        self.use_elitism = use_elitism
        self.use_crossover = use_crossover
        self.crossover_rate = 0.3
        
        # Initialization strategy
        self.init_strategy = 'uniform'  # 'uniform', 'normal', 'xavier'
        
        # Adaptive mutation parameters
        self.min_sigma = 0.01
        self.max_sigma = 0.3
        self.sigma_decay = 0.99
        self.use_adaptive_mutation = use_adaptive_mutation
        
        if archive_type == 'grid':
            self.archive = GridArchive(
                solution_dim=self.length,
                dims=self.map_size,
                ranges=self.bounds,
                seed=seed,
            )
        
        self.coverage = 0.0

    def empty(self):
        """
        Check if the archive is empty.
        """
        return self.archive.empty
    
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
        Generate a new population of individuals using pyribs GridArchive.
        """
        # If the archive is empty, generate random individuals
        if len(self.archive) == 0:
            return np.array([self.initialize_individual() for _ in range(self.pop_size)])

        # If the archive is not empty, sample from the archive
        else:
            # Get all solutions from the archive
            elite_data = self.archive.data()
            
            # Filter out empty cells (solutions with -inf objective)
            occupied_mask = elite_data['objective'] > -np.inf
            if not np.any(occupied_mask):
                # No valid solutions in archive, generate random ones
                return np.array([self.initialize_individual() for _ in range(self.pop_size)])
            
            # Extract valid solutions and their fitness values
            valid_solutions = elite_data['solution'][occupied_mask]
            valid_objectives = elite_data['objective'][occupied_mask]
            
            # Determine how many individuals to generate from each method
            num_elites = int(self.pop_size * 0.1) if self.use_elitism else 0
            num_crossover = int(self.pop_size * self.crossover_rate) if self.use_crossover else 0 
            num_mutation = self.pop_size - num_elites - num_crossover
            
            offspring = []
            
            # 1. Elite individuals (top performers) - direct copies
            if num_elites > 0:
                # Sort solutions by fitness (objective)
                sorted_indices = np.argsort(valid_objectives)[::-1]  # Descending order
                
                # Select top elite individuals (handle case where there are fewer solutions than num_elites)
                num_elites_actual = min(num_elites, len(valid_solutions))
                elite_solutions = valid_solutions[sorted_indices[:num_elites_actual]]
                offspring.extend(elite_solutions.tolist())
            
            # 2. Crossover individuals
            if num_crossover > 0 and len(valid_solutions) >= 2:  # Need at least 2 parents for crossover
                for _ in range(num_crossover):
                    # Select two parents via tournament selection
                    parent1_idx = self._tournament_selection_pyribs(valid_solutions, valid_objectives, min(3, len(valid_solutions)))
                    parent2_idx = self._tournament_selection_pyribs(valid_solutions, valid_objectives, min(3, len(valid_solutions)))
                    
                    parent1 = valid_solutions[parent1_idx]
                    parent2 = valid_solutions[parent2_idx]
                    
                    # Apply crossover
                    child = self._crossover(parent1, parent2)
                    offspring.append(child)
            
            # 3. Mutation individuals
            if num_mutation > 0 and len(valid_solutions) > 0:  # Need at least one parent for mutation
                # Sample random solutions from the archive with fitness-proportionate selection
                selected_indices = self._sample_archive_solutions_pyribs(valid_solutions, valid_objectives, num_mutation)
                
                # Get the corresponding individuals from the archive
                parents = valid_solutions[selected_indices]
                
                # Create offspring applying mutation
                mutated = self._mutate(parents)
                offspring.extend(mutated)
            
            # 4. Always include the best individual found so far (if there's room for it)
            if self.best_individual is not None and self.use_elitism and len(offspring) > 0:
                # Calculate a valid replace_idx
                if len(offspring) <= num_elites:  # All offspring are elites or we have few offspring
                    # Replace a random individual
                    replace_idx = self.rng.integers(0, len(offspring))
                else:
                    # Replace a random non-elite individual
                    non_elite_count = len(offspring) - num_elites
                    replace_idx = num_elites + self.rng.integers(0, non_elite_count)
                
                # Double-check index bounds before assignment
                if 0 <= replace_idx < len(offspring):
                    offspring[replace_idx] = self.best_individual
            
            # If we somehow ended up with no offspring (very rare edge case), generate random ones
            if not offspring:
                return np.array([self.initialize_individual() for _ in range(self.pop_size)])
                
            return np.array(offspring)


    def _tournament_selection_pyribs(self, solutions, objectives, tournament_size):
        """
        Tournament selection for pyribs GridArchive.
        
        Args:
            solutions: Array of solutions from the archive
            objectives: Array of objective values corresponding to solutions
            tournament_size: Number of individuals in tournament
        
        Returns:
            Index of selected individual
        """
        # Randomly select tournament_size individuals
        tournament_indices = self.rng.choice(len(solutions), size=tournament_size, replace=False)
        
        # Find the best individual in the tournament (highest objective)
        tournament_objectives = objectives[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_objectives)]
        
        return winner_idx


    def _sample_archive_solutions_pyribs(self, solutions, objectives, num_samples):
        """
        Sample solutions from archive with fitness-proportionate selection for pyribs.
        
        Args:
            solutions: Array of solutions from the archive
            objectives: Array of objective values corresponding to solutions
            num_samples: Number of samples to draw
        
        Returns:
            Array of indices of selected solutions
        """
        # Convert objectives to probabilities (handle negative values)
        min_obj = np.min(objectives)
        if min_obj < 0:
            shifted_objectives = objectives - min_obj + 1e-8  # Shift to positive
        else:
            shifted_objectives = objectives + 1e-8  # Add small epsilon to avoid zero probabilities
        
        # Normalize to probabilities
        probabilities = shifted_objectives / np.sum(shifted_objectives)
        
        # Sample with replacement based on fitness
        selected_indices = self.rng.choice(len(solutions), size=num_samples, replace=True, p=probabilities)
        
        return selected_indices

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
        
        if self.use_prob_sampling:
            # Normalize fitnesses to get probabilities
            fitnesses = fitnesses / np.sum(fitnesses)
            # Calculate selection probabilities
            probs = fitnesses / np.sum(fitnesses)
            
            # Sample positions based on probabilities
            selected_indices = self.rng.choice(
                len(positions), size=num_samples, replace=True, p=probs
            )
        else:
            # Uniform sampling
            selected_indices = self.rng.choice(
                len(positions), size=num_samples, replace=False
            )
            
        # Return the selected positions
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
        
        # Store individuals in the archive
        self.archive.add(individuals, fitnesses, descriptors)
        
        self.coverage = self.archive.stats.coverage

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
        
        return self.archive.best_elite
    
    def visualize_archive(self):
        """
        Visualize the archive.
        """
        # Create the directory if it doesn't exist
        os.makedirs(self.path_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 10))
        grid_archive_heatmap(self.archive, transpose_measures=True, cmap='viridis')
        plt.title("Archive Visualization")
        plt.xlabel("Descriptor 1")
        plt.ylabel("Descriptor 2")
        plt.savefig(os.path.join(self.path_dir, 'archive_visualization.png'))
        plt.close()

    def save(self):
        """
        Save the archive and parameters to a file.
        """
        save_data = {
            'seed': self.rng.bit_generator._seed_seq.entropy,  # save the original seed
            'archive': self.archive,
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'map_size': self.map_size,
            'length': self.length,
            'pop_size': self.pop_size,
            'bounds': self.bounds,
            'threshold': self.threshold,
            'path_dir': self.path_dir,
            'sigma': self.sigma,
            'archive_type': 'grid',
            'use_elitism': self.use_elitism,
            'use_crossover': self.use_crossover,
            'use_adaptive_mutation': self.use_adaptive_mutation,
            'use_prob_sampling': self.use_prob_sampling,
            'init_strategy': self.init_strategy,
            'min_sigma': self.min_sigma,
            'max_sigma': self.max_sigma,
            'sigma_decay': self.sigma_decay,
        }
        # Create the directory if it doesn't exist
        os.makedirs(self.path_dir, exist_ok=True)
        # Save the archive to a file
        archive_path = os.path.join(self.path_dir, 'archive.pkl')
        with open(archive_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Archive saved to {archive_path}")
        return archive_path

    @classmethod        
    def load(cls, path_file):
        """
        Load the archive and parameters from a file.
        """
        # Load the archive from a file
        with open(path_file, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create a new MapElites instance with the loaded parameters
        instance = cls(
            seed=save_data['seed'],
            map_size=save_data['map_size'],
            length=save_data['length'],
            pop_size=save_data['pop_size'],
            bounds=save_data['bounds'],
            threshold=save_data['threshold'],
            path_dir=save_data['path_dir'],
            sigma=save_data['sigma'],
            archive_type=save_data['archive_type'],
            use_elitism=save_data['use_elitism'],
            use_crossover=save_data['use_crossover'],
            use_adaptive_mutation=save_data['use_adaptive_mutation'],
            use_prob_sampling=save_data['use_prob_sampling']
        )
        
        # Restore the archive and best individual
        instance.archive = save_data['archive']
        instance.best_individual = save_data['best_individual']
        instance.best_fitness = save_data['best_fitness']
        
        return instance
    

    