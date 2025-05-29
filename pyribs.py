import os
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter, EvolutionStrategyEmitter, GradientArborescenceEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap


class QDBase(ABC):
    def __init__(self, config):
        self.config = config
        self.path_dir = config["path_dir"]

        self.archive = GridArchive(
            solution_dim=config["solution_dim"],
            dims=config["dims"],
            ranges=config["ranges"],
            qd_score_offset=-200 #TODO: adjust this value based on your task and archive settings
        )

        self.emitters = self._create_emitters()
        self.scheduler = Scheduler(self.archive, self.emitters)

    @abstractmethod
    def _create_emitters(self):
        pass

    def ask(self):
        return self.scheduler.ask()

    def tell(self, objectives, measures):
        self.scheduler.tell(objectives, measures)

    def get_best(self):
        return self.archive.best_elite

    def get_stats(self):
        stats = self.archive.stats
        return {
            "num_elites": stats.num_elites,
            "coverage": stats.coverage,
            "qd_score": stats.qd_score,
            "obj_max": stats.obj_max,
            "obj_mean": stats.obj_mean
        }
        
    def save_archive(self):
        os.makedirs(self.path_dir, exist_ok=True)
        with open(os.path.join(self.path_dir, 'archive.pkl'), 'wb') as f:
            pickle.dump(self.archive, f)
        
    def load_archive(self):
        with open(os.path.join(self.path_dir, 'archive.pkl'), 'rb') as f:
            self.archive = pickle.load(f)
        
    def visualize_archive(self):
        os.makedirs(self.path_dir, exist_ok=True)
        plt.figure(figsize=(10, 10))
        grid_archive_heatmap(
            self.archive, transpose_measures=True, cmap='viridis')
        plt.title(f'Archive Heatmap ({self.__class__.__name__})')
        plt.ylabel("std mean neuron activations")
        plt.xlabel("std mean neuron weight changes")
        plt.savefig(os.path.join(self.path_dir,
                    'archive_heatmap.png'), bbox_inches='tight')
        plt.close()

# -


class MAPElites(QDBase):

    def _create_emitters(self):

        return [
            GaussianEmitter(
                archive=self.archive,
                sigma=self.config["sigma"],
                x0=np.random.uniform(-1, 1, self.config["solution_dim"]),
                batch_size=self.config["batch_size"],

            ) for _ in range(self.config["num_emitters"])
        ]

# -

class CMAME(QDBase):

    def _create_emitters(self):

        return [
            GradientArborescenceEmitter(
                archive=self.archive,
                sigma0=0.5,
                ranker="2imp",
                x0=np.random.uniform(-1, 1, self.config["solution_dim"]),
                batch_size=self.config["batch_size"]
            ) for _ in range(self.config["num_emitters"])
        ]

# -


class CMAMAE(QDBase):
    def __init__(self, config):

        super().__init__(config)
        
        self.archive = GridArchive(
            solution_dim=config["solution_dim"],
            dims=config["dims"],
            ranges=config["ranges"],
            qd_score_offset=-200, # TODO: adjust this value based on your task and archive settings
            learning_rate=0.01,
            threshold_min=-200.0 #TODO: adjust this value based on your task and archive settings
        )

        self.result_archive = GridArchive(
            solution_dim=self.config["solution_dim"],
            dims=self.config["dims"],
            ranges=self.config["ranges"]
        )

        self.scheduler = Scheduler(
            self.archive, self.emitters, result_archive=self.result_archive)

    def _create_emitters(self):
        return [
            EvolutionStrategyEmitter(
                archive=self.archive,
                x0=np.random.uniform(-1, 1, self.config["solution_dim"]),
                sigma0=0.5,
                ranker="imp",
                selection_rule="mu",
                restart_rule="basic",
                batch_size=self.config["batch_size"]
            ) for _ in range(self.config["num_emitters"])
        ]
        
# -

class CMAMEGA(QDBase):

    def _create_emitters(self):

        return [
            GradientArborescenceEmitter(
                archive=self.archive,
                sigma0=0.5,
                lr=0.05,
                ranker="2imp",
                x0=np.random.uniform(-1, 1, self.config["solution_dim"]),
                batch_size=self.config["batch_size"]
            ) for _ in range(self.config["num_emitters"])
        ]

# -


class CMAMAEGA(QDBase):
    def __init__(self, config):

        super().__init__(config)
        
        self.archive = GridArchive(
            solution_dim=config["solution_dim"],
            dims=config["dims"],
            ranges=config["ranges"],
            qd_score_offset=-200, # TODO: adjust this value based on your task and archive settings
            learning_rate=0.01,
            threshold_min=-200.0 #TODO: adjust this value based on your task and archive settings
        )

        self.result_archive = GridArchive(
            solution_dim=self.config["solution_dim"],
            dims=self.config["dims"],
            ranges=self.config["ranges"]
        )

        self.scheduler = Scheduler(
            self.archive, self.emitters, result_archive=self.result_archive)

    def _create_emitters(self):
        return [
            GradientArborescenceEmitter(
                archive=self.archive,
                x0=np.random.uniform(-1, 1, self.config["solution_dim"]),
                sigma0=0.5,
                lr=0.05
                ranker="imp",
                selection_rule="mu",
                restart_rule="basic",
                batch_size=self.config["batch_size"]
            ) for _ in range(self.config["num_emitters"])
        ]