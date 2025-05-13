import numpy as np
import torch
import torch.nn as nn
import pickle
import os

class Neuron:
    """
    Neuron class representing a single neuron (the Hebbian rule) in the network.
    """
    def __init__(self, neuron_id: int, params=None, device="cpu"):
        self.neuron_id = neuron_id
        self.device = device

        # Hebbian rule parameters initialized at random - converted to float32
        self.pre_factor = torch.tensor(
            np.random.uniform(-1.0, 1.0), device=device, dtype=torch.float32)
        self.post_factor = torch.tensor(
            np.random.uniform(-1.0, 1.0), device=device, dtype=torch.float32)
        self.correlation = torch.tensor(
            np.random.uniform(-1.0, 1.0), device=device, dtype=torch.float32)
        self.decorrelation = torch.tensor(
            np.random.uniform(-1.0, 1.0), device=device, dtype=torch.float32)
        self.eta = torch.tensor(np.random.uniform(0, 1.0), device=device, dtype=torch.float32)

        self.params = None

        # Current activation value of the neuron
        self.activation = torch.tensor(0.0, device=device, dtype=torch.float32)

        # Store activations and weight changes for the neuron for descriptors
        self.activations = []
        self.weight_changes = []

        if params is not None:
            self.set_params(params)

    def add_activation(self, activation):
        """Add an activation to the list of activations."""
        self.activations.append(activation.item())

    def add_weight_change(self, weight_change):
        """Add a weight change to the list of weight changes."""
        self.weight_changes.append(weight_change)

    def set_params(self, params: list):
        """Set the Hebbian learning parameters and learning rate for this neuron."""
        self.params = params
        self.set_hebbian_params(params[0], params[1], params[2], params[3])
        self.set_eta(params[4])

    def set_hebbian_params(self, pre, post, corr, decorr):
        """Set the Hebbian learning parameters for this neuron."""
        self.pre_factor = pre.clone().detach().to(self.device).float() if isinstance(
            pre, torch.Tensor) else torch.tensor(pre, device=self.device, dtype=torch.float32)
        self.post_factor = post.clone().detach().to(self.device).float() if isinstance(
            post, torch.Tensor) else torch.tensor(post, device=self.device, dtype=torch.float32)
        self.correlation = corr.clone().detach().to(self.device).float() if isinstance(
            corr, torch.Tensor) else torch.tensor(corr, device=self.device, dtype=torch.float32)
        self.decorrelation = decorr.clone().detach().to(self.device).float() if isinstance(
            decorr, torch.Tensor) else torch.tensor(decorr, device=self.device, dtype=torch.float32)

    def set_eta(self, eta):
        """Set the learning rate for this neuron."""
        self.eta = eta.clone().detach().to(self.device).float() if isinstance(
            eta, torch.Tensor) else torch.tensor(eta, device=self.device, dtype=torch.float32)

    def set_activation(self, activation):
        """Set the current activation value of the neuron."""
        self.activation = activation.to(self.device).float()
        self.add_activation(activation)

    def get_hebbian_terms(self):
        return (
            (self.pre_factor * self.activation).to(self.device),
            (self.post_factor * self.activation).to(self.device),
            torch.tensor(1.0, device=self.device, dtype=torch.float32) if self.correlation == 1. else (
                self.correlation * self.activation).to(self.device),
            self.decorrelation.to(self.device)
        )

    def get_rule(self):
        return [self.pre_factor, self.post_factor, self.correlation, self.decorrelation, self.eta]

    def __getstate__(self):
        """Return state values to be pickled."""
        state = {}
        # Store all attributes except tensors
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                # Convert tensors to lists for pickling
                state[f"{key}_list"] = value.detach().cpu().tolist()
            else:
                state[key] = value

        # Handle special case for lists of tensors
        if hasattr(self, 'activations') and self.activations:
            state['activations'] = self.activations

        if hasattr(self, 'weight_changes') and self.weight_changes:
            state['weight_changes'] = self.weight_changes

        return state

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        # Recreate the dictionary
        self.__dict__ = {}

        # Process each key in the state
        for key, value in state.items():
            if key.endswith('_list'):
                # Convert lists back to tensors with float32 dtype
                tensor_key = key[:-5]  # Remove '_list' suffix
                self.__dict__[tensor_key] = torch.tensor(value, device="cpu", dtype=torch.float32)
            else:
                self.__dict__[key] = value

        # Ensure device is set
        self.device = state.get('device', 'cpu')


class NCHL(nn.Module):
    """
    Neuron Centric Hebbian Learning (NCHL) class.
    """

    def __init__(self, nodes: list, params=None, population=None, grad=False, device="cpu", init=None):

        super(NCHL, self).__init__()
        self.float() 

        self.device = device
        self.grad = grad
        self.nodes = torch.tensor(nodes, device=device, dtype=torch.float32)
        self.nweights = sum([self.nodes[i] * self.nodes[i + 1]
                            for i in range(len(self.nodes) - 1)])

        # Initialize Neurons
        self.all_neurons = []
        self.neurons = self._initialize_neurons(nodes, population, device)

        # Create network layers
        self.network = self._initialize_network(nodes, init)

        self.to(device)

        self.nparams = int(sum(self.nodes) * 5 - self.nodes[0] - self.nodes[-1])

        if params is not None:
            self.set_params(params)

    def _initialize_neurons(self, nodes, population, device):
        neurons = []
        neuron_id = 0

        if population is not None:
            assert len(population) == sum(nodes), (
                f"Population size does not match number of neurons. "
                f"Expected: {sum(nodes)}, Got: {len(population)}"
            )
            i = 0
            for n_neurons in nodes:
                layer_neurons = []
                for _ in range(n_neurons):
                    layer_neurons.append(population[i])
                    self.all_neurons.append(population[i])
                    i += 1
                neurons.append(layer_neurons)
        else:
            for n_neurons in nodes:
                layer_neurons = []
                for _ in range(n_neurons):
                    neuron = Neuron(neuron_id, device=device)
                    layer_neurons.append(neuron)
                    self.all_neurons.append(neuron)
                    neuron_id += 1
                neurons.append(layer_neurons)

        return neurons

    def _initialize_network(self, nodes, init):
        network = []
        for i in range(len(nodes) - 1):
            layer = nn.Linear(nodes[i], nodes[i + 1], bias=False)
            layer.float()  

            if init is None:
                nn.init.xavier_uniform_(layer.weight.data, 0.5)
            else:
                self._initialize_weights(layer, init)

            layer.to(self.device)
            network.append(layer)
        return network

    def _initialize_weights(self, layer, init):
        if init == 'xa_uni':
            nn.init.xavier_uniform_(layer.weight.data, 0.3)
        elif init == 'sparse':
            nn.init.sparse_(layer.weight.data, 0.8)
        elif init == 'uni':
            nn.init.uniform_(layer.weight.data, -0.1, 0.1)
        elif init == 'normal':
            nn.init.normal_(layer.weight.data, 0, 0.024)
        elif init == 'ka_uni':
            nn.init.kaiming_uniform_(layer.weight.data, 3)
        elif init == 'uni_big':
            nn.init.uniform_(layer.weight.data, -1, 1)
        elif init == 'xa_uni_big':
            nn.init.xavier_uniform_(layer.weight.data)

    def forward(self, inputs):
        with torch.no_grad():
            x = inputs.to(self.device).float()  
            if x.dim() == 1:
                x = x.unsqueeze(0)

            # Set input layer activations (using first item in batch)
            for i, neuron in enumerate(self.neurons[0]):
                neuron.set_activation(x[0, i])

            # Forward pass
            for layer_idx, layer in enumerate(self.network):
                x = torch.tanh(layer(x))
                # Set activations for neurons in current layer
                for i, neuron in enumerate(self.neurons[layer_idx + 1]):
                    neuron.set_activation(x[0, i])
            return x

    def update_weights(self):
        weights = self.get_weights()

        for layer_idx in range(len(weights)):
            pre_neurons = self.neurons[layer_idx]
            post_neurons = self.neurons[layer_idx + 1]

            # Pre-compute Hebbian terms for the layer
            pre_terms = torch.stack(
                [torch.stack(n.get_hebbian_terms()).to(self.device) for n in pre_neurons])
            post_terms = torch.stack(
                [torch.stack(n.get_hebbian_terms()).to(self.device) for n in post_neurons])

            # Create weight update matrix
            pre_contribution = pre_terms[:, 0].unsqueeze(
                0).expand(len(post_neurons), -1).to(self.device)
            post_contribution = post_terms[:, 1].unsqueeze(
                1).expand(-1, len(pre_neurons)).to(self.device)

            # Correlation terms
            corr_i = pre_terms[:, 2].unsqueeze(0).to(self.device)
            corr_j = post_terms[:, 2].unsqueeze(1).to(self.device)
            corr_contrib = torch.where(
                (corr_i == 1.) & (corr_j == 1.),
                torch.zeros_like(pre_contribution, device=self.device),
                corr_i * corr_j
            )

            # Decorrelation terms
            decorr_i = pre_terms[:, 3].unsqueeze(0).to(self.device)
            decorr_j = post_terms[:, 3].unsqueeze(1).to(self.device)
            decorr_contrib = torch.where(
                (decorr_i == 1.) & (decorr_j == 1.),
                torch.zeros_like(pre_contribution, device=self.device),
                decorr_i * decorr_j
            )

            # Combine all contributions
            dw = (pre_contribution + post_contribution +
                  corr_contrib + decorr_contrib).to(self.device)

            # Learning rates
            pre_etas = torch.stack([n.eta.to(self.device)
                                   for n in pre_neurons])
            post_etas = torch.stack([n.eta.to(self.device)
                                    for n in post_neurons])
            eta_matrix = (
                (pre_etas.unsqueeze(0) + post_etas.unsqueeze(1)) / 2).to(self.device)

            # Final weight update
            weight_change = (eta_matrix * dw).to(self.device)

            # Store weight changes and update weights
            for i, post_neuron in enumerate(post_neurons):
                for j, pre_neuron in enumerate(pre_neurons):
                    change = weight_change[i, j].item()
                    pre_neuron.add_weight_change(change)
                    post_neuron.add_weight_change(change)

            # Update weights
            weights[layer_idx] = (weights[layer_idx].to(
                self.device) + weight_change).to(self.device)

        self.set_weights(weights)

    def get_weights(self):
        return [l.weight.data for l in self.network]

    def set_weights(self, weights):
        if isinstance(weights[0], torch.Tensor):
            for i, weight in enumerate(weights):
                self.network[i].weight = nn.Parameter(
                    weight.to(self.device).float(), requires_grad=self.grad)
        else:
            tmp = self.get_weights()
            start = 0
            for i, l in enumerate(tmp):
                size = l.size()[0] * l.size()[1] + start
                params = torch.tensor(weights[start:size], device=self.device, dtype=torch.float32)
                start = size
                self.network[i].weight = nn.Parameter(
                    torch.reshape(params, (l.size()[0], l.size()[1])),
                    requires_grad=self.grad
                )

    def set_params(self, params: list):
        """Set learning rates (etas) and Hebbian rules for all neurons."""
        start = 0
        
        # Input layer neurons (4 parameters each:eta, pre, corr, decorr, eta)
        for neuron in self.neurons[0]:
            pre = params[start]
            corr = params[start + 1]
            decorr = params[start + 2]
            eta = params[start + 3]
            
            neuron.set_hebbian_params(pre=pre, post=0.0, corr=corr, decorr=decorr)
            neuron.set_eta(eta)
            start += 4
        
        # Hidden layer neurons (5 parameters each: pre, post, corr, decorr, eta)
        for layer in self.neurons[1:-1]:
            for neuron in layer:
                pre = params[start]
                post = params[start + 1]
                corr = params[start + 2]
                decorr = params[start + 3]
                eta = params[start + 4]
                
                neuron.set_hebbian_params(pre=pre, post=post, corr=corr, decorr=decorr)
                neuron.set_eta(eta)
                start += 5
        
        # Output layer neurons (4 parameters each: post, corr, decorr, eta)
        for neuron in self.neurons[-1]:
            post = params[start]
            corr = params[start + 1]
            decorr = params[start + 2]
            eta = params[start + 3]
            
            neuron.set_hebbian_params(pre=0.0, post=post, corr=corr, decorr=decorr)
            neuron.set_eta(eta)
            start += 4

    def get_descriptors(self):
        # 1. Activation Diversity Descriptor 
        # Standard deviation of activation patterns across neurons (where pattern = mean of each neuron's activation history)
        act_pattern = []
        for neuron in self.all_neurons:
            if neuron.activations:
                act_mean = np.mean(neuron.activations)
                act_pattern.append(act_mean)
        # Standard deviation of the mean activation patterns            
        act_diversity = np.std(act_pattern) if len(act_pattern) > 0 else 0.0
        
        # 2. Weight Change Diversity Descriptor
        # Standard deviation of weight change magnitudes across neurons
        weight_changes = []
        for neuron in self.all_neurons:
            if neuron.weight_changes:
                weight_change = np.mean(np.abs(neuron.weight_changes))
                weight_changes.append(weight_change)
                
        # Standard deviation of the mean weight changes
        weight_diversity = np.std(weight_changes) if len(weight_changes) > 0 else 0.0
        
        return act_diversity, weight_diversity
    
    def save(self, path_dir=None, model_name="best_nchl.pkl"):
        """
        Save the network to a file.
        """
        with open(os.path.join(path_dir, model_name), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path_dir=None, model_name="best_nchl.pkl"):
        """
        Load the network from a file.
        """
        with open(os.path.join(path_dir, model_name), "rb") as f:
            return pickle.load(f)
        
    def __getstate__(self):
        """Return state values to be pickled."""
        state = {
            'nodes': self.nodes.tolist() if isinstance(self.nodes, torch.Tensor) else self.nodes,
            'nweights': self.nweights,
            'grad': self.grad,
            'device': 'cpu', 
            'nparams': self.nparams
        }

        # Store neuron states
        neurons_state = []
        for layer in self.neurons:
            layer_state = []
            for neuron in layer:
                # Get the neuron's state
                neuron_state = neuron.__getstate__()
                layer_state.append(neuron_state)
            neurons_state.append(layer_state)
        state['neurons_state'] = neurons_state

        # Store all neurons list (just their IDs for reconstruction)
        all_neurons_ids = [neuron.neuron_id for neuron in self.all_neurons]
        state['all_neurons_ids'] = all_neurons_ids

        # Store network weights rather than layer objects
        weights = []
        for layer in self.network:
            weights.append(layer.weight.data.detach().cpu().tolist())
        state['weights'] = weights

        return state

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        super(NCHL, self).__init__()

        # Restore basic attributes
        self.nodes = torch.tensor(state['nodes'], device='cpu', dtype=torch.float32)
        self.nweights = state['nweights']
        self.grad = state['grad']
        self.device = 'cpu' 
        self.nparams = state['nparams']

        # Reconstruct neurons
        self.neurons = []
        self.all_neurons = []

        # Recreate neurons using their stored states
        for layer_state in state['neurons_state']:
            layer_neurons = []
            for neuron_state in layer_state:
                neuron = Neuron(
                    neuron_id=neuron_state['neuron_id'], device='cpu')
                neuron.__setstate__(neuron_state)
                layer_neurons.append(neuron)
                self.all_neurons.append(neuron)
            self.neurons.append(layer_neurons)

        # Reconstruct network layers
        self.network = []
        for i in range(len(self.nodes) - 1):
            layer = nn.Linear(int(self.nodes[i]), int(self.nodes[i + 1]), bias=False)
            layer.float()  
            layer.to('cpu')
            weights = torch.tensor(state['weights'][i], device='cpu', dtype=torch.float32)
            layer.weight = nn.Parameter(weights, requires_grad=self.grad)
            self.network.append(layer)

        self.float() 
        self.to('cpu')  