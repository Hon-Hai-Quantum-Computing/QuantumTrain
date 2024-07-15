import numpy as np 
import torch
import torch.nn as nn
import torchquantum as tq
import torch.nn.functional as F

# Function to estimate the required number of qubits for a given model
def required_qubits_estimation(model):
    numpy_weights = {}  # Dictionary to store model parameters as numpy arrays
    nw_list = []  # List to store flattened model parameters
    nw_list_normal = []  # List to store all parameters in a single list

    # Convert model parameters to numpy arrays and store in dictionary
    for name, param in model.state_dict().items():
        numpy_weights[name] = param.cpu().numpy()
    
    # Flatten each parameter and add to nw_list
    for i in numpy_weights:
        nw_list.append(list(numpy_weights[i].flatten()))
    
    # Flatten the nw_list into a single list
    for i in nw_list:
        for j in i:
            nw_list_normal.append(j)
    
    # Print the total number of parameters
    print("# of NN parameters: ", len(nw_list_normal))
    
    # Calculate the required number of qubits
    n_qubits = int(np.ceil(np.log2(len(nw_list_normal))))
    print("Required qubit number: ", n_qubits)

    return n_qubits, nw_list_normal

# Function to convert probabilities to model weights
def probs_to_weights(probs_, model):
    new_state_dict = {}
    data_iterator = probs_.view(-1)  # Flatten the probabilities tensor

    # Iterate over model parameters and assign new weights from data_iterator
    for name, param in model.state_dict().items():
        shape = param.shape
        num_elements = param.numel()
        chunk = data_iterator[:num_elements].reshape(shape)
        new_state_dict[name] = chunk
        data_iterator = data_iterator[num_elements:]
        
    return new_state_dict

# Function to generate all possible qubit states
def generate_qubit_states_torch(n_qubit):
    # Create a tensor of shape (2**n_qubit, n_qubit) with all possible combinations of 0 and 1
    all_states = torch.cartesian_prod(*[torch.tensor([-1, 1]) for _ in range(n_qubit)])
    return all_states

# Function to apply a layer to the input tensor
def apply_layer(x, layer_config, state_dict, device, dtype):
    layer_type = layer_config["type"]
    name       = layer_config["name"]
    
    # Apply the specified layer based on its type
    if layer_type == "Conv2d":
        weight = state_dict[f'{name}.weight'].to(device).type(dtype)
        bias = state_dict[f'{name}.bias'].to(device).type(dtype)
        x = F.conv2d(x, weight, bias, **layer_config["params"])
        
    elif layer_type == "MaxPool2d":
        x = F.max_pool2d(x, **layer_config["params"])
        
    elif layer_type == "Linear":
        weight = state_dict[f'{name}.weight'].to(device).type(dtype)
        bias = state_dict[f'{name}.bias'].to(device).type(dtype)
        x = F.linear(x, weight, bias)
    
    elif layer_type == "Flatten":
        x = x.view(x.size(0), -1)
    
    elif layer_type == "ReLU":
        x = F.relu(x)
    
    elif layer_type == "Sigmoid":
        x = F.sigmoid(x)
    
    elif layer_type == "Tanh":
        x = F.tanh(x)
    
    elif layer_type == "LeakyReLU":
        x = F.leaky_relu(x, negative_slope=0.01)
    
    elif layer_type == "Softmax":
        x = F.softmax(x, dim=-1)
    
    elif layer_type == "Softplus":
        x = F.softplus(x)
    
    return x

# Function to extract network configuration from a model
def network_config_extract(model):
    network_config = []

    # Iterate over model layers and extract their configuration
    for name, layer in model.named_modules():
        layer_type = str(type(layer)).split('.')[-1].split("'")[0]
        config = {"type": layer_type, "name": name, "params": {}}
        
        if hasattr(layer, 'kernel_size'):
            kernel_size = layer.kernel_size
            if type(kernel_size) == int:
                config["params"]["kernel_size"] = kernel_size 

        if hasattr(layer, 'stride'):
            stride = layer.stride
            if type(stride) == int:
                config["params"]["stride"] = stride 
            elif type(stride)  != int:
                config["params"]["stride"] = stride[0]
                
        network_config.append(config)
    
    return network_config

# Quantum training class
class QuantumTrain(nn.Module):
    def __init__(self, model, q_depth, device):
        super().__init__()
        self.model          = model 
        self.q_depth        = q_depth 
        self.device         = device
        self.network_config = network_config_extract(model)
        self.n_qubit, self.nw_list_normal = required_qubits_estimation(model)

        # Initialize mapping network and quantum neural network layers
        self.MappingNetwork = self.MappingModel(self.n_qubit+1, [4, 20, 4], 1).to(device)  
        self.QuantumNN = self.QLayer(self.q_depth, self.nw_list_normal, self.n_qubit).to(self.device) 
        

    class QLayer(nn.Module):
        def __init__(self, n_blocks, nw_list_normal_, n_qubit_):
            super().__init__()
            self.n_wires = int(np.ceil(np.log2(len(nw_list_normal_)))),
            self.n_qubit_ = n_qubit_
            self.nw_list_normal_ = nw_list_normal_
            self.n_wires = self.n_wires[0]
            self.n_blocks = n_blocks
            self.u3_layers = tq.QuantumModuleList()
            self.cu3_layers = tq.QuantumModuleList()

            # Initialize quantum layers
            for _ in range(self.n_blocks):
                self.u3_layers.append(
                    tq.Op1QAllLayer(
                        op=tq.U3,
                        n_wires=self.n_wires,
                        has_params=True,
                        trainable=True,
                    )
                )
                self.cu3_layers.append(
                    tq.Op2QAllLayer(
                        op=tq.CU3,
                        n_wires=self.n_wires,
                        has_params=True,
                        trainable=True,
                        circular=True,
                    )
                )
                
        def forward(self):
            # Initialize a quantum device
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires, bsz=1, device=next(self.parameters()).device
            )
            easy_scale_coeff = 2**(self.n_qubit_-1)
            gamma = 0.1
            beta  = 0.8
            alpha = 0.3
            
            # Apply quantum layers to the quantum device
            for k in range(self.n_blocks):
                self.u3_layers[k](qdev)
                self.cu3_layers[k](qdev)
                
            # Get the state magnitude from the quantum device
            state_mag = qdev.get_states_1d().abs()[0] 
            state_mag = state_mag[:len(self.nw_list_normal_)]
            x = torch.abs(state_mag) ** 2
            x = x.reshape(len(self.nw_list_normal_),1)
            x = (beta*torch.tanh(gamma*easy_scale_coeff*x))**(alpha) 
            x = x - torch.mean(x)
            return x
        
        
    class MappingModel(nn.Module):
        def __init__(self, input_size, hidden_sizes, output_size):
            super().__init__()
            # Initialize layers: an input layer, multiple hidden layers, and an output layer
            self.input_layer = nn.Linear(input_size, hidden_sizes[0])
            self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
            self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
            
        def forward(self, X):
            # Ensure the input tensor is the same type as the weights
            X = X.type_as(self.input_layer.weight)

            # Input layer with ReLU activation
            X = self.input_layer(X)

            # Hidden layers with ReLU activation
            for hidden in self.hidden_layers:
                X = hidden(X)

            # Output layer with linear activation
            output = self.output_layer(X)
            return output
    
    def forward(self, x):
        device_ = x.device

        # Get probabilities from the QuantumNN
        probs_ = self.QuantumNN()
        probs_ = probs_[:len(self.nw_list_normal)]
        
        # Generate qubit states and combine with probabilities
        qubit_states_torch = generate_qubit_states_torch(self.n_qubit)[:len(self.nw_list_normal)]
        qubit_states_torch = qubit_states_torch.to(device_)
        combined_data_torch = torch.cat((qubit_states_torch, probs_), dim=1)
        combined_data_torch = combined_data_torch.reshape(len(self.nw_list_normal), 1, self.n_qubit+1)
        
        # Post-process the combined data
        prob_val_post_processed = self.MappingNetwork(combined_data_torch)
        prob_val_post_processed = prob_val_post_processed - prob_val_post_processed.mean()
        
        # Convert probabilities to model weights
        state_dict = probs_to_weights(prob_val_post_processed, self.model)
        # Apply layers to the input tensor
        dtype = torch.float32
        for layer in self.network_config:
            x = apply_layer(x, layer, state_dict, device_, dtype)
        
        return x 
