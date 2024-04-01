import numpy as np 
import torch
import torch.nn as nn
import torchquantum as tq
import torch.nn.functional as F


def required_qubits_estimation(model):
    numpy_weights = {}
    nw_list = [] 
    nw_list_normal = []
    for name, param in model.state_dict().items():
        numpy_weights[name] = param.cpu().numpy()
    for i in numpy_weights:
        nw_list.append(list(numpy_weights[i].flatten()))
    for i in nw_list:
        for j in i:
            nw_list_normal.append(j)
    print("# of NN parameters: ", len(nw_list_normal))
    n_qubits = int(np.ceil(np.log2(len(nw_list_normal))))
    print("Required qubit number: ", n_qubits)

    n_qubit = n_qubits
    
    return n_qubit, nw_list_normal

### Some tool function definition ###########
def probs_to_weights(probs_, model):

    new_state_dict = {}
    data_iterator = probs_.view(-1)

    for name, param in model.state_dict().items():
        shape = param.shape
        num_elements = param.numel()
        chunk = data_iterator[:num_elements].reshape(shape)
        new_state_dict[name] = chunk
        data_iterator = data_iterator[num_elements:]
        
    return new_state_dict

def generate_qubit_states_torch(n_qubit):
    # Create a tensor of shape (2**n_qubit, n_qubit) with all possible combinations of 0 and 1
    all_states = torch.cartesian_prod(*[torch.tensor([-1, 1]) for _ in range(n_qubit)])
    return all_states

def apply_layer(x, layer_config, state_dict, device, dtype):
    
    layer_type = layer_config["type"]
    name       = layer_config["name"]
    
    ## Layers ## 
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
    
    ## Flatten operation ##
    elif layer_type == "Flatten":
        x = x.view(x.size(0), -1)
    
    ## Activation function ##
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



def network_config_extract(model):
    network_config = []

    for name, layer in model.named_modules():
        layer_type = str(type(layer)).split('.')[-1].split("'")[0]  # Extracting the layer type
        config = {"type": layer_type, "name": name, "params": {}}
        
        # Handling kernel_size
        if hasattr(layer, 'kernel_size'):
            kernel_size = layer.kernel_size
            if type(kernel_size) == int:
                config["params"]["kernel_size"] = kernel_size #kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
            # elif type(kernel_size) != int:
            #     config["params"]["kernel_size"] = kernel_size
                
        # Handling stride
        if hasattr(layer, 'stride'):
            stride = layer.stride
            if type(stride) == int:
                config["params"]["stride"] = stride #if isinstance(stride, tuple) else (stride, stride)
            elif type(stride)  != int:
                config["params"]["stride"] = stride[0]
                
        # Handling in_features and out_features for linear layers
        # if hasattr(layer, 'in_features'):
        #     config["params"]["in_features"] = layer.in_features
            
        # if hasattr(layer, 'out_features'):
        #     config["params"]["out_features"] = layer.out_features
        
        network_config.append(config)
    
    return network_config

def QuantumTrain(model, n_qubit, nw_list_normal, q_depth, device, network_config):
    class LewHybridNN(nn.Module):
        class QLayer(nn.Module):
            def __init__(self, n_blocks):
                super().__init__()
                self.n_wires = int(np.ceil(np.log2(len(nw_list_normal)))),
                self.n_wires = self.n_wires[0]
                self.n_blocks = n_blocks
                self.u3_layers = tq.QuantumModuleList()
                self.cu3_layers = tq.QuantumModuleList()
                # self.measure = tq.MeasureAll(tq.PauliZ)
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
                qdev = tq.QuantumDevice(
                    n_wires=self.n_wires, bsz=1, device=next(self.parameters()).device
                )
                easy_scale_coeff = 2**(n_qubit-1)
                gamma = 0.1
                beta  = 0.8
                alpha = 0.3
                for k in range(self.n_blocks):
                    self.u3_layers[k](qdev)
                    self.cu3_layers[k](qdev)
                    
                state_mag = qdev.get_states_1d().abs()[0] 
                state_mag = state_mag[:len(nw_list_normal)]
                x = torch.abs(state_mag) ** 2
                # x = torch.log(x)
                x = x.reshape(len(nw_list_normal),1)
                x = (beta*torch.tanh(gamma*easy_scale_coeff*x))**(alpha) 
                x = x - torch.mean(x)
                x.to(device)
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
                # output = F.tanh(output)  # It's often better to use ReLU or similar; tanh is used here as it was in the original model.
                return output

        def __init__(self):
            """
            """
            super().__init__()
            self.MappingNetwork = self.MappingModel(n_qubit+1, [4, 20, 4], 1).to(device)  
            self.QuantumNN = self.QLayer(q_depth).to(device)   #arch={"n_blocks": q_depth})
            # self.reconstruct_model = model.to(device)
        
        def forward(self, x):
            """
            """
            device = x.device

            probs_ = self.QuantumNN()
            probs_ = probs_[:len(nw_list_normal)]
            
            # Generate qubit states using PyTorch
            qubit_states_torch = generate_qubit_states_torch(n_qubit)[:len(nw_list_normal)]
            qubit_states_torch = qubit_states_torch.to(device)
            # Combine qubit states with probability values using PyTorch
            
            combined_data_torch = torch.cat((qubit_states_torch, probs_), dim=1)
            combined_data_torch = combined_data_torch.reshape(len(nw_list_normal), 1, n_qubit+1)
            
            prob_val_post_processed = self.MappingNetwork(combined_data_torch)
            prob_val_post_processed = prob_val_post_processed - prob_val_post_processed.mean()
            
            state_dict = probs_to_weights(prob_val_post_processed, model)
            
            
            dtype = torch.float32
            for layer in network_config:
                x = apply_layer(x, layer, state_dict, device, dtype)
            
            return x 
        
    return LewHybridNN


"""
network_config = []

for name, layer in model.named_modules():
    layer_type = str(type(layer)).split('.')[-1].split("'")[0]  # Extracting the layer type
    config = {"type": layer_type, "name": name, "params": {}}
    
    # Handling kernel_size
    if hasattr(layer, 'kernel_size'):
        kernel_size = layer.kernel_size
        config["params"]["kernel_size"] = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    
    # Handling stride
    if hasattr(layer, 'stride'):
        stride = layer.stride
        config["params"]["stride"] = stride if isinstance(stride, tuple) else (stride, stride)

    # Handling in_features and out_features for linear layers
    # if hasattr(layer, 'in_features'):
    #     config["params"]["in_features"] = layer.in_features
        
    # if hasattr(layer, 'out_features'):
    #     config["params"]["out_features"] = layer.out_features
    
    network_config.append(config)
    
    
for i in network_config:
    print(i)


# Example network configuration
# network_config = [
#     {"type": "conv2d", "params": {"in_channels": 1, "out_channels": 32, "kernel_size": 3}},
#     {"type": "max_pool2d", "params": {"kernel_size": 2, "stride": 2}},
#     {"type": "conv2d", "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3}},
#     {"type": "max_pool2d", "params": {"kernel_size": 2, "stride": 2}},
#     {"type": "flatten"},
#     {"type": "linear", "params": {"in_features": 64*6*6, "out_features": 128}},
#     {"type": "linear", "params": {"in_features": 128, "out_features": 10}}
# ]




            # model_recontruct = self.reconstruct_model
            # model_recontruct.load_state_dict(state_dict)    
            # self.reconstruct_model.load_state_dict(state_dict)    
            # model_recontruct.eval() 
            # for param in model_recontruct.parameters():
            #     param.requires_grad = False
            # for name, param in self.reconstruct_model.named_parameters():
            #     if name in state_dict:
            #         param.data.copy_(state_dict[name])
            # # This resolve the problem that loading model in forward pass vanishing the gradient
            # x  = self.reconstruct_model(x) # .detach() 
            # x = x * prob_val_post_processed[-1] 
            # x = x / prob_val_post_processed[-1]



            ##############################
            
            # dtype = torch.float32  # Ensure all tensors are of this type
            
            conv1_weight = state_dict['conv1.weight'].to(device).type(dtype)
            conv1_bias = state_dict['conv1.bias'].to(device).type(dtype)
            conv2_weight = state_dict['conv2.weight'].to(device).type(dtype)
            conv2_bias = state_dict['conv2.bias'].to(device).type(dtype)
            fc1_weight = state_dict['fc1.weight'].to(device).type(dtype)
            fc1_bias = state_dict['fc1.bias'].to(device).type(dtype)
            fc2_weight = state_dict['fc2.weight'].to(device).type(dtype)
            fc2_bias = state_dict['fc2.bias'].to(device).type(dtype)
            
            x = F.conv2d(x, conv1_weight, conv1_bias, stride=1)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.conv2d(x, conv2_weight, conv2_bias, stride=1)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = x.view(x.size(0), -1)
            x = F.linear(x, fc1_weight, fc1_bias)
            x = F.linear(x, fc2_weight, fc2_bias)
            
            ##############################

for  name, layer in model.named_modules():
    if hasattr(layer, 'kernel_size'):
        print(layer.kernel_size)
    if hasattr(layer, 'stride'):
        print(layer.stride)
        
        
"""