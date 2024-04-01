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



def QuantumTrain(model, n_qubit, nw_list_normal, q_depth, device):
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
            
            model_recontruct = model.to(device)
            model_recontruct.load_state_dict(state_dict)    
            model_recontruct.eval() 
            
            for param in model_recontruct.parameters():
                param.requires_grad = False
                
            x  = model_recontruct(x)
            
            
            return x 
        
    return LewHybridNN

def spsa_step(model, loss_fn, inputs, targets, lr=0.01, epsilon=1e-4, blocking_threshold=1e-2):
    """ Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer
    Perform SPSA parameter update for QNN parameters.

    Args:
        model (torch.nn.Module): The model to be optimized.
        loss_fn (callable): The loss function.
        inputs (torch.Tensor): Input data batch.
        targets (torch.Tensor): Target data batch.
        lr (float): Learning rate.
        epsilon (float): Perturbation size.
    """

    params = torch.nn.utils.parameters_to_vector(model.parameters())
    # params_with_no_grad = [p for p in model.parameters() if not p.requires_grad]
    # params = torch.nn.utils.parameters_to_vector(params_with_no_grad)


    # Create a random perturbation vector
    delta = torch.randint(low=0, high=2, size=params.shape, device=params.device).float() * 2 - 1  # Random {-1, 1}

    # Perturb parameters positively and negatively
    params_plus = params + epsilon * delta
    params_minus = params - epsilon * delta

    # Update model parameters to compute loss for perturbed parameters
    torch.nn.utils.vector_to_parameters(params_plus, params)
    outputs_plus = model(inputs)
    loss_plus = loss_fn(outputs_plus, targets)

    torch.nn.utils.vector_to_parameters(params_minus, params)
    outputs_minus = model(inputs)
    loss_minus = loss_fn(outputs_minus, targets)

    # Approximate the gradient
    grad_approx = (loss_plus - loss_minus) / (2 * epsilon * delta)

    # Update the parameters using the approximated gradient
    updated_params = params - lr * grad_approx

    # Apply the updated parameters to the model
    torch.nn.utils.vector_to_parameters(updated_params, params)

