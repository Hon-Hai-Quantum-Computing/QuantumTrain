# nohup python -u torchquantum_FF_mapping_model_L38_b1000_40_200_40.py > output_tq_FF_mm_L38_b1000_40_200_40.log &
# watch -n 0.5 tail -n 50 output_tq_FF_mm_L38_b1000_40_200_40.log


import time
import os
import copy

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms as transforms

## Distributed training
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# # Pennylane
# import pennylane as qml
# from pennylane import numpy as np
import torchquantum as tq
import numpy as np 



torch.manual_seed(42)
np.random.seed(42)

# Plotting
import matplotlib.pyplot as plt


import itertools

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


### Classical target model initialization ###

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 1

# Data loading and preprocessing for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # in[N, 3, 32, 32] => out[N, 16, 16, 16]
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.MaxPool2d(kernel_size=2)
        )
        # in[N, 16, 16, 16] => out[N, 32, 8, 8]
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.MaxPool2d(2)
        )
        # in[N, 32 * 8 * 8] => out[N, 128]
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
        )
        # in[N, 128] => out[N, 64]
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
        )
        # in[N, 64] => out[N, 10]
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # [N, 32 * 8 * 8]
        x = self.fc1(x)
        x = self.fc2(x)

        output = self.out(x)
        return output



# Instantiate the classical model, move it to GPU
model = SimpleCNN().to(device)

#############################################

### required qubits estimation ##############
# NN weights

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

#############################################

### Some tool function definition ###########
def probs_to_weights(probs_):

    new_state_dict = {}
    data_iterator = probs_.view(-1)

    for name, param in SimpleCNN().state_dict().items():
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

#############################################

### Main Learning-wise Hybridization model ##

class LewHybridNN(nn.Module):
    class QLayer(nn.Module):
        def __init__(self, n_blocks):
            super().__init__()
            self.n_wires = int(np.ceil(np.log2(len(nw_list_normal)))),
            self.n_wires = self.n_wires[0]
            self.n_blocks = n_blocks
            self.u3_layers = tq.QuantumModuleList()
            self.cu3_layers = tq.QuantumModuleList()
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
    
    
    class ConvMappingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
            self.fc1 = nn.Linear(32 * 7, 32)  # Adjust the size accordingly
            self.fc2 = nn.Linear(32, 1)  # num_classes is for final output

        def forward(self, x):
            x = self.conv1(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = x.view(-1, 32 * 7)  # Flatten the array
            x = self.fc1(x)
            x = self.fc2(x)
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

    def __init__(self):
        """
        Definition of the *dressed* layout.
        """
        super().__init__()
        self.MappingNetwork = self.MappingModel(n_qubit+1, [40, 200, 40], 1).to(device)  
        self.QuantumNN = self.QLayer(q_depth).to(device)   #arch={"n_blocks": q_depth})
    
    def forward(self, x):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """
        device = x.device

        probs_ = self.QuantumNN()
        probs_ = probs_[:len(nw_list_normal)]
        # Generate qubit states using PyTorch
        qubit_states_torch = generate_qubit_states_torch(n_qubit)[:len(nw_list_normal)]
        qubit_states_torch = qubit_states_torch.to(device)

        # Combine qubit states with probability values using PyTorch
        combined_data_torch = torch.cat((qubit_states_torch, probs_), dim=1)
        combined_data_torch = combined_data_torch.reshape(285226, 1, 20)
        
        prob_val_post_processed = self.MappingNetwork(combined_data_torch)
        prob_val_post_processed = prob_val_post_processed - prob_val_post_processed.mean()
        
        state_dict = probs_to_weights(prob_val_post_processed)

        ######## 
            
        dtype = torch.float32  # Ensure all tensors are of this type

        # Convolution 1
        weight = state_dict['conv1.0.weight'].to(device).type(dtype)
        bias = state_dict['conv1.0.bias'].to(device).type(dtype)
        x = F.conv2d(x, weight, bias, stride=1, padding=2)
        x = F.max_pool2d(x, kernel_size=2)

        # Convolution 2
        weight = state_dict['conv2.0.weight'].to(device).type(dtype)
        bias = state_dict['conv2.0.bias'].to(device).type(dtype)
        x = F.conv2d(x, weight, bias, stride=1, padding=2)
        x = F.max_pool2d(x, kernel_size=2)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected 1
        weight = state_dict['fc1.0.weight'].to(device).type(dtype)
        bias = state_dict['fc1.0.bias'].to(device).type(dtype)
        x = F.linear(x, weight, bias)

        # Fully connected 2
        weight = state_dict['fc2.0.weight'].to(device).type(dtype)
        bias = state_dict['fc2.0.bias'].to(device).type(dtype)
        x = F.linear(x, weight, bias)

        # Output layer
        weight = state_dict['out.weight'].to(device).type(dtype)
        bias = state_dict['out.bias'].to(device).type(dtype)
        x = F.linear(x, weight, bias)

    
        return x 
    
#############################################

### Training setting ########################

step = 1e-4                 # Learning rate
batch_size = 1000            # Number of samples for each training step
num_epochs = 1000             # Number of training epochs
q_depth = 19*2             # Number of variational layers
gamma_lr_scheduler = 0.1    # Learning rate reduction applied every 10 epochs.
q_delta = 0.1              # Initial spread of random quantum weights


train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Instantiate the model, move it to GPU, and set up loss function and optimizer
model = LewHybridNN().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=step, weight_decay=1e-5, eps=1e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, verbose = True, factor = 0.5)  # 'min' because we're minimizing loss

num_trainable_params_MM = sum(p.numel() for p in LewHybridNN.MappingModel(n_qubit+1,  [40, 200, 40], 1).parameters() if p.requires_grad)
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("# of trainable parameter in Mapping model: ", num_trainable_params_MM)
print("# of trainable parameter in QNN model: ", num_trainable_params - num_trainable_params_MM)
print("# of trainable parameter in full model: ", num_trainable_params)


#############################################
### Training loop ###########################
### (Optional) Start from pretrained model ##
# model = torch.load('result_FF_mm_b1000_40_200_40/tq_mm_acc_70_bsf')
# model.eval()  # Set the model to evaluation mode
#############################################

loss_list = [] 
acc_list = [] 
acc_best = 0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        correct = 0
        total = 0
        since_batch = time.time()
        
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        labels_one_hot = F.one_hot(labels, num_classes=10).float()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # Compute loss
        loss = criterion(outputs, labels_one_hot)
        
        loss_list.append(loss.cpu().detach().numpy())
        acc = 100 * correct / total
        acc_list.append(acc)
        train_loss += loss.cpu().detach().numpy()
        
        # np.array(loss_list).dump("result_FF_mm_L38_b1000_40_200_40/loss_list.dat")
        # np.array(acc_list).dump("result_FF_mm_L38_b1000_40_200_40/acc_list.dat")
        if acc > acc_best:
            # torch.save(model, 'result_FF_mm_L38_b1000_40_200_40/tq_mm_acc_'+str(int(acc))+'_bsf')
            acc_best = acc
        # Backward pass and optimization
        loss.backward()
        
        optimizer.step()
        # if (i+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, batch time: {time.time() - since_batch:.2f}, accuracy:  {(acc):.2f}%")
    
    train_loss /= len(train_loader)
    scheduler.step(train_loss)
    
#############################################