
# Quantum-Train Toolkit
### *Reducing the training parameters in Classical ML model by Quantum Machine Learning*


## Project description 
In this project, we propose a new perspective on hybrid quantum-classical machine learning (QCML) that involves training classical neural networks (NNs) by mapping $M$ classical NN weights into an $\lceil \log_2 M \rceil$-qubit parameterized quantum state with  $2^{\lceil \log_2 M \rceil} \sim M$ amplitudes. Assuming that our parameterized quantum state, or quantum neural network (QNN), has a polynomial number of layers, the action of tuning this QNN with $\text{PolyLog}(M)$ parameters effectively tunes a classical NN with $M$ parameters. 

In practice, more specifically, we need an additional mapping model (which also has  parameters) to map the probability outputs to NN weights. That is, we could train the same classical NN with a PolyLog parameter reduction. As shown in below : 

<img src="images/training_flow.png" width="800px" align="center">

Next, we apply the QT flow described above to the phase identification problem in quantum many-body physics (A Matter of Taste challenge). Using the quantum dataset provided by PennyLane, we train a classifier that, given the classical shadow measurement result and the corresponding basis information, outputs the label of the phase, as shown in


## Why it is interesting

Besides the PolyLog reduction behavior of the proposed approach, one may also observe that we are tackling three of the main challenges in the QCML domain.

* The data embedding issue is resolved: 
Firstly, since we are essentially generating classical NN weights through a QNN, the input and output of the ML model are entirely classical. As such, we don't have to worry about the data embedding issues associated with QNNs (for example, the need for data-loading QNN layers that impose constraints on input data size or require data compression).

* Inference without quantum computers:
The trained model is compatible with classical hardware. In fact, model inference relies solely on classical computers, enhancing its practicality, especially given the limited availability of quantum computers compared to classical ones. However, it's important to remember that we still benefit from quantum computing for its PolyLog reduction in the training process, which stems from the exponentially large Hilbert space.

* General approach for QCML, QML, and ML:
Although our example in the QHack 2024 is the application to the quantum many-body physics, the proposed approach in Fig. nv. 1 is actually a general training flow for ANY QCML, QML, and ML use case, with PolyLog parameter reduction behavior. In the future, we would also like to tackle larger ML model that the parameter reduction will make the applicability totally different.  

## CUDA Quantum and NVIDIA GPUs. 

We conducted our numerical simulations of QCML primarily using TorchQuantum with NVIDIA GPUs. In fact, we experimented with Qiskit-gpu built with cuQuantum, PennyLane with Lightning.gpu (also built on cuQuantum), and cuTensorNet, and found that TorchQuantum is the most suitable for our project. It's worth noting that we also provide a PennyLane version of the code, although some details may differ. The NVIDIA GPUs we used are the NVIDIA RTX3080 and RTX3080ti with CUDA 11.6.

## Performance result, numerical / scientific analysis. 
<img src="images/result_group2.png" width="800px" align="center">
<img src="images/result_group.png" width="800px" align="center">

The use of NVIDIA GPUs led to a training speedup of up to 3.28X for the QT model proposed in this hackathon project, compared to training with a CPU, as illustrated in (a). This speedup is significant, especially given the extensive volume of training data and the prolonged duration of the training process in certain instances.

For numerical and scientific analysis, we provide some preliminary results on the TFIM. We have collected 4,000 data points for each classification task, with 20% designated as testing data and the remaining 80% used for training. Each data point consists of a matrix of size , where  represents the number of spins in this case. The  is derived from  samples of the classical shadow measurement, while  is obtained from the combined information of both the measurement result and the basis. The labels correspond to the phase of the system.

Results for the TFIM with different system layouts are presented in (b), (c), and (d), where the topology of the QT (QNN + Mapping Model), the test accuracy, and the number of parameters (parameter size) are shown, as well as the target classical model. As one may observe, the parameter size is reduced to about 12.9% for a 1x4 layout with QT-14, 17% for a 1x8 layout with QT-14, and 5.4% for a 1x16 layout with QT-16, while the test accuracies remain at a similar level.

## On MNIST, FashionMNIST, CIFAR-10 datasets 

For those interested, we also present additional results of the QT approach applied to well-known datasets such as MNIST, FashionMNIST, and CIFAR-10. The codes to generate these result are provided in the `appendix_example` folder. Note that there is a pennylane version of the code for CIFAR-10 dataset, and only torchquantum version for others.


<img src="images/result_1.png" width="800px" align="center">
<img src="images/result_group3.png" width="800px" align="center">

## Future direction 

* ###  Smart quantum state tomography method
     It will be great to reconstruct the quantum state with as few as possible measurement shots in practical .
* ### Optimization algorithm investigation
     Efficient parameter update is crucial

* ### Fine-tuning Large classical ML models   
    * Parameter reduction
    * Practical in real world
    * No need quantum computers for inference

## Installation
```bash
conda create --name tqtest python=3.9
pip Install qiskit==0.45.0 qiskit_ibm_runtime==0.16.0 qiskit-qer==0.13.0
git clone https://github.com/mit-han-lab/torchquantum.git
cd torchquantum
pip install --editable .
```

## Citation

This project is based on and has been advanced beyond the work described in the paper:  
* [Training Classical Neural Networks by Quantum Machine Learning](https://arxiv.org/abs/2402.16465)  
***Chen-Yu Liu***, En-Jui Kuo, Chu-Hsuan Abraham Lin, Sean Chen, Jason Gemsun Young, Yeong-Jar Chang, Min-Hsiu Hsieh  
arXiv preprint, arXiv: 2402.16465 (2024)

```
@misc{liu2024training,
      title={Training Classical Neural Networks by Quantum Machine Learning}, 
      author={Chen-Yu Liu and En-Jui Kuo and Chu-Hsuan Abraham Lin and Sean Chen and Jason Gemsun Young and Yeong-Jar Chang and Min-Hsiu Hsieh},
      year={2024},
      eprint={2402.16465},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
