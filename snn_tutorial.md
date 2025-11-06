# Building, Training, and Deploying Spiking Neural Networks from Scratch

## Model Overview

### Introduction

Spiking Neural Networks (SNNs) are brain-inspired models that possess more biologically realistic characteristics compared to artificial neural network models. SNNs exhibit features such as state memory, analog computation, and binary spike transmission, enabling them to generate complex dynamical behaviors, particularly suitable for processing spatiotemporal information. With the advancement of computational neuroscience and artificial intelligence, the development of SNNs primarily focuses on aspects such as model design, encoding techniques, and training algorithms.

### Models, Encoding, and Algorithms

Research on SNN models encompasses the study of neurons and network models. The Hodgkin-Huxley (H-H) neuron model explains the generation of action potentials by modeling the electrophysiological properties of ion channels. However, due to high computational complexity and difficulties in scaling to large-scale networks, more convenient phenomenological models have emerged, such as the Integrate-and-Fire (IF) neuron model. The IF model uses a linear differential equation to describe membrane potential and possesses temporal and spatial integration capabilities, as well as spike generation based on threshold comparison. Based on the IF model, the Leaky Integrate-and-Fire (LIF) neuron introduces a leakage term to enhance the discrimination of input timing while maintaining computational simplicity and biological realism. However, the LIF model cannot produce complex spike response adaptation or bursting behavior. The Adaptive Exponential (AdEx) model incorporates threshold adaptation and exponential dependence to generate complex spike firing behavior. The Izhikevich model also models threshold adaptation using quadratic dependence and can simulate various firing patterns of cortical neurons.

The development of SNN models is primarily concentrated in the fields of computational neuroscience and artificial intelligence. In computational neuroscience, the Synfire Chain model is a classic multi-layer feedforward spike network that achieves synchronous propagation of spikes through interlayer excitatory connections, enabling functions such as signal transmission and state memory. Inspired by deep artificial neural networks, artificial intelligence incorporates fully connected, convolutional, and gating structures into SNN models, achieving good pattern recognition performance. Additionally, SNNs also support classical structures such as residual connections, batch normalization, and attention modules, contributing to the development of deep spiking neural networks.

The encoding mechanism of SNNs is crucial for representing, storing, and processing information within the network. For non-spike-form input such as real-valued vectors, spike encoding is required in SNNs. The main encoding methods include time encoding and rate encoding. Time encoding utilizes precise spike timing to encode information, while rate encoding uses the frequency of spike firing within a certain time window. Rate encoding has strong robustness but lower efficiency and higher latency. In contrast, time encoding can carry more information but is susceptible to noise interference affecting the accuracy of encoding. Therefore, in general scenarios, rate encoding is suitable for situations requiring high robustness, while time encoding is suitable for scenarios that demand fast response and encoding of large amounts of information. Common time encoding methods include First-Spike-Time Encoding, Spike Order Encoding, Time Delay Encoding, Phase Encoding, and Synchrony Encoding. In addition to encoding with a single spike sequence, multiple spike sequences can be used for encoding, known as population encoding. Multiple synchronous encoding is a type of population time encoding that utilizes connected neurons to generate precise spike time patterns for encoding. Overall, time encoding can be seen as rate encoding with extremely small time windows, so they are not clearly distinct from each other to some extent.

The training methods of SNNs can be categorized into biologically inspired synaptic plasticity mechanisms, direct supervised learning, and pre-training using artificial neural network conversion. Hebbian learning is a classic synaptic plasticity mechanism that updates synaptic weights based on the relationship of spike firing between neurons. Direct supervised learning algorithms update network weights using optimization methods such as gradient descent to achieve learning, requiring approximate methods to obtain the gradient of SNNs. The Spike-Time-Dependent Backpropagation (STBP) algorithm, proposed in 2018, trains deep SNNs by approximating spike firing threshold functions and time unfolding, achieving good performance. Pre-training network conversion methods transform a pre-trained artificial neural network into an equivalent structured SNN with performance close to the source network. However, conversion methods have slower convergence, higher computational cost, significant latency, and difficulties in handling dynamic spike sequences.

To obtain a complete SNN model, various aspects need to be determined, including neuron models, network structures, encoding/decoding schemes, and training algorithms. In the following implementation, the LIF neuron model is selected, the fully connected structure is used for the network, rate encoding is chosen for encoding/decoding, and a backpropagation-based surrogate gradient descent algorithm is employed for training. The model is trained and tested on the MNIST image classification dataset. The differential equation for the LIF neuron model is described as follows:

$$
\tau \frac{{\rm d}u(t)}{{\rm d}t}=-\left(u(t)-u_{\rm reset}\right)+wx(t)
$$

$$
 {\rm if}\; u(t)>V_{\rm th} \; {\rm then} \; u(t)\leftarrow u_{\rm reset};\; y(t)\leftarrow \delta(t)
$$


Here, $u(t)$represents the membrane potential, $u_{\rm reset}$ is the reset voltage, $\tau$ is the time constant, and $V_{\rm th}$ is the the threshold voltage. $y(t)$ and $x(t)$ are the spike sequence output and input, respectively. The iterative form of the single LIF neuron model after time discretization is as follows:

$$
V_{i+1}=V_{i}\lambda\left(1-o_{i}\right)+w\ast s_{i+1}
$$

$$
o_{i}=\mathcal{H}\left(V_{i}-V_{\rm th} \right)
$$

$$
\lambda={\rm e}^{-\frac{T_{\rm d}}{\tau}}
$$

Here, $V_i$，$o_i$ and $s_{i}$ represent the membrane potential, spike output, and spike input at the $i$ time step, respectively.$\mathcal{H}$ denotes the unit step function, and$\lambda$ is the decay coefficient of the membrane potential.

## Model Construction

In the following section, we will build an SNN model from scratch using the PyTorch framework.

### Custom Implementation of Surrogate Gradient Function

The commonly used unit step function for spike activation in SNNs is not differentiable at $x=0$. One solution is to use a surrogate gradient approach, where the activation function is smoothed during backpropagation to obtain derivatives. For example, we can replace the discontinuity at $x=0$ with a finite-slope ramp of a certain span, and the gradient becomes a rectangular window corresponding to that span. To achieve this, we need to customize the pulse activation function in PyTorch and provide its forward and backward computation methods. Here is an implementation example:

```python
class Spike_Act_with_Surrogate_Gradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, vth):
        ctx.save_for_backward(input)
        ctx.vth = vth
        return input.gt(vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        lens = 0.5
        input, = ctx.saved_tensors
        vth = ctx.vth
        grad_input = grad_output.clone()
        # the derivative of the surrogate function is a window function with a
        # width of 2*lens
        temp = abs(input - vth) < lens
        return grad_input * temp.float() / (lens * 2), None
```

In the given code snippet, the parameter $lens$ determines the span of the rectangular window that has gradients. $spike_activation$ is an encapsulated spike activation function, and $vth$ represents the threshold at which a neuron fires.

### Definition of the Spiking Neural Network class

Next, we define the Spiking Neural Network class, which includes network structure definition, membrane potential initialization, membrane potential update, and the single forward propagation process of the network. Here is an example of a three-layer feedforward fully connected spiking neural network definition:

```python
class Spiking_Network_1step(nn.Module):
    def __init__(self, num_classes, image_size, channel,
                 vth, decay, activation, on_apu):
        super(Spiking_Network_1step, self).__init__()
        # Spike_Act is the activation function
        self.Spike_Act = activation
        self.on_apu = on_apu
        self.channel = channel
        self.num_classes = num_classes

        # vth and decay are the parameters of the LIF neuron
        self.vth = nn.Parameter(vth * torch.ones(1), requires_grad=False)
        self.decay = nn.Parameter(decay * torch.ones(1), requires_grad=False)

        self.linear_1 = nn.Linear(image_size**2, channel)
        self.linear_2 = nn.Linear(channel, channel)
        self.linear_3 = nn.Linear(channel, num_classes)

        self.init_mem()
```

Among them, the synaptic connection part mainly consists of 3 linear layers, with the hidden layer dimension set as the channel. The threshold and membrane potential decay coefficient are registered as parameters of the network, which can be configured to be updated by gradients during training. "on_apu" indicates whether the network is running on GPU/CPU or on the KA200 chip's APU. During training, CPU/GPU can be used, while during deployment and inference, CPU/GPU/APU can be selected. "Spike_Act" refers to the specific implementation of the spiking activation function used in the network, which can use the proxy gradient function defined earlier during training. When using APU for inference, only the forward process of "Spike_Act" is needed, so it can be replaced with torch.gt(). Additionally, since the current compilation tool for APU does not support the apply() operation in PyTorch, this replacement is necessary. After instantiation, the membrane potential and other states of the network need to be initialized. Here is one possible implementation:

```python
def init_mem(self):
        # initialize the membrane potential of each neuron
        # lychip_state_wrapper is a wrapper for the membrane potential
        # if on apu is False, the wrapper can be ignored
        self.states_mem_1 = lynchip_state_wrapper(
            torch.zeros(1, self.channel), self.on_apu)
        self.states_mem_2 = lynchip_state_wrapper(
            torch.zeros(1, self.channel), self.on_apu)
        self.states_mem_3 = lynchip_state_wrapper(
            torch.zeros(1, self.num_classes), self.on_apu)
```
The above code initializes the membrane potential to all zeros. lynchip_state_wrapper is a wrapper used for deployment on the APU. During training, when not running on the APU, lynchip_state_wrapper can be omitted or kept with on_apu set to False. The functionality and specific implementation of lynchip_state_wrapper will be discussed in detail in the model deployment section below.

After initialization, the next step is to define how the membrane potential of a single-layer neuron updates at each time step. Based on the discrete iteration form of the LIF (Leaky Integrate-and-Fire) neuron, the update process for a single-layer spiking neuron can be implemented as follows:

```python
def mem_update(self, operator, inputs, mem, vth, decay):
        # update the membrane potential of each neuron
        # load the membrane potential
        mem_ = mem.get().to(inputs.device)
        last_spike = self.Spike_Act(mem_, vth).float()
        # the membrane potential is updated by the following equation
        state = operator(inputs)
        mem_ = mem_ * (1 - last_spike) * decay + state
        spike_out = self.Spike_Act(mem_, vth).float()
        # save the membrane potential
        mem.set(mem_)
        return spike_out
```

Among them, the "operator" is a type of linear operation, such as nn.linear(), nn.Conv2d(), etc. "inputs" are the spike inputs from the upper layer to the current layer, "spike" is the spike output of the current layer, "mem" is the membrane potential state of the current layer, and "vth" and "decay" are respectively the threshold and decay coefficient of the neurons in the current layer. Due to the state memory and reset characteristics of spiking neurons, the output "spike" and "mem" of the current layer will be fed back into the membrane potential update of the neurons in the current layer.

It is worth noting that in order to be consistent with the implementation when deployed on APU, the membrane potential undergoes "get" and "set" operations, respectively, before and after the update. These operations are responsible for reading the membrane potential from DDR memory and writing it back to DDR memory. These two operations can be omitted during training on CPU/GPU.

To achieve arbitrary iterations in the network, when compiling and deploying on APU, only a single feedforward process is compiled. The specific implementation is as follows:

```python
 def forward(self, x):
        # one step forward

        spike_out_1 = self.mem_update(
            self.linear_1, x, self.states_mem_1, self.vth, self.decay)

        spike_out_2 = self.mem_update(
            self.linear_2, spike_out_1, self.states_mem_2, self.vth, self.decay)

        spike_out_3 = self.mem_update(
            self.linear_3, spike_out_2, self.states_mem_3, self.vth, self.decay)

        return spike_out_3
```

Input x is the pulse input at the current time step. After being processed by three layers of pulse neurons, the output spike_out_3 for that time step is obtained.

Based on a single feedforward network, the following is a way to encapsulate the overall multi-time-step iterative pulse neural network:

```python
def spike_net_forward(net, inputs, time_steps):
    # forward the network for multiple steps
    spike_out_sum = 0
    net.init_mem()
    for step in range(time_steps):        
        spike_out = net(inputs)
        # accumulate the spike outputs
        spike_out_sum += spike_out
    return spike_out_sum / time_steps
```

The main computation of this function is an instantiated single-shot feed-forward spiking neural network model. During computation, the number of loop iterations is controlled by specifying time_steps.

### Complete Code

During the training phase, the complete code for the model definition part (snn_model.py) is as follows:

```python
import torch
import torch.nn as nn

class Spike_Act_with_Surrogate_Gradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, vth):
        ctx.save_for_backward(input)
        ctx.vth = vth
        return input.gt(vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        lens = 0.5
        input, = ctx.saved_tensors
        vth = ctx.vth
        grad_input = grad_output.clone()
        # the derivative of the surrogate function is a window function with a
        # width of 2*lens
        temp = abs(input - vth) < lens
        return grad_input * temp.float() / (lens * 2), None


class Spiking_Network_1step(nn.Module):
    def __init__(self, num_classes, image_size, channel,
                 vth, decay, activation, on_apu):
        super(Spiking_Network_1step, self).__init__()
        # Spike_Act is the activation function
        self.Spike_Act = activation
        self.on_apu = on_apu
        self.channel = channel
        self.num_classes = num_classes

        # vth and decay are the parameters of the LIF neuron
        self.vth = nn.Parameter(vth * torch.ones(1), requires_grad=False)
        self.decay = nn.Parameter(decay * torch.ones(1), requires_grad=False)

        self.linear_1 = nn.Linear(image_size**2, channel)
        self.linear_2 = nn.Linear(channel, channel)
        self.linear_3 = nn.Linear(channel, num_classes)

        self.init_mem()

    def init_mem(self):
        # initialize the membrane potential of each neuron
        # lychip_state_wrapper is a wrapper for the membrane potential
        # if on apu is False, the wrapper can be ignored
        self.states_mem_1 = lynchip_state_wrapper(
            torch.zeros(1, self.channel), self.on_apu)
        self.states_mem_2 = lynchip_state_wrapper(
            torch.zeros(1, self.channel), self.on_apu)
        self.states_mem_3 = lynchip_state_wrapper(
            torch.zeros(1, self.num_classes), self.on_apu)

    def mem_update(self, operator, inputs, mem, vth, decay):
        # update the membrane potential of each neuron
        # load the membrane potential
        mem_ = mem.get().to(inputs.device)
        last_spike = self.Spike_Act(mem_, vth).float()
        # the membrane potential is updated by the following equation
        state = operator(inputs)
        mem_ = mem_ * (1 - last_spike) * decay + state
        spike_out = self.Spike_Act(mem_, vth).float()
        # save the membrane potential
        mem.set(mem_)
        return spike_out

    def forward(self, x):
        # one step forward
        if self.on_apu:
            rr = rand(x[0,0,0,0], x.size(), 0x4000, mode=0)
            x = (rr < x).float()
        else:
            x = torch.bernoulli(x)        
        x = x.view(x.size(0), -1)
        
        spike_out_1 = self.mem_update(
            self.linear_1, x, self.states_mem_1, self.vth, self.decay)

        spike_out_2 = self.mem_update(
            self.linear_2, spike_out_1, self.states_mem_2, self.vth, self.decay)

        spike_out_3 = self.mem_update(
            self.linear_3, spike_out_2, self.states_mem_3, self.vth, self.decay)

        return spike_out_3
```
Before the external real-valued excitatory input is fed into the network, Bernoulli sampling is performed to achieve a rate encoding, resulting in binary excitations. This encoding is implemented on the APU using rand operators. During decoding, the average of the pulse outputs of the last layer neurons is taken as the final predicted output, which represents a rate-based decoding scheme.
It should be noted that during training, the lynchip_state_wrapper can be omitted or the specific implementation provided below can be included in snn_model.py.

## Model Training

The SNN model training in this paper is based on the backpropagation algorithm and utilizes PyTorch's automatic differentiation mechanism. The following sections provide implementation details for setting hyperparameters, loading the dataset, learning rate scheduling, optimizer configuration, gradient descent training, and accuracy testing.

### Hyperparameter Settings

The training hyperparameters used in this paper are as follows:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', default='exp/test', type=str)
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--image_size', default=28, type=int)
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--time_steps', default=20, type=int)
parser.add_argument('--channel', default=512, type=int)
parser.add_argument('--vth', default=0.4, type=float)
parser.add_argument('--decay', default=0.8, type=float)
parser.add_argument('--lr', default=0.02, type=float)
parser.add_argument('--seed', default=20231026, type=int)
args = parser.parse_args()
```

Among them, the number of iterations is set to 20, the threshold is 0.4, the membrane potential decay coefficient is 0.8, the batch size is 200, the initial learning rate is 0.02, the number of hidden layer neurons is 512, and the random seed is set to 20231026. These parameter settings can be adjusted according to your needs.

### Dataset Loading
This article focuses on the classification problem of the MNIST dataset. The MNIST dataset is a commonly used dataset for handwritten digit recognition, consisting of 60,000 training samples and 10,000 testing samples, with each sample being a grayscale image of size $28\times28$. The dataset can be loaded using the torchvision library, as follows:

```python
transform_train = transforms.Compose([
    transforms.Resize(size=args.image_size),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.Resize(size=args.image_size),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
```

In the pre-processing step, only resizing and converting the images to PyTorch tensors were performed. No other data augmentation techniques were used.

### Instantiating the Model, Loss Function, and Optimizer

Before training, we need to instantiate the defined SNN network, the loss function, and the optimizer. The implementation details are as follows:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Spiking_Network_1step(num_classes=args.num_classes, image_size=args.image_size,
                            channel=args.channel, vth=args.vth, decay=args.decay, activation=Spike_Act_with_Surrogate_Gradient.apply, on_apu=False).to(device)
criterion_cross_entropy = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-3)
```

Among them, the cross-entropy loss function is chosen, which has good gradient properties when training a classification task. The optimizer used is basic stochastic gradient descent with momentum, and the momentum coefficient is set to 0.9. The weight decay coefficient is set to 1e-3.

### Learning rate scheduling

Learning rate adjustment is done using a step-wise decay approach, implemented as follows:

```python
def get_lr_list():
    # schedule the learning rate
    base_lr = args.lr
    lr_list = []
    for i in range(10):
        lr_list.append(base_lr)
    for i in range(10):
        lr_list.append(base_lr * 0.1)
    for i in range(10):
        lr_list.append(base_lr * 0.01)
    return lr_list

lr_list = get_lr_list()
```

For every 10 epochs of training, the learning rate decays to 0.1 times the original.

### Gradient Descent Training

Based on the definition provided above, here is the training process for one epoch:

```python
def train(epoch):
    net.train()
    lr = lr_list[epoch]
    # update the learning rate of the optimizer
    for p in optimizer.param_groups:
        p['lr'] = lr
    print('\nEpoch: %d,lr: %.5f ' % (epoch, lr), args.root_path)

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        pred = spike_net_forward(net, inputs, args.time_steps)
        loss = criterion_cross_entropy(pred, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        indicator = int(len(trainloader) / 2)
        if ((batch_idx + 1) % indicator == 0):
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
```

In each iteration of training, first the gradient of the model is cleared, then the model does feed-forward operation on the batch of samples, after obtaining the prediction result, the loss function value is calculated and the gradient back propagation of the error is done, and finally the model parameters are updated according to the settings of the optimizer.In addition, the training loss function value and the accuracy are done in the training to facilitate debugging.

### Accuracy Test

After each training round, the model can be tested for its accuracy on a test set, which can be achieved as follows:

```python
def test(epoch):
    net.eval()
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        pred = spike_net_forward(net, inputs, args.time_steps)
        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('test_acc:', 100. * correct / total)
    torch.save(net.state_dict(), args.root_path +
               '/latest.pkl')
```

The above process is similar to training, but without the gradient backpropagation and parameter updates. after testing, save the model parameters at this point, so that they can be used for subsequent deployments and so on.

### Full Code

The complete training code (train_bp.py) consisting of the above components is as follows:

```python
# Copyright (c) OpenBII
# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
from snn_model import Spiking_Network_1step, Spike_Act_with_Surrogate_Gradient
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch
import numpy as np
import torch.optim as optim
import argparse


def setup_seed(seed):
    # set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', default='exp/test', type=str)
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--image_size', default=28, type=int)
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--time_steps', default=20, type=int)
parser.add_argument('--channel', default=512, type=int)
parser.add_argument('--vth', default=0.4, type=float)
parser.add_argument('--decay', default=0.8, type=float)
parser.add_argument('--lr', default=0.02, type=float)
parser.add_argument('--seed', default=20231026, type=int)
args = parser.parse_args()

setup_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
if(os.path.exists(args.root_path) == False):
    os.makedirs(args.root_path)
print('new: ', args.root_path)
print('root path: ', args.root_path)


transform_train = transforms.Compose([
    transforms.Resize(size=args.image_size),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.Resize(size=args.image_size),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Spiking_Network_1step(num_classes=args.num_classes, image_size=args.image_size,
                            channel=args.channel, vth=args.vth, decay=args.decay, activation=Spike_Act_with_Surrogate_Gradient.apply, on_apu=False).to(device)
criterion_cross_entropy = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-3)


def get_lr_list():
    # schedule the learning rate
    base_lr = args.lr
    lr_list = []
    for i in range(10):
        lr_list.append(base_lr)
    for i in range(10):
        lr_list.append(base_lr * 0.1)
    for i in range(10):
        lr_list.append(base_lr * 0.01)
    return lr_list


lr_list = get_lr_list()


def spike_net_forward(net, inputs, time_steps):
    # forward the network for multiple steps
    spike_out_sum = 0
    net.init_mem()
    for step in range(time_steps):        
        spike_out = net(inputs)
        # accumulate the spike outputs
        spike_out_sum += spike_out
    return spike_out_sum / time_steps


def train(epoch):
    net.train()
    lr = lr_list[epoch]
    # update the learning rate of the optimizer
    for p in optimizer.param_groups:
        p['lr'] = lr
    print('\nEpoch: %d,lr: %.5f ' % (epoch, lr), args.root_path)

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        pred = spike_net_forward(net, inputs, args.time_steps)
        loss = criterion_cross_entropy(pred, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        indicator = int(len(trainloader) / 2)
        if ((batch_idx + 1) % indicator == 0):
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    net.eval()
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        pred = spike_net_forward(net, inputs, args.time_steps)
        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('test_acc:', 100. * correct / total)
    torch.save(net.state_dict(), args.root_path +
               '/latest.pkl')


def main():
    for i in range(30):
        train(epoch=i)
        test(epoch=i)


if __name__ == '__main__':
    main()
```

### Training process and results

The next step is training and testing, which can be done on a machine with a GPU. after installing dependencies such as pytorch, run the following command:

```bash
python train_bp.py
```

This will be trained using the default parameters and the expected output during training is as follows:

```bash
new:  exp/test
root path:  exp/test

Epoch: 0,lr: 0.02000  exp/test
149 300 Loss: 1.666 | Acc: 81.983% (24595/30000)
299 300 Loss: 1.611 | Acc: 86.643% (51986/60000)
test_acc: 92.63
...

Epoch: 9,lr: 0.02000  exp/test
149 300 Loss: 1.495 | Acc: 96.900% (29070/30000)
299 300 Loss: 1.494 | Acc: 96.995% (58197/60000)
test_acc: 96.83

Epoch: 10,lr: 0.00200  exp/test
149 300 Loss: 1.493 | Acc: 97.163% (29149/30000)
299 300 Loss: 1.492 | Acc: 97.270% (58362/60000)
test_acc: 96.94

Epoch: 11,lr: 0.00200  exp/test
149 300 Loss: 1.491 | Acc: 97.527% (29258/30000)
299 300 Loss: 1.492 | Acc: 97.377% (58426/60000)
test_acc: 96.97

Epoch: 12,lr: 0.00200  exp/test
149 300 Loss: 1.491 | Acc: 97.380% (29214/30000)
299 300 Loss: 1.492 | Acc: 97.340% (58404/60000)
test_acc: 97.02
...

Epoch: 29,lr: 0.00020  exp/test
149 300 Loss: 1.490 | Acc: 97.547% (29264/30000)
299 300 Loss: 1.490 | Acc: 97.503% (58502/60000)
test_acc: 97.08

```

After 30 epochs of training, the accuracy is about 97%. this can be further improved by adjusting the coding and decoding strategies, introducing data enhancement, drop out, batch norm, etc.

## Model deployment

### Model modifications applicable to compilation

When the KA200 chip is deployed, only the computational graph for the single time-step feed-forward of the pulsed neural network is compiled, and the iteration of multiple time-steps needs to be done multiple times by using resource management tools such as LynSDK to do the round-robin scheduling.In addition to this, in the multi-time-step inference, each membrane potential update needs to utilize the membrane potential state of the previous time-step, which needs to be inserted into the single feed-forward's computational graph by inserting the DDR load and save operations in order to achieve the function to read and write the membrane potential state from the DDR memory before and after the membrane potential update.The above load/save functions are accomplished by using the lynchip_state_wrapper as mentioned above, and its specific implementation is as follows:

```python
class lynchip_state_wrapper():
    def __init__(self, mem, on_apu):
        # mem is a tensor
        self.mem = mem
        # id is a unique identifier for each tensor to be saved
        self.id = uuid1()
        self.on_apu = on_apu

    @staticmethod
    def lynload(tensor, uuid):
        # custom operation to load tensor from ddr to apu
        from torch import ops
        from custom_op_in_lyn.custom_op_load_save import load
        # load custom operation library
        ops.load_library('./custom_op_in_pytorch/build/libcustom_ops.so')
        # register custom operation to pytorch
        from wrap_load_save import load
        return load(tensor, f'{uuid}')

    @staticmethod
    def lynsave(tensor, uuid):
        # custom operation to save tensor from apu to ddr
        from torch import ops
        from custom_op_in_lyn.custom_op_load_save import save
        # load custom operation library
        ops.load_library('./custom_op_in_pytorch/build/libcustom_ops.so')
        # register custom operation to pytorch
        from wrap_load_save import save
        save(tensor, f'{uuid}')

    def get(self):
        # if on_apu is False, return the tensor directly
        if (self.on_apu == False):
            return self.mem
        else:
            # if on_apu is True, load the tensor from ddr to apu
            return self.lynload(self.mem.clone(), self.id)

    def set(self, mem):
        # if on_apu is False, save the tensor directly
        if (self.on_apu == False):
            self.mem = mem
        else:
            # if on_apu is True, save the tensor from apu to ddr
            self.lynsave(mem.clone(), self.id)
```
where on_apu indicates whether it is running on APU or not, id marks the variable name, and uuid can be used to name it implicitly. mem is the encapsulated membrane potential state tensor. when running on APU, the function of loading and storing data from DDR memory is realized by the custom pytorch arithmetic, and the related implementations are in custom_op_load_save.py and wrap_load_save.py, which are described in detail in the sample code package.

Among other things, the custom load/save operations need to be compiled under a certain lyngor compilation environment, which requires running the following script after installing the appropriate lyngor version:
```bash
bash ./build_run_op.sh
```
This completes the compilation process for the custom operator, as detailed in the sample code package.

### Model compilation and loading

The compilation deployment and testing of the impulse neural network is carried out next. the basic parameter settings are given first:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='exp/test/latest.pkl', type=str)
parser.add_argument('--compile', action='store_true')
parser.add_argument('--device_id', default=0, type=int)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--image_size', default=28, type=int)
parser.add_argument('--time_steps', default=20, type=int)
parser.add_argument('--channel', default=512, type=int)
parser.add_argument('--vth', default=0.4, type=float)
parser.add_argument('--decay', default=0.8, type=float)
parser.add_argument('--v', default=0, type=int, help='compile version flag: 1 or 0, means compile v1 or v0')
parser.add_argument('--b', default=1, type=int,help='test batch size')
args = parser.parse_args()
```
Most of the parameters are consistent with the training script. batch_size is set to 1 for the purpose of applying inference scenarios. device_id indicates the number of the specific KA200 chip on which the model is running. the dataset is loaded in the same way as in the training script, and will not be repeated here.

Next, the model is compiled and loaded, as described in the following code:

```python
if args.compile == True:
	net = Spiking_Network_1step(
		args.num_classes,
		args.image_size,
		args.channel,
		args.vth,
		args.decay,
		torch.gt,
		True)
	# load the checkpoint
	net.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
	from utils import lyngor_compile
	# compile the model with lyngor and save the compiled model to out_path
	lyngor_compile(net, model_type="Pytorch", inputs_dict={'data_0': [(1, 1, args.image_size, args.image_size), 'float32']},
				   trace_graph=True, out_path="./mapping_output", save_graph=False, profiler=False, version=args.v, batch_size=args.b)
# load the compiled model
from utils import LynSdkModel_Snn, cons_conn
apu_model = LynSdkModel_Snn(
	'./mapping_output/Net_0',
	time_steps=args.time_steps,
	apu_device=args.device_id)
```

In the above code, the PyTorch model is first instantiated and the trained model parameters are loaded. Then, the lyngor_compile function is called to compile the model, and the compiled artifacts are stored in "./mapping_output". After the compilation is complete, the model is loaded using LynSdkModel_Snn, which implements multiple iterations of scheduling based on LynSDK. It is worth noting that the saved membrane potential state can be configured to be enabled or disabled during inference on different samples, with resetting as the default behavior. cons_conn is the pipe used to send the results after APU inference. The APU inference can be configured as either synchronous or asynchronous callback mode, and in this example, the callback mode is used for lower latency. Meanwhile, the results processed by the APU are read back through a pipe using multiple threads.

### Model inference and comparison of results

In order to improve parallelism and reduce latency, two threads are opened to complete the input data feeding into the APU and the output results reading back to the CPU respectively, which are implemented as follows:

```python
    def spike_net_forward_apu(net, inputs, time_steps, i_idx):
        spike_input_train = inputs.repeat(time_steps, 1, 1, 1, 1)
        net(spike_input_train.numpy(), i_idx)

    correct = 0
    total = 0
    gt_labels = []
    
    def arrange_callback():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            spike_net_forward_apu(
                    apu_model.apu_forward, inputs, args.time_steps, batch_idx)
            for i in range(len(targets)):
                gt_labels.append(targets[i].item())

    def read_result_and_compare_with_gt():
        print("start evaluation")
        global correct, total, gt_labels, stop_flag
        while total < args.b * len(testloader):
            # read the result from apu through pipe
            pred = cons_conn.recv()
            pred = np.mean(pred, axis=(0, 1))
            pred = np.argmax(pred, axis=1)
            j = total  
            for i in range(args.b):
                if (pred[i] == gt_labels[j+i]):
                    correct += 1
                total += 1
            if total + args.b > 10000:
                break
    
    time_start = time.time()
    threading.Thread(target=arrange_callback).start()
    read_p = threading.Thread(target=read_result_and_compare_with_gt)
    read_p.start()
    read_p.join()
    time_end = time.time()
    time.sleep(1)
    print('time cost', time_end - time_start, 's')
    print('fps', args.b * len(testloader) / (time_end - time_start))
    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))
```

Among them, the range_callback() thread feeds the input data into the APU inference one by one, and the read_result_and_compare_with_gt() thread reads back the result via pipe and compares it with really labels to count the accuracy. spike_net_forward_apu() is an input coding implementation for APUs with multiple time-step iterations, which is in line with the computation implemented on CPUs/GPUs.

### Full Code

The above code forms the body of test_on_gpu.py, the complete code is below:

```python
from snn_model import Spiking_Network_1step
import time
import torch
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np

import threading
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='exp/test/latest.pkl', type=str)
parser.add_argument('--compile', action='store_true')
parser.add_argument('--device_id', default=0, type=int)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--image_size', default=28, type=int)
parser.add_argument('--time_steps', default=20, type=int)
parser.add_argument('--channel', default=512, type=int)
parser.add_argument('--vth', default=0.4, type=float)
parser.add_argument('--decay', default=0.8, type=float)
parser.add_argument('--v', default=0, type=int, help='compile version flag: 1 or 0, means compile v1 or v0')
parser.add_argument('--b', default=1, type=int,help='test batch size')
args = parser.parse_args()

transform_train = transforms.Compose([
    transforms.Resize(size=args.image_size),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.Resize(size=args.image_size),
    transforms.ToTensor(),
])

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.b, shuffle=False, num_workers=0)



if __name__ == '__main__':

    if args.compile == True:
        net = Spiking_Network_1step(
            args.num_classes,
            args.image_size,
            args.channel,
            args.vth,
            args.decay,
            torch.gt,
            True)
        # load the checkpoint
        net.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        from utils import lyngor_compile
        # compile the model with lyngor and save the compiled model to out_path
        lyngor_compile(net, model_type="Pytorch", inputs_dict={'data_0': [(1, 1, args.image_size, args.image_size), 'float32']},
                       trace_graph=True, out_path="./mapping_output", save_graph=False, profiler=False, version=args.v, batch_size=args.b)
    # load the compiled model
    from utils import LynSdkModel_Snn, cons_conn
    apu_model = LynSdkModel_Snn(
        './mapping_output/Net_0',
        time_steps=args.time_steps,
        apu_device=args.device_id)

    def spike_net_forward_apu(net, inputs, time_steps, i_idx):
        spike_input_train = inputs.repeat(time_steps, 1, 1, 1, 1)
        net(spike_input_train.numpy(), i_idx)

    correct = 0
    total = 0
    gt_labels = []
    
    def arrange_callback():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            spike_net_forward_apu(
                    apu_model.apu_forward, inputs, args.time_steps, batch_idx)
            for i in range(len(targets)):
                gt_labels.append(targets[i].item())

    def read_result_and_compare_with_gt():
        print("start evaluation")
        global correct, total, gt_labels, stop_flag
        while total < args.b * len(testloader):
            # read the result from apu through pipe
            pred = cons_conn.recv()
            pred = np.mean(pred, axis=(0, 1))
            pred = np.argmax(pred, axis=1)
            j = total  
            for i in range(args.b):
                if (pred[i] == gt_labels[j+i]):
                    correct += 1
                total += 1
            if total + args.b > 10000:
                break
    
    time_start = time.time()
    threading.Thread(target=arrange_callback).start()
    read_p = threading.Thread(target=read_result_and_compare_with_gt)
    read_p.start()
    read_p.join()
    time_end = time.time()
    time.sleep(1)
    print('time cost', time_end - time_start, 's')
    print('fps', args.b * len(testloader) / (time_end - time_start))
    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))
```

### APU inference result

After completing the above steps, run the following command:

```bash
python test_on_apu.py  --compile --b 28 
```
--compile means recompile the model, --b 28 means set the inference batchsize to 28, the number of supported batchsizes is 1,2,4,7,14,28.
The expected output of the reasoning result is as follows:

```
==================================>
 def @main(%vth: Tensor[(1), float32], %decay: Tensor[(1), float32], %data_0: Tensor[(1, 1, 28, 28), float32], %linear_1.weight: Tensor[(512, 784), float32], %linear_1.bias: Tensor[(512), float32], %linear_2.weight: Tensor[(512, 512), float32], %linear_2.bias: Tensor[(512), float32], %linear_3.weight: Tensor[(10, 512), float32], %linear_3.bias: Tensor[(10), float32]) -> Tensor[(1, 10), float32] {
  %0 = reshape(meta[relay.Constant][0] /* ty=Tensor[(1, 10), float32] */ /* ty=Tensor[(1, 10), float32] */, newshape=[1, 10]) /* ty=Tensor[(1, 10), float32] */;
  %1 = annotation.load(%0, meta[relay.attrs.LoadSaveAttrs][0]) /* ty=Tensor[(1, 10), float32] */;
  %2 = cast(%1, dtype="float32") /* ty=Tensor[(1, 10), float32] */;
  %3 = cast(1f /* ty=float32 */, dtype="float32") /* ty=float32 */;
  %4 = cast(1f /* ty=float32 */, dtype="float32") /* ty=float32 */;
  %5 = greater(%2, %vth) /* ty=Tensor[(1, 10), bool] */;
  %6 = cast(%5, dtype="float32") /* ty=Tensor[(1, 10), float32] */;
  %7 = multiply(%4, %6) /* ty=Tensor[(1, 10), float32] */;
  %8 = subtract(%3, %7) /* ty=Tensor[(1, 10), float32] */;
  %9 = multiply(%2, %8) /* ty=Tensor[(1, 10), float32] */;
  %10 = multiply(%9, %decay) /* ty=Tensor[(1, 10), float32] */;
  %11 = reshape(meta[relay.Constant][1] /* ty=Tensor[(1, 512), float32] */ /* ty=Tensor[(1, 512), float32] */, newshape=[1, 512]) /* ty=Tensor[(1, 512), float32] */;
  %12 = annotation.load(%11, meta[relay.attrs.LoadSaveAttrs][1]) /* ty=Tensor[(1, 512), float32] */;
  %13 = cast(%12, dtype="float32") /* ty=Tensor[(1, 512), float32] */;
  %14 = cast(1f /* ty=float32 */, dtype="float32") /* ty=float32 */;
  %15 = cast(1f /* ty=float32 */, dtype="float32") /* ty=float32 */;
  %16 = greater(%13, %vth) /* ty=Tensor[(1, 512), bool] */;
  %17 = cast(%16, dtype="float32") /* ty=Tensor[(1, 512), float32] */;
  %18 = multiply(%15, %17) /* ty=Tensor[(1, 512), float32] */;
  %19 = subtract(%14, %18) /* ty=Tensor[(1, 512), float32] */;
  %20 = multiply(%13, %19) /* ty=Tensor[(1, 512), float32] */;
  %21 = multiply(%20, %decay) /* ty=Tensor[(1, 512), float32] */;
  %22 = reshape(meta[relay.Constant][2] /* ty=Tensor[(1, 512), float32] */ /* ty=Tensor[(1, 512), float32] */, newshape=[1, 512]) /* ty=Tensor[(1, 512), float32] */;
  %23 = annotation.load(%22, meta[relay.attrs.LoadSaveAttrs][2]) /* ty=Tensor[(1, 512), float32] */;
  %24 = cast(%23, dtype="float32") /* ty=Tensor[(1, 512), float32] */;
  %25 = cast(1f /* ty=float32 */, dtype="float32") /* ty=float32 */;
  %26 = cast(1f /* ty=float32 */, dtype="float32") /* ty=float32 */;
  %27 = greater(%24, %vth) /* ty=Tensor[(1, 512), bool] */;
  %28 = cast(%27, dtype="float32") /* ty=Tensor[(1, 512), float32] */;
  %29 = multiply(%26, %28) /* ty=Tensor[(1, 512), float32] */;
  %30 = subtract(%25, %29) /* ty=Tensor[(1, 512), float32] */;
  %31 = multiply(%24, %30) /* ty=Tensor[(1, 512), float32] */;
  %32 = multiply(%31, %decay) /* ty=Tensor[(1, 512), float32] */;
  %33 = take(%data_0, 0 /* ty=int32 */, axis=0, indices=[0]) /* ty=Tensor[(1, 28, 28), float32] */;
  %34 = take(%33, 0 /* ty=int32 */, axis=0, indices=[0]) /* ty=Tensor[(28, 28), float32] */;
  %35 = take(%34, 0 /* ty=int32 */, axis=0, indices=[0]) /* ty=Tensor[(28), float32] */;
  %36 = take(%35, 0 /* ty=int32 */, axis=0, indices=[0]) /* ty=float32 */;
  %37 = lyn_random_data_1_6712(%36, meta[DictAttrs][0]) /* ty=Tensor[(1, 1, 28, 28), float16] */;
  %38 = cast(%37, dtype="float32") /* ty=Tensor[(1, 1, 28, 28), float32] */;
  %39 = less(%38, %data_0) /* ty=Tensor[(1, 1, 28, 28), bool] */;
  %40 = cast(%39, dtype="float32") /* ty=Tensor[(1, 1, 28, 28), float32] */;
  %41 = reshape(%40, newshape=[1, -1]) /* ty=Tensor[(1, 784), float32] */;
  %42 = nn.dense(%41, %linear_1.weight, units=None) /* ty=Tensor[(1, 512), float32] */;
  %43 = nn.bias_add(%42, %linear_1.bias, axis=-1) /* ty=Tensor[(1, 512), float32] */;
  %44 = multiply(%43, 1f /* ty=float32 */) /* ty=Tensor[(1, 512), float32] */;
  %45 = add(%32, %44) /* ty=Tensor[(1, 512), float32] */;
  %46 = greater(%45, %vth) /* ty=Tensor[(1, 512), bool] */;
  %47 = cast(%46, dtype="float32") /* ty=Tensor[(1, 512), float32] */;
  %48 = nn.dense(%47, %linear_2.weight, units=None) /* ty=Tensor[(1, 512), float32] */;
  %49 = nn.bias_add(%48, %linear_2.bias, axis=-1) /* ty=Tensor[(1, 512), float32] */;
  %50 = multiply(%49, 1f /* ty=float32 */) /* ty=Tensor[(1, 512), float32] */;
  %51 = add(%21, %50) /* ty=Tensor[(1, 512), float32] */;
  %52 = greater(%51, %vth) /* ty=Tensor[(1, 512), bool] */;
  %53 = cast(%52, dtype="float32") /* ty=Tensor[(1, 512), float32] */;
  %54 = nn.dense(%53, %linear_3.weight, units=None) /* ty=Tensor[(1, 10), float32] */;
  %55 = nn.bias_add(%54, %linear_3.bias, axis=-1) /* ty=Tensor[(1, 10), float32] */;
  %56 = multiply(%55, 1f /* ty=float32 */) /* ty=Tensor[(1, 10), float32] */;
  %57 = add(%10, %56) /* ty=Tensor[(1, 10), float32] */;
  %58 = greater(%57, %vth) /* ty=Tensor[(1, 10), bool] */;
  %59 = cast(%58, dtype="float32") /* ty=Tensor[(1, 10), float32] */;
  %60 = reshape(%45, newshape=[1, 512]) /* ty=Tensor[(1, 512), float32] */;
  %61 = annotation.save(%60, meta[relay.attrs.LoadSaveAttrs][3]) /* ty=Tensor[(1, 512), float32] */;
  %62 = reshape(%51, newshape=[1, 512]) /* ty=Tensor[(1, 512), float32] */;
  %63 = annotation.save(%62, meta[relay.attrs.LoadSaveAttrs][4]) /* ty=Tensor[(1, 512), float32] */;
  %64 = reshape(%57, newshape=[1, 10]) /* ty=Tensor[(1, 10), float32] */;
  %65 = annotation.save(%64, meta[relay.attrs.LoadSaveAttrs][5]) /* ty=Tensor[(1, 10), float32] */;
  lyn_vir_op_4_4000(%59, %61, %63, %65) /* ty=Tensor[(1, 10), float32] */
}

 
<==================================
opt_level= 3, save_graph = False, profiler = False
lyn.op_graph_batch_limit =  1
version =  0
util.py[line:268]-INFO: [optimize] Total time running optimize(2): 2.1731 seconds
util.py[line:268]-INFO: [apu_build+optimize] Total time running apu_build(1): 2.4052 seconds
util.py[line:268]-INFO: [abc_map] Total time running abc_map(175): 0.2746 seconds
util.py[line:268]-INFO: [build+map] Total time running build(0): 5.3839 seconds
chip count =  3
######## model informations ########
batchsize: 28
inputnum: 1
inputshape: [[1, 1, 28, 28]]
inputdatalen: 3136
inputdatatype: lyn_data_type_t.DT_FLOAT
outputnum: 1
outputshape: [[1, 10]]
outputdatalen: 40
outputdatatype: lyn_data_type_t.DT_FLOAT
####################################
start evaluation
time cost 5.104900121688843 s
fps 1963.6035497367934
Accuracy of the network on the 10000 test images: 97 %
################
##### PASS #####
################
```

As can be seen, the accuracy of inference on the APU is 97%, which is essentially the same as the results on the GPU. inference frame rate is 1963.60.

At this point, we have completed the whole process of building an impulse neural network model from scratch based on the pytorch framework and training it on the CPU/GPU, and later deploying the inference on the APU of the KA200 chip.