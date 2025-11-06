# Copyright (c) OpenBII
# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
from snn_model_cnn import Spiking_Network_CNN, Spike_Act_with_Surrogate_Gradient
import os
import torch.nn as nn
from tqdm import tqdm
import torch
import numpy as np
import torch.optim as optim
import argparse
from loguru import logger

train_losses = []
test_losses = []

best_epoch = 0

def setup_seed(seed):
    # set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', default='exp/enhance1', type=str)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--num_classes', default=10, type=int)

# for dvs

# parser.add_argument('--image_size', default=28, type=int)
# parser.add_argument('--batch_size', default=200, type=int)
# parser.add_argument('--channel', default=512, type=int)
parser.add_argument('--image_size', default=200, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--time_steps', default=20, type=int)
parser.add_argument('--channel', default=100, type=int)
parser.add_argument('--epoch', default=500, type=int)

parser.add_argument('--vth', default=0.4, type=float)
parser.add_argument('--decay', default=0.1, type=float)

# for dvs
# parser.add_argument('--lr', default=0.02, type=float)
parser.add_argument('--lr', default=0.001*0.1, type=float)

parser.add_argument('--seed', default=20241029, type=int)
args = parser.parse_args()

setup_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
if(os.path.exists(args.root_path) == False):
    os.makedirs(args.root_path)

import os

device = torch.device('gpu')

net = Spiking_Network_CNN(img_size=args.image_size,
                            num_channels=args.channel, vth=args.vth, decay=args.decay, activation=Spike_Act_with_Surrogate_Gradient.apply, on_apu=False, init_batch=args.batch_size).to(device)
net.load_state_dict(torch.load('./exp/decay01/best_model.pkl', weights_only=True))
# for dvs
# criterion_cross_entropy = nn.CrossEntropyLoss().to(device)
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
                    #   momentum=0.9, weight_decay=1e-3)
# criterion = nn.MSELoss().to(device)
criterion = nn.L1Loss().to(device)


def spike_net_forward(net, inputs, num_timestamps):
    net.init_mem(args.batch_size)

    membrane_potentials_sum = None

    for t in range(num_timestamps):
        # 选择当前时间戳的数据并增加一个维度，形状为 [batch_size, 1, 200, 200]
        current_input = inputs[:, t, :, :].unsqueeze(1)

        membrane_potentials = net(current_input)

        if membrane_potentials_sum is None:
            membrane_potentials_sum = torch.zeros_like(membrane_potentials)
        

        membrane_potentials_sum += membrane_potentials

    return membrane_potentials_sum / num_timestamps


def test():
    global best_epoch
    net.eval()
    test_loss = 0
    labels = np.loadtxt('./data/combined_data/label.csv', delimiter=",")
    loss = 0
    N=100
    with torch.no_grad():
        for i in range(11):
            inputs = np.float32(np.load('./data/combined_data/'+str(i)+'_processed.npy'))
            inputs = torch.from_numpy(inputs).unsqueeze(0).to(device)
            pred = spike_net_forward(net, inputs, args.time_steps).squeeze()
            
            print(labels[i])
            print(pred.numpy()*180/3.14,'\n')
            loss+=criterion(pred, torch.from_numpy(labels[i]))
            # print(pred*180/3.14)
    print(loss/N)
            



def read_weights():
    with open("cnn_weights.txt", "w") as f:
        for name, param in net.named_parameters():
            f.write(f"Layer: {name}\n")
            f.write(f"Weights:\n{param.data.numpy()}\n")
            print(np.max(np.abs(param.data.numpy())),np.min(np.abs(param.data.numpy())))
            f.write("\n")



def main():
    test()
    # read_weights()

        

if __name__ == '__main__':
    main()
