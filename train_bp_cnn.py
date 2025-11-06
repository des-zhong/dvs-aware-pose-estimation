# Copyright (c) OpenBII
# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
from snn_model_cnn import Spiking_Network_CNN, Spike_Act_with_Surrogate_Gradient
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from loguru import logger

# for dvs
from data_loader_slice import EVSDataset, salt_and_pepper_noise
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

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

save_name = "decay08_01"
parser.add_argument('--root_path', default=f'exp/{save_name}', type=str)
parser.add_argument('--gpu', default=0, type=int)

# for dvs

# parser.add_argument('--image_size', default=28, type=int)
# parser.add_argument('--batch_size', default=200, type=int)
# parser.add_argument('--channel', default=512, type=int)
parser.add_argument('--image_size', default=200, type=int)
parser.add_argument('--batch_size', default=64, type=int)
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
logger.info('new: ', args.root_path)
logger.info('root path: ', args.root_path)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os

data_dir = "./data/combined_data/"
label_file = "./data/combined_data/label.csv"

files = os.listdir(data_dir)
npy_files = [f for f in files if f.endswith('.npy')]
logger.info(f"Total .npy files in directory: {len(npy_files)}")


dataset = EVSDataset(data_dir, label_file)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
logger.info("Total samples in dataset:", len(dataset))
trainset, testset = random_split(dataset, [train_size, test_size])
logger.info("Training set size:", len(trainset))
logger.info("Test set size:", len(testset))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=True)


net = Spiking_Network_CNN(img_size=args.image_size, num_channels=args.channel, vth=args.vth, decay=args.decay, activation=Spike_Act_with_Surrogate_Gradient.apply, on_apu=False, init_batch=args.batch_size).to(device)

# for dvs
# criterion_cross_entropy = nn.CrossEntropyLoss().to(device)
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
                    #   momentum=0.9, weight_decay=1e-3)
# criterion = nn.MSELoss().to(device)
criterion = nn.L1Loss().to(device)
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.lr*0.01)



def spike_net_forward(net, inputs, num_timestamps):
    net.init_mem(args.batch_size)

    membrane_potentials_sum = None

    for t in range(num_timestamps):
        # [batch_size, 1, 200, 200]
        current_input = inputs[:, t, :, :].unsqueeze(1)

        membrane_potentials = net(current_input)

        if membrane_potentials_sum is None:
            membrane_potentials_sum = torch.zeros_like(membrane_potentials)
        

        membrane_potentials_sum += membrane_potentials

    return membrane_potentials_sum / num_timestamps

def train(epoch):
    net.train()

    logger.info('\nEpoch: %d, lr: %.5f ' % (epoch, optimizer.param_groups[0]['lr']))

    train_loss = 0.0
    total_batches = 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs += salt_and_pepper_noise(inputs)
        # data = inputs.to('cpu').numpy()[0,0]
        # plt.imshow(data, cmap='gray',interpolation='nearest')
        # plt.savefig('visualize.png')
        
        pred = spike_net_forward(net, inputs, args.time_steps).squeeze()
        optimizer.zero_grad()
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        total_batches += 1

        if batch_idx % args.batch_size == 0:
            current_loss = loss.item()
            logger.info(f'Train Epoch: {epoch}, Batch: {batch_idx}, Current Loss: {current_loss:.6f}')
    scheduler.step()

    average_loss = train_loss / total_batches
    train_losses.append(average_loss)
    
    logger.info(f'Epoch: {epoch}, Average Training Loss: {average_loss:.6f}')
    return average_loss

def test(epoch):
    global best_epoch
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            
            inputs, targets = inputs.to(device), targets.to(device)
            pred = spike_net_forward(net, inputs, args.time_steps).squeeze()
            test_loss += criterion(pred, targets).item()

    avg_test_loss = test_loss / len(testloader)
    logger.info('\nTest set: Average loss: {:.4f}\n'.format(avg_test_loss))

    if should_save_model(avg_test_loss):
        best_epoch = epoch
        torch.save(net.state_dict(), args.root_path + '/best_model.pkl')
        logger.info("Saved improved model to '{}'.".format(args.root_path + '/best_model.pkl'))
    test_losses.append(avg_test_loss)
    
    return avg_test_loss

def should_save_model(current_test_loss):
    if not test_losses:
        return True  
    elif current_test_loss < min(test_losses):
        return True
    else:
        return False
    
def plot_and_save_losses(train_losses, test_losses, epoch_step=5, file_name=f'loss_plot_{save_name}.png'):
    average_train_losses = [np.mean(train_losses[i:i+epoch_step]) for i in range(0, len(train_losses), epoch_step)]
    average_test_losses = [np.mean(test_losses[i:i+epoch_step]) for i in range(0, len(test_losses), epoch_step)]
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(train_losses), epoch_step), average_train_losses, label='Average Train Loss')
    plt.plot(range(0, len(test_losses), epoch_step), average_test_losses, label='Average Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train and Test Loss Every {epoch_step} Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(file_name)
    plt.close()
    
def save_loss2npy(train_losses, test_losses, filename=str(args.seed)):
    data = np.array([train_losses, test_losses])
    np.save(filename+".npy", data)

def main():
    try:
        for i in range(args.epoch):
            train(epoch=i)
            test(epoch=i)
            if i%100==0 and i>0:
                plot_and_save_losses(train_losses, test_losses)

    finally:
        plot_and_save_losses(train_losses, test_losses)
        save_loss2npy(train_losses, test_losses)
        logger.info(f"final test_loss: {min(test_losses)}")
        logger.info(f"best epoch at: {best_epoch}")
        

if __name__ == '__main__':
    main()
