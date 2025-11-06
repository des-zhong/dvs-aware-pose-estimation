import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
np.random.seed(1)
def salt_and_pepper_noise(tensor, noise_level=0.1):
    """
    Adds the same salt and pepper noise across all frames in the tensor.
    
    Parameters:
    tensor (torch.Tensor): The input tensor of shape (batch_size, 20, 200, 200).
    noise_level (float): Percentage of pixels to be changed to salt and pepper noise.
    
    Returns:
    torch.Tensor: The tensor with salt and pepper noise added.
    """
    # Get tensor properties
    _, _, height, width = tensor.shape
    num_pixels = height * width
    num_salt_pepper = int(noise_level * num_pixels*np.random.random())
    
    # Create a mask for salt and pepper noise
    y_coords = np.random.randint(0, height, num_salt_pepper)
    x_coords = np.random.randint(0, width, num_salt_pepper)
    
    # Split indices for salt and pepper
    half = num_salt_pepper // 2
    salt_coords = (y_coords[:half], x_coords[:half])
    pepper_coords = (y_coords[half:], x_coords[half:])
    
    # Clone tensor to avoid modifying the original
    noisy_tensor = torch.zeros_like(tensor)
    
    # Apply the salt and pepper noise across all frames
    noisy_tensor[:, :, salt_coords[0], salt_coords[1]] = 1.0  # Salt noise
    
    return noisy_tensor


class EVSDataset(Dataset):
    def __init__(self, data_dir, label_file):
        """
        Init dataset
        :param data_dir
        :param label_file
        """
        self.data_dir = data_dir
        self.label_file = label_file
        self.labels = torch.tensor(np.loadtxt(label_file, delimiter=",")*np.pi/180, dtype=torch.float32).to('cuda')
        
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('_processed.npy')]

        self.file_list.sort(key=lambda x: int(x.split('_')[0]))
        print(f"Loaded {len(self.file_list)} data files.")
        print(f"Loaded {len(self.labels)} labels.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])

        sample = torch.from_numpy(np.load(file_path)).float()

        label = self.labels[idx].clone().detach()
        
        return sample, label

def create_dataloader(data_dir, label_file, batch_size=32, num_workers=4):
    dataset = EVSDataset(data_dir, label_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

def split_dataset(dataset, train_ratio=0.8):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


class ann_dataset(Dataset):
    def __init__(self, data_dir, label_file):
        self.data_dir = data_dir
        self.label_file = label_file
        self.labels = torch.tensor(np.loadtxt(label_file, delimiter=",")*np.pi/180, dtype=torch.float32)
        
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('_processed.npy')]

        self.file_list.sort(key=lambda x: int(x.split('_')[0]))
        self.data=[]
        for file_name in self.file_list:
            file_path = os.path.join(self.data_dir, file_name)
            self.data.append(torch.from_numpy(np.load(file_path)).unsqueeze(0))
        print(f"Loaded {len(self.file_list)} data files.")
        print(f"Loaded {len(self.labels)} labels.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        sample = self.data[idx]

        label = self.labels[idx].clone().detach()
        
        return sample, label

if __name__ == "__main__":
    data_dir = "/home/apollo/Downloads/hyper_dvs/bidl-v1.7.0/tutorial/build_snn_from_scratch_pytorch/evs_slice"
    label_file = "/home/apollo/Downloads/hyper_dvs/bidl-v1.7.0/tutorial/build_snn_from_scratch_pytorch/label.csv"
    batch_size = 16

    dataloader = create_dataloader(data_dir, label_file, batch_size)

    for data, labels in dataloader:
        print(data.shape, labels.shape)
        break
