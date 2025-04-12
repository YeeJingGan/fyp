import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, npz_dir, is_training=True, mol=1):
        self.files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
        self.is_training = is_training
        self.mol = mol

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # Load from .npz
        data = np.load(file_path)
        image = torch.tensor(data["image"], dtype=torch.float32)

        if self.is_training:
            noisy_image = torch.tensor(data["noisy_image"], dtype=torch.float32)
        else:
            noisy_image = torch.zeros_like(image) 

        # Determine modality 
        if self.mol == 1:
            modality = torch.tensor([1, 0], dtype=torch.float32)  # T1
        else:
            modality = torch.tensor([0, 1], dtype=torch.float32)  # T2

        return image, noisy_image, modality, file_path