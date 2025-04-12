import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, npz_dir, mol=1):
        self.files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
        self.mol = mol

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # Load from .npz
        data = np.load(file_path)
        ground_truth = torch.tensor(data["ground_truth"], dtype=torch.float32)
        noisy_image = torch.tensor(data["noisy_image"], dtype=torch.float32)
        mask = torch.tensor(data["mask"], dtype=torch.float32)

        # Determine modality based on directory 
        if self.mol == 1:
            modality = torch.tensor([1, 0], dtype=torch.float32)  # T1
        else:
            modality = torch.tensor([0, 1], dtype=torch.float32)  # T2

        return ground_truth, noisy_image, mask, modality