import os
import sys
sys.path.append(os.getenv('PROJECT_ROOT'))
import torch
from scipy.ndimage import median_filter
from src.unet3d_dae import UNet3D
from src.data_preprocessor import DataPreprocessor

def load_model(model_path=os.path.join(os.getenv('PROJECT_ROOT'), r'model\model_epoch_10.pth')):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3D().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    return device, model

def preprocess_mri(mri_file, modality):
    img_without_noise, _ = DataPreprocessor().preprocess(mri_path=mri_file, mdl=modality, add_noise=False)

    img_without_noise = img_without_noise.unsqueeze(0)

    # Determine modality 
    if modality == 1:
        modal = torch.tensor([1, 0], dtype=torch.float32)  # T1
    else:
        modal = torch.tensor([0, 1], dtype=torch.float32)  # T2
    
    return img_without_noise, modal

def prediction(mri_file, modality):
    device, model = load_model()

    image, mdl = preprocess_mri(mri_file, modality)
    image, mdl = image.to(device), mdl.to(device)

    with torch.no_grad():
        reconstruction = model(image, mdl)
        residual = torch.abs(image - reconstruction)

        # Apply 3D median filtering
        residual_np = residual.squeeze().cpu().numpy()
        filtered_residual = median_filter(residual_np, size=5)
        filtered_residual = torch.tensor(filtered_residual, dtype=residual.dtype)

    return image, filtered_residual