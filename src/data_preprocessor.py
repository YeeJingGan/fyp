import ants
import antspynet
import torch
import torchio as tio
import torch.nn.functional as F

class DataPreprocessor:
    def __init__(self, target_shape=(128, 128, 128)):
        self.target_shape = target_shape
    
    def preprocess(self, mri_path, mdl=1, add_noise=True):
        # 1. Reorient to RAS
        image = ants.image_read(mri_path)
        reoriented_image = ants.reorient_image2(image, orientation='RAS')

        # Convert to TorchIO ScalarImage
        torchio_image = tio.ScalarImage(tensor=torch.from_numpy(reoriented_image.numpy()).unsqueeze(0))

        if add_noise:
            # 2. Brain extraction
            brain_mask = antspynet.brain_extraction(reoriented_image, modality="t1" if mdl == 1 else "t2", verbose=False)
            torchio_mask = tio.ScalarImage(tensor=torch.from_numpy(brain_mask.numpy()).unsqueeze(0))

            # Create binary mask and extract brain region
            binary_mask = (torchio_mask.data > 0.9).to(torch.uint8)
            extracted_image = torchio_image.data * binary_mask
        else:
            # No brain extraction — use full image and mask as ones
            extracted_image = torchio_image.data
            binary_mask = torch.ones_like(extracted_image, dtype=torch.uint8)

        # 3. Intensity normalization
        normalised_image = tio.RescaleIntensity(percentiles=(0, 99), masking_method=lambda x: x > 0)(extracted_image)

        # 4. Resample to 1x1x1 mm
        resampled_image = tio.Resample((1, 1, 1))(normalised_image)
        resampled_mask = tio.Resample((1, 1, 1))(binary_mask)

        # 5. Resize
        resized_image = tio.Resize(target_shape=self.target_shape, image_interpolation='linear')(resampled_image)
        resized_mask = tio.Resize(target_shape=self.target_shape, image_interpolation='linear')(resampled_mask)

        # 6. Optional noise
        if not add_noise:
            return resized_image, None
        
        shape = (1, 1, 16, 16, 16)
        gaussian_noise = torch.rand(shape) * 0.2
        high_resolution_noise = F.interpolate(gaussian_noise, size=self.target_shape, mode='trilinear')
        augmented_noise = tio.transforms.RandomAffine(scales=0, degrees=0, translation=5)(high_resolution_noise.squeeze(0))

        noisy_image = resized_image.clone()
        noisy_image += augmented_noise
        noisy_image *= resized_mask

        return resized_image, noisy_image
