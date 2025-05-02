**Title: Abnormalities Detection in Brain MRI Using Unsupervised Approach**

This is a Final Year Project carried out by Gan Yee Jing (student ID: 22WMR15427) who persuaing Bachelor's Degree in Data Science (Honours) in Tunku Abdul Rahman University of Management and Technology (TAR UMT). 
Project title is as mentioned above. 
The concept is training a model who excel in reconstructing a healthy state of any 3D brain MRI. 
When a 3D MRI with unknown status is given, the model will try to reconstruct its healthy state, whether or not the brain is healthy or unhealthy.
If the brain is unhealthy, the model will struggle to construct those unhealthy region, as it not leanred not reconstruct unhealthy region, leading to a high recontruction error.
Those region with high reconstruction error can be viwed as the anomaly.
Then abnormalities in the brain can be localised, by looking at the reconstruction anomaly map.

Architecture Used: Denoising Autoencoder (reference paper: https://proceedings.mlr.press/v172/kascenas22a.html)
Architecture Modification: Feature-wise Modulation Layer (reference paper: https://doi.org/10.48550/arXiv.1709.07871)
Final Architecture:

Training Dataset Used: IXI Dataset (reference: https://brain-development.org/ixi-dataset/)
Testing Dataset Used:
1. BraTS2020 Dataset (reference: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
2. MSLUB Dataset (reference: https://lit.fe.uni-lj.si/en/research/resources/3D-MR-MS/)
