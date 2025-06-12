# Class to load the dataset.
import torch
import cv2
import numpy as np
    
class ParticleDataset():
    '''
    Class: To read the images and pass it into the dataloader for pytorch.
    Since the images are sequences of moving particles, we shuffle the frames 
    to prevent the NN from learning the physics of the trajectories and purely focus
    the features and overlap within each image.
    
    ** NOTE: The shuffle is done on the index to maintain the pair of image and its 
             corresponding Heatmap.    
    
    Params: 
        Dataset: takes 1. the image directory path 2. the heatmap directory  
        
    Returns: 
        img: (shuffled sequence)
        heatmap: (the corresponding heatmap of the image)
    
    '''
    
    def __init__(self, img_paths_, heatmap_paths_):
        self.img_paths = img_paths_
        self.heatmap_paths = heatmap_paths_

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        
        # Load image and pass to torch tensor
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE) / 255.0 # normalise it 
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        # Load heatmap and pass to torch tensor
        heatmap = np.load(self.heatmap_paths[idx], allow_pickle=True)
        heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        return img, heatmap