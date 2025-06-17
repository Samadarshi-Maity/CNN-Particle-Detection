# this file contains the all the   

# load the necessary python packages 
import numpy as np
import pandas as pd 
import cv2
from scipy.ndimage import maximum_filter, label, center_of_mass
import torch

# function to convert coordinates to heatmaps 
def coor2heatmap(frame_coord_path):
        '''
        Function to extract the XY coordinates and add them to  
        the 
        Params: 
            path to the tsv file 
        Returns: 
            
        '''
        data_check = pd.read_csv(frame_coord_path, sep = '\t')
        # duplicates can cause problems with the output
        data_check.sort_values(by = ['X'])
        vall1 = data_check.groupby('X')['Y'].count()
        vall2 = data_check.groupby('X')['Y'].nunique()
        dupli = np.sum(vall1.values- vall2.values)

        # convert the spatial_distribution from the X, Y coordinates 
        H, xedges, yedges = np.histogram2d(Yy, Xx, bins=(xedges, yedges))
        return H 

# function to convert heatmap to coordinates
def image2coord(model, img_path, device, threshold = 20, scan_radius = 2):
    '''
    Takes the image and predicts the heatmap. Then extracts the postions from the heat map coordinates.
    Then finds the local maxima based on the threshold and the scan_radius
    
    Params: 
        model       : CNN model 
        img_path  : path to the image location 
        device      : processor: cpu, gpu
        threshold   : cutoff based on the bacnkground noise
        scan_radius : plateau size for the heatmap pulse
        
    Returns: 
        x: array of x coordinates
        y: array of y coordinates 
        
    '''
    model.to(device)
    model.eval()
    
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  
    
    # no need for the grad
    with torch.no_grad():
        HtMap_pred = model(img.unsqueeze(0).to(device))
    
    # converts to standard grey map and pushes the heatmap to the CPU 
    heatmap = HtMap_pred.squeeze().cpu().numpy()*255

    # Apply maximum filter
    max_filt = maximum_filter(heatmap, size=scan_radius, mode='constant')

    # Find pixels equal to local max
    local_max = (heatmap == max_filt)

    # Threshold for noise tolerance (same as max threshold in imageJ)
    detected_peaks = local_max & (heatmap > threshold)

    # Label connected components (plateaus)
    labeled, num_features = label(detected_peaks)

    # Get centroids of plateaus (peak positions)
    peak_coords = center_of_mass(heatmap, labeled, range(1, num_features+1))
    y, x =  np.array(peak_coords).transpose()
    
    # two vectors containing the x and the y coordinates
    return x, y, heatmap
    