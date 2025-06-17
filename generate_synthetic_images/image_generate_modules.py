
# import some necessary python modules 
import numpy as np
import pandas as pd
import random 
import os
import glob 
import tqdm
import matplotlib.pyplot as plt
plt.style.use('classic')
from scipy.ndimage import gaussian_filter
import math
import scipy



# create a for loop for geerating the images
def gen_path(folder, N_images):
    '''
    This function is used to create path list for the images
    Params:
        folder    : path to the folder to save the data
        N_images  : number of images
    Returns: 
        path_list : a list containing paths(names) for each image.
    '''
    path_list = [(folder + '\\' +str(n).zfill(6)) for n in range(N_images)]
    return path_list   

# generate random particles at different spots 
def pick_xy_with_min_distance(x_choices, y_choices, num_points, min_distance):
    """
    Randomly pick X and Y values from predefined lists such that the (X[i], Y[i]) pairs
    are not too close to each other (based on `min_distance`).
    
    Params:
        x_choices:    list of x ppossible x coordinates 
        y_choices:    list of x ppossible x coordinates
        num_points:   number of particles 
        min_distance: minimum separation distance between teh particles
        
    Returns:
        x_array: numpy array of X coordinates
        y_array: numpy array of Y coordinates
    """
    selected_x = []
    selected_y = []
    selected_points = []
    attempts = 0
    max_attempts = 100000  # avoid infinite loop
    
    # Loops for all points in the 
    while len(selected_x) < num_points and attempts < max_attempts:
        x = random.choice(x_choices)
        y = random.choice(y_choices)
        new_point = (x, y)

        if all(math.hypot(px - x, py - y) >= min_distance for px, py in selected_points):
            selected_points.append(new_point)
            selected_x.append(x)
            selected_y.append(y)

        attempts += 1

    if len(selected_x) < num_points:
        raise RuntimeError(f"Only found {len(selected_x)} valid points with the given spacing.")

    return np.array(selected_x), np.array(selected_y)


# creates overlapping circles. 
# @ Samadarshi .... you can use this same core for squares or traingles or any other shapes
def image_generator(outpath_path, image_size, N_particles, PtoB_ratio=0.02, packing_frac = 0.5, min_dist=2)
    '''
    Generates N images of a specified size in .jpg format
    Params:
        output_path  : list of the location+name (paths) to save the each image generated 
        image_size   : The X and Y dimensions of the image
        N_particles  : Number of particles in the image 
        PtoB_ratio   : The Pixel to micron ratio correecting for the magnification
        packing_frac : The packing fraction of the particles in the image. 
        
    Outputs:
        Xx: array of x coordinates
        Yy: array of y coordinates
        H : 2D logistic map of pixels with 0 if no particle is located and 1 otherwise
    '''
    # define the frame dimensions
    X_frames = image_size[0]
    Y_frames = image_size[1]

    # for the grids that will be later made
    xedges =  np.linspace(-X_frames/2, X_frames/2, X_frames+1)
    yedges =  np.linspace(-Y_frames/2, Y_frames/2, Y_frames+1)

    # measure typical number density
    Packing_fraction  = packing_frac

    # define the particle to box ratio
    PtoB_ratio = PtoB_ratio

    # generate a number between these values
    num_circles = N_particles

    # create a smapling list a bit smaller than the image dimensions
    length_rand = int(np.floor(X_frames/10)*10)
    
    random_list_ = sorted(np.unique(np.array(np.linspace(-length_rand/2, length_rand/2, length_rand), dtype = int)))
    
    # develop points
    Xx, Yy =  pick_xy_with_min_distance(random_list_, random_list_, num_points=N_particles, min_distance=min_dist)

    # scale the coorinates to the correct size
    X = np.array(Xx)/X_frames/10
    Y = np.array(Yy)/Y_frames/10
    scale_ = 5

    # develop the image with consistent dimensions
    fig = plt.figure(figsize = (scale_, scale_))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('equal')
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.scatter(X,Y, c = 'k', s = (PtoB_ratio*scale_*70.7)**2)
    plt.xlim([-0.05, 0.05])
    plt.ylim([-0.05, 0.05])
    plt.savefig(outpath_path+'.png', dpi=X_frames/scale_)
    plt.close()

    # now create the simulated response .... H basically encapsulates the particle coordinates via bin indices.
    H, xedges, yedges = np.histogram2d(Yy, Xx, bins=(xedges, yedges))

    return Xx,Yy, np.flipud(H)

# function to convert the 2D binary array into heatmap
def binary_to_heatmap(output_path, sigma, H_val, window):
    '''
    Converts the 2D binary histogram into a heatmap. the 2D binaryy array contains the bins whose indices correspond the particle location
    
    Params: 
        output_path: filename+location for storing the heatmap
        sigma      : Spread of the gaussian of the heatmap
        H_val      : The correspnding 2D binary array to create a heatmap
        
    Returns:
        None
    '''
    
    # window dimensions 
    X_size = window[0]
    Y_size = window[1]
    
    # resize the data into a 2D binary mask
    binary_mask =  H_val.reshape(X_size, Y_size) # Should be shape (128, 128) with 1s at centers
    
    # create heatmaps 
    heatmap = gaussian_filter(binary_mask.astype(np.float32), sigma=sigma)
    
    # arrest the values of teh htmap between 0 and 1 
    heatmap = np.clip(heatmap, 0, 1)
    
    # save as a 2D numpy array
    np.save(output_path +'.npy', heatmap)

    return None 
