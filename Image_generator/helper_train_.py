# keep the modules needed for the training purposes. 
# if sonething goes wrong .... email @ Samadarshi 

# ................ import some modules ................
#import os
import subprocess
#import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt
import logging

def get_nvidia_info():
    '''
    Check for the GPU being detected correctly 
    Logs the output of nvidia-smi
    '''
    # set the logger 
    logger = logging.getLogger(__name__)

    try:
        output = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
        logger.info("....... NVIDIA-SMI Output ......\n%s", output)

        # Parse GPU name and driver version 
        lines = output.split('\n')
        for line in lines:
            if "Driver Version" in line and "CUDA Version" in line:
                logger.info("Driver and CUDA Versions: %s", line.strip())
            elif "GPU" in line and "|" in line:
                logger.info("GPU Info Line: %s", line.strip())
    
    # for command process failure 
    except subprocess.CalledProcessError as e:
        logger.error("nvidia-smi command failed: %s", e)
        
    # for not finding the drivers    
    except FileNotFoundError:
        logger.error("nvidia-smi not found. NVIDIA drivers might not be installed.")
        
        
        
# Function to train the model 
def train_model(model, train_loader, val_loader=None, epochs=10, lr=1e-3):
    
    # ...........set the device to GPU if it is available otherwise set it to cpu  .... 
    # ..... if the GPU is detected then this code should be able to operate on the GPU .... check the log file for this 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Intantiate the optimizer ..... Adam is heavy but does the best job. 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # MSE loss function since we perform a heatmap regression
    loss_fn = nn.MSELoss()
    
    # set the training loops ... default set to 10
    for epoch in range(epochs):
        
        
        # Set the model to train 
        model.train()
        
        # set the train loss to 0 
        total_train_loss = 0
        
        # load the training data: batch buy batch
        for img, heatmap in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            img, heatmap = img.to(device), heatmap.to(device)
            
            # predict and check the loss
            pred = model(img)
            loss = loss_fn(pred, heatmap)
            
            # reset the gradient 
            optimizer.zero_grad()
            
            # back propagation
            loss.backward()
            
            # take the next step
            optimizer.step()
            
            # total loss for the batch 
            total_train_loss += loss.item()
        
        # Compute the training loss for the batch 
        avg_train_loss = total_train_loss / len(train_loader)

        # validation --->  @ Samadarshi Maity ... You can now choose this.
        if val_loader is not None:
            
            # set the model to evaluation mode 
            model.eval()
            total_val_loss = 0
            
            # gradients not needed during the evaluation 
            with torch.no_grad():
                
                for img, heatmap in val_loader:
                    img, heatmap = img.to(device), heatmap.to(device)
                    
                    
                    pred = model(img)
                    loss = loss_fn(pred, heatmap)
                    total_val_loss += loss.item()
            
            # compute the loss in the validation   
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs} || Train Loss: {avg_train_loss:.4f} || Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} || Train Loss: {avg_train_loss:.4f}")

    return model
    
    
# Function to visualize the training process
def visualize(model, dataset, index=0):
    
    # Set the model to eval mode 
    model.eval()
    
    # push the model parameters to the device 
    device = next(model.parameters()).device
    
    # Do the predictions 
    img, gt_heatmap = dataset[index]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))
    
    # Push the prediction into the cpu can convert to plottable format
    pred = pred.squeeze().cpu().numpy()
    
    # make figures for the plotting.
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title("Input Image")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_heatmap.squeeze())
    plt.title("Ground Truth Heatmap")

    plt.subplot(1, 3, 3)
    plt.imshow(pred)
    plt.title("Predicted Heatmap")

    plt.tight_layout()
    plt.show()
