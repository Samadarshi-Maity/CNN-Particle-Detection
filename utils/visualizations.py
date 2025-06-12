# contains all the plotting functions 
import matplotlib.pyplot as plt
import torch

# Function to visualize the training process
def visualize(model, device, dataset, index=0):
    '''
    Plotting function to check the detection with the ground truth.
    
    Params: 
        model :  object of the CNN model 
        device:  device type 
        index :  references the index of the image-heatmap pair
    Returns:     
        None 
    '''
    
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
