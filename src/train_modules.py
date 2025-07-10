# ................ import python packages ................
import subprocess
from tqdm import tqdm
import torch
from torch import nn
        
        
# Function to train the model step by step .............
def train_model(model, train_loader, val_loader=None, epochs=15, lr=1e-3): 
    """
    Function that encapsulates all the training steps and protocols into a single place.
    
    Params: 
        model        : The CNN models that are to be trained
        train_loader : Pytorch dataloader for loader for loading the training data.
        val_loader   : Loads the validation dataset. 
        epochs       : no. of training cycles
        lr           : learning rate 
    Returns:
        model        : returns the 'trained' CNN model 
    """
    
    # ...........set the device to GPU if it is available otherwise set it to cpu  .... 
    # ..... if the GPU is detected then this code should be able to operate on the GPU .... check the log file for this 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Instantiate the optimizer ..... Adam optimizer is heavy, but it does the best job. 
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
            
            # predict  and check the loss
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
                
                # load the data for validation
                for img, heatmap in val_loader:
                    img, heatmap = img.to(device), heatmap.to(device)
                    
                    #  compute the validation loss
                    pred = model(img)
                    loss = loss_fn(pred, heatmap)
                    total_val_loss += loss.item()
            
            # compute the loss in the validation   
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs} || Train Loss: {avg_train_loss:.4f} || Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} || Train Loss: {avg_train_loss:.4f}")

    return model
