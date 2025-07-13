import torch
import torch.nn as nn         
import torch.optim as optim

def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    """
    Save the model state and the training  state as checkpoint 

    Params: 
        model(nn.Module) : The CNN model 
        optimizer        : The Optimizer used during the training
        epochs           : The epochs covered ats stop
        loss             : The losss value at stop  
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename='checkpoint.pth', device='cpu'):
    """
    Load the model state and the training state from checkpoint

    Params: 
        filename: Path to the checkpoint  
        device: the device to load to
    Returns: 
        model(nn.module) : CNN model
        optimizer        : State of the optimizer used
        epoch            : Epochs completed
        loss             : current loss value 
    """
    # load the check point 
    checkpoint = torch.load(filename, map_location=device)

    # load the model and the optimizer 
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # get epochs and loss
    epoch = checkpoint.get('epoch')
    loss = checkpoint.get('loss')

    print("Loaded checkpoint")
    return model, optimizer, epoch, loss