# import the python modules 
import torch
import torch.nn as nn
from torchsummary import summary
import logging
import os
import glob
from src.data_funnel import ParticleDataset
from torch.utils.data import DataLoader
from testguard import test_guard

class CNNModelTester:
    '''
    Class to test the training workflow. 
    Contains methods for testing the image dimensions, data loading, forward pass and backpropagation
    Provides a summary of the model.
    
    '''
    
    def __init__(self, image_path, heatmap_path, batch, model:nn.Module):
        """
        Constructor for the CNNModeltester class. Initialises the model and the log files 

        Params 
        image_path (str)   : path to the image directory
        heatmap_path (str) : path to the heatmap directory
        batch (str)        : numbe rof images in a batch
        model (obj)        : model object
            
        """
        #Intialise the instance variables 
        self.model = model
        self.image_path = image_path
        self.heatmap_path = heatmap_path
        self.batch  = batch

        # push the model into the device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # start the logging operation through the logger instance
        self.logger = logging.getLogger()
        self.logger.info(f"Initialized CNNModelTester")
        self.logger.info(f"Model running on {self.device}")

        # sets the a flag that dictates whether the subsequent test operation shoul coontinue
        # check the test_guard decorator that for more information.
        self.can_continue = True

    @test_guard
    def check_dir_file_counts(self):
        """
        Checks if two directories exist and contain the same number of files.
        Logs the result using your logger.
        
        Returns:
            result (dict): Result summary with existence and file count comparison.
        """
        result = {
            "images_exists": os.path.isdir(self.image_path),
            "heatmaps_exists": os.path.isdir(self.heatmap_path),
            "same_file_count": False,
            "images_file_count": 0,
            "heatmaps_file_count": 0
        }
    
        self.logger.info("Checking image and heatmap directories")

        
        if not result["images_exists"] or not result["heatmaps_exists"]:
            if not result["images_exists"]:
                self.logger.error(" Image Directory does not exist")
            if not result["dir2_exists"]:
                self.logger.error(" Heatmaps Directory does not exist")

            self.can_continue = False
            return result

        image_files = glob.glob(self.image_path + '\*')
        heatmap_files = glob.glob(self.heatmap_path + '\*')
    
        result["images_file_count"] = len(image_files)
        result["heatmaps_file_count"] = len(heatmap_files)
        result["same_file_count"] = len(image_files) == len(heatmap_files)
    
        self.logger.info(f" images file count: {result['images_file_count']}")
        self.logger.info(f" heatmaps file count: {result['heatmaps_file_count']}")
    
        if result["same_file_count"]:
            self.logger.info(" File counts match.")
            self.logger.info("Passing the file through the data funnel.")

            image_data = ParticleDataset(image_files[0:self.batch], heatmap_files[0:self.batch])
            image_loader = DataLoader(image_data, batch_size = self.batch)

            self.batch_images, self.batch_heatmaps = next(iter(image_loader ))
        else:
            self.logger.warning("File counts do not match.")
            
            self.can_continue = False
        return result 
            
        
    @test_guard
    def check_shape_match(self):
        """
        Checks if the image batch shape matches with the shape of the heatmap

        """
        try:
            assert self.batch_images.shape == self.batch_heatmaps.shape, (
                f"Input shape mismatch: {self.batch_images.shape} vs {self.batch_heatmaps.shape}"
            )
            self.logger.info(" Input shape check passed.")
            
        except AssertionError as e:
            self.logger.error(e)
            self.can_continue = False
      
            
    @test_guard    
    def forward_pass_runs(self):
        """
        Checks if the model internals are correctly tied though a forward pass.
        """
        self.batch_images = self.batch_images.to(self.device)
        try:
            y = self.model(self.batch_images)
            self.logger.info(f" Forward pass successful. Output shape: {y.shape}")
        except Exception as e:
            self.logger.error(f" Forward pass failed: {e}")
            self.can_continue = False
    
    @test_guard
    def check_output_shape_matches_input(self):
        """
        Checks if the output shape is correct.
        """

        try:
            y = self.model(self.batch_images.to(self.device))
            assert y.shape == self.batch_heatmaps.shape, (
            f"Output shape {y.shape} does not match input spatial dims {self.batch_heatmaps.shape}"
            )
            self.logger.info(f"Output shape matches input spatial dims: {y.shape}")
        except Exception as e:
            self.logger.error(e)
            self.can_continue = False

    @test_guard
    def backward_pass_runs(self):
        """
        Check if the backpropagation executes correctly
        """
        target = torch.randn_like(self.model(self.batch_images.to(self.device)))
        loss_fn = nn.MSELoss()
        try:
            y = self.model(self.batch_images)
            loss = loss_fn(y, target)
            loss.backward()
            self.logger.info(f" Backward pass ran without errors.")
        except Exception as e:
            self.logger.error(f"Backward pass failed: {e}")
            self.can_continue = False
    
    @test_guard
    def print_model_summary(self):
        """
        Computes the model summary : layers and the parameters involved.
        """
        self.logger.info("Model summary:")
        summary(self.model, input_size = tuple(self.batch_images.shape)[1:], batch_size = self.batch)
