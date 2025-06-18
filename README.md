
# CNN-Particle-Detection
( <i> under development </i>
## Dependencies
Python       == 3.9.7 <br>
Pytorch      == 2.0.1+cu117<br>
Numpy        == 1.25.2<br>
matplotlib   == 3.4.3<br>
pandas       == 1.3.4<br>
cv2          == 4.11.0<br>
scipy        == 1.13.1<br>
logging      == 0.5.1.2<br>
 (More recent torch versions should also work; a lower version is used for compatibility with our NVIDIA GPU)
## Objective:
This project aims to precisely detect the centers of swarming particles that are heavily overlapping, which are primarily spherical. We are currently working on extending the detection to anisotropic particles, with a particular focus on detecting high aspect ratio rods and chains formed by attaching particles flexibly, which we will soon update. Particular focus was given to ensure that the models are 'lightweight' enough to allow for fast detection while meeting the expectation of >95% accuracy.  

Please click the link below to watch a short clip of the swarming particles. Videos of other types of swarms are present on my website. 
![click here to download an short clip of the swarming particles](https://github.com/Samadarshi-Maity/CNN-Particle-Detection/raw/main/Roger_data.mp4)

Traditional computer vision approaches for detecting particles often reach their limits when particles show significant overlaps (as in the above video) or are positioned inside a complex, crowded environment. Deep Learning using Convolutional Neural Networks can be an excellent alternative for such cases. They also offer higher detection accuracy and do not require "hard-tuning" several parameters, often done manually in traditional computer vision toolkits. 


## Contents 
This repository contains CNN models: <b>UNet</b> and <b>ResNet</b>, and associated modules for generating synthetic images of overlapping particles, training the CNN models, and evaluating their performance (F1, confusion, and accuracy). The training approach was developed particularly for cases where a large collection of unlabeled imaging data is available. We implement a semi-supervised learning approach with "human in the loop". Below is a schematic of the training approach.
<p align="center">
<img src="https://github.com/Samadarshi-Maity/CNN-Particle-Detection/raw/main/Images_description/Training_CNN.svg" alt="Description of the image" style="height: 400px; width: auto;" />
</p>

### Usage
The package consists of Jupyter notebooks that can be run after correctly modifying (if necessary) the filepath to the preferred data sources/destinations and adjusting to the desired parameters for the entire training cycle after directly cloning this repository. 
1. The synthetic images are randomly combined with the experimental data for the first training cycle, after which we can train purely on the annotated experimental data. The training steps are detailed in the "training.ipynb" notebook. The associated classes and functions in train_modules.py and the models (UNet and ResNet) are located in " models.py" within src. 
  
2. After the first training, we can evaluate the performance of the model by using the sample code presented in the "Prediction.ipynb" notebook, where we can extract particle coordinates from the images. It contains the functions to compute the metrics (F1, accuracy, precision, and recall) redefined for this task, and visualize the predictions, and the associated classes and functions are positioned within the utils subfolder. 

3. If the desired Metrics are not achieved, then we can increase the predictions by checking the predictions of the model, which I do using <a href = "https://imagej.net/">ImageJ</a>. The tool helps to detect the wrongly detected particles and also quickly locate the correct centers (it is a beautiful tool, and can do tons of other things ...do check it out). The corrected coordinates can be converted into heatmaps using the functions defined in coord_htmp_exchange.py within utils. 

4. The folder subsystem for correctly placing the images and the evaluations at various stages can be directly managed using the config.YAML file. I have placed a template for such management. Please feel free to modify it as per your data locations, ensuring the correct changes are also made into each notebook.
   
## Results
 The following image compares the detection quality of detection (yellow circles for detected particles) using traditional computer vision methods (preprocessed in ImageJ and detected in MATLAB imaging toolkit) vs the direct (unpreprocessed) and progressively trained UNet Network. The Accuracy, Precision, F1, and Recall range between 0.5 to 0.6, whereas the UNet-based training yields values ranging between 0.95 to 0.98.

<p align="center">
<img src="https://github.com/Samadarshi-Maity/CNN-Particle-Detection/raw/main/Images_description/improvement.svg" alt="Description of the image" style="height: 500px; width: auto;" />
</p>


