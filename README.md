
# CNN-Particle-Detection
## Dependencies
Python       == 3.9.7 <br>
Pytorch      == 2.0.1+cu117 (Higher versions also work, lower version is used here for compatibility with the available NVDIA GPU)<br>
Numpy        == 1.25.2<br>
matplotlib   == 3.4.3<br>
pandas       == 1.3.4<br>
(OpenCV)cv2  == 4.11.0<br>
scipy        == 1.13.1<br>
logging      == 0.5.1.2<br>

## Objective:
This project aims to precisely detect the centers of heavily overlapping particles, which are primarily spherical. We have ongoing efforts to extend the detection to anisotropic particles and particularly towards detecting high aspect ratio rods and chains created by attaching particles flexibly, which we will soon update. Particular focus was given to ensure that the models are 'lightweight' enough to allow for fast detection while meeting the expectation of >95% accuracy.  

Traditional computer vision approaches start to reach their limits when particles show significant overlaps or are positioned inside a complex, crowded environment. Deep Learning using Convolutional Neural Networks can be an excellent alternative for such cases. They also offer higher detection accuracy and do not require "hard-tuning" several parameters, often done manually in traditional computer vision toolkits. 

## Contents 
This repository holds CNN models (architectures and weights) like UNet and ResNet, and associated modules for generating synthetic images of overlapping particles, training the CNN models, and evaluating their performance (F1, confusion, and accuracy). The training approach was developed particularly for cases where there is a large collection of unlabelled imaging data. We implement a semi-supervised learning approach with "human in the loop". Below is a schematic of the training approach.
<p align="center">
<img src="https://github.com/Samadarshi-Maity/CNN-Particle-Detection/raw/main/Images_description/Training_CNN.svg" alt="Description of the image" style="height: 300px; width: auto;" />
</p>




