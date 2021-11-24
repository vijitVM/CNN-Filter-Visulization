# Convolutional Neural Network Filter Visulization

## Overview 
- Convolutional Neural Networks
- Using pre Trained Model
- How would we visulize the CNN Filters ?
- What is the Algorithm that will be Used ?
- Predictions and Plotting

## Convolutional Neural Networks 
<p>A Convolutional Neural Netowork is a Deep Learning Alogorithm. CNN takes in a input image, Learnable weights and bias in an image and tries to distinguish betwween one another.</p> 
<p>The main criteria of CNN is produce a good prediction that results in better images after continuous learning without eliminating the features.</p>

## CNN Filters
<p>In CNN Filters are used to extract information from images. These information are then used and passed through a given network to make certain predictions.  We refer to filters as kernals in any Neural Netowrk architecture. Filters help to redcue the noise in the image and also give us the basic features. </p>

<p> A neural Netowrk model can consist of multiple Filters. These Filters make up the depth of the neural network model architecture. The following image below shows hows a simple filter works on a image: </p>
<p align="center">
<img width="250" src = "Images/Filters.gif">
</p>

## Pre Trained Model 
We will be using the VGG16 Model and the basic way to load the entire model is as follows: 

```python
# Import the required Libraries
import tensorflow as tf 
import math
import numpy
import matplotlib.pyplot as plt

# Load the VGG16 Model 
model = tf.keras.applications.vgg16.VGG16(
    include_top = False, # Final Fully connected layers in CNN not inclued in the model
    weights ='imagenet', # CNN model trained on the given weights
    input_shape =(96,96,3) # 96 rows and columns and 3 channels
)
model.summary()
```
## How would we visulize the pre trained Model ?
The first step is to see how the VGG16 Model looks like and then we can decide what we want to work with. 
```
Model: "vgg16"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 96, 96, 3)]       0         
                                                                 
 block1_conv1 (Conv2D)       (None, 96, 96, 64)        1792      
                                                                 
 block1_conv2 (Conv2D)       (None, 96, 96, 64)        36928     
                                                                 
 block1_pool (MaxPooling2D)  (None, 48, 48, 64)        0         
                                                                 
 block2_conv1 (Conv2D)       (None, 48, 48, 128)       73856     
                                                                 
 block2_conv2 (Conv2D)       (None, 48, 48, 128)       147584    
                                                                 
 block2_pool (MaxPooling2D)  (None, 24, 24, 128)       0         
                                                                 
 block3_conv1 (Conv2D)       (None, 24, 24, 256)       295168    
                                                                 
 block3_conv2 (Conv2D)       (None, 24, 24, 256)       590080    
                                                                 
 block3_conv3 (Conv2D)       (None, 24, 24, 256)       590080    
                                                                 
 block3_pool (MaxPooling2D)  (None, 12, 12, 256)       0         
                                                                 
 block4_conv1 (Conv2D)       (None, 12, 12, 512)       1180160   
                                                                 
 block4_conv2 (Conv2D)       (None, 12, 12, 512)       2359808   
                                                                 
 block4_conv3 (Conv2D)       (None, 12, 12, 512)       2359808   
                                                                 
 block4_pool (MaxPooling2D)  (None, 6, 6, 512)         0         
                                                                 
 block5_conv1 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
 block5_conv2 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
 block5_conv3 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
 block5_pool (MaxPooling2D)  (None, 3, 3, 512)         0         
                                                                 
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________
```
