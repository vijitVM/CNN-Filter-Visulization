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


