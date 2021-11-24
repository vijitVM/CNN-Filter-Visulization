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
The above summary of the VGG16 Model shows the Input layer and the outup shape of each layer and the total number of parameters. In VGG16 each block comprises of  multiple 2D conv layer followed by a max pooling layer. 

The max pooling layer in CNN basically it selects the maximum value from the feature map that is covered inside a filter. Each CNN Layer can be accessed using the `layer.name` property. This will yield the output `block#_conv#` where `#` refers to an interger value. 

The second Step now would be  to use only the  specific layers in the VGG16 model,  in simple words we would create a submodel as follows: 
```python
def get_submodel(name):
  return tf.keras.models.Model(
      model.input, 
      model.get_layer(name).output)
```

The Third  and Final Step would be to create the Random Image and then Visulize it, which can be done as follows: 
```python 
# create the Image 
def create_image():
  return tf.random.uniform((96,96,3))

# Visulize the Image 
def plot(image, title ='Random Generated Image'):
  image = image -tf.math.reduce_min(image)
  image = image / tf.math.reduce_max(image) # scaling the image for values between 0 & 1
  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])
  plt.title(title)
  plt.show()

```

## What is the Algorithm that will be used ?
For this purpose I'll be utilising the Gradient Ascent Algorithm. The main aim of this algorithm is to maximize the loss values and thus inturn this will help the neural network  learn more improved features.  This can be done as follows: 

```python 
def filter_visulise(layer_name, filter_index =None, max_iters = 500):
  submodel =get_submodel(layer_name)
  num_filters = submodel.output.shape[-1]

  if filter_index is None:
    filter_index =tf.experimental.numpy.random.randint(0,num_filters -1)
  assert num_filters > filter_index, 'Filer_index is out of bonds'

  image =create_image()
  loss_step =int(max_iters / 50)
  
  # Gradient Ascent 
  for i in range(0,max_iters):
    with tf.GradientTape() as tape:
      tape.watch(image)
      out = submodel(tf.expand_dims(image,axis=0))[:,:,:,filter_index]
      loss = tf.math.reduce_mean(out)
    gradient = tape.gradient(loss,image)
    gradient =tf.math.l2_normalize(gradient)
    image = image + gradient * 0.1 

    if (i + 1) % loss_step == 0:
      print(f'Iteration: {i + 1}, Loss : {loss.numpy():4f}')


  plot(image, f'{layer_name},{filter_index}')
  ```

##Plotting the Results 
After we have created our ALogirthm we will now just run it and visulize the output, which can be done as follows: 
```python 

layer_name = 'block1_conv1' #param [''block1_conv1','block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block4_conv1', 'block4_conv2', 'block5_conv1', 'block5_conv2', 'block5_conv3']
filter_visulise(layer_name)
```

The output would be as follows for a random generated Image: 
```
Iteration: 10, Loss : 0.417100
Iteration: 20, Loss : 0.438312
Iteration: 30, Loss : 0.459604
Iteration: 40, Loss : 0.480967
Iteration: 50, Loss : 0.502391
Iteration: 60, Loss : 0.523861
Iteration: 70, Loss : 0.545367
Iteration: 80, Loss : 0.566900
Iteration: 90, Loss : 0.588461
Iteration: 100, Loss : 0.610050
Iteration: 110, Loss : 0.631658
Iteration: 120, Loss : 0.653284
Iteration: 130, Loss : 0.674928
Iteration: 140, Loss : 0.696581
Iteration: 150, Loss : 0.718240
Iteration: 160, Loss : 0.739906
Iteration: 170, Loss : 0.761580
Iteration: 180, Loss : 0.783257
Iteration: 190, Loss : 0.804935
Iteration: 200, Loss : 0.826616
Iteration: 210, Loss : 0.848296
Iteration: 220, Loss : 0.869977
Iteration: 230, Loss : 0.891657
Iteration: 240, Loss : 0.913337
Iteration: 250, Loss : 0.935019
Iteration: 260, Loss : 0.956702
Iteration: 270, Loss : 0.978385
Iteration: 280, Loss : 1.000067
Iteration: 290, Loss : 1.021750
Iteration: 300, Loss : 1.043433
Iteration: 310, Loss : 1.065115
Iteration: 320, Loss : 1.086798
Iteration: 330, Loss : 1.108481
Iteration: 340, Loss : 1.130163
Iteration: 350, Loss : 1.151846
Iteration: 360, Loss : 1.173529
Iteration: 370, Loss : 1.195211
Iteration: 380, Loss : 1.216894
Iteration: 390, Loss : 1.238577
Iteration: 400, Loss : 1.260260
Iteration: 410, Loss : 1.281942
Iteration: 420, Loss : 1.303625
Iteration: 430, Loss : 1.325307
Iteration: 440, Loss : 1.346990
Iteration: 450, Loss : 1.368673
Iteration: 460, Loss : 1.390355
Iteration: 470, Loss : 1.412038
Iteration: 480, Loss : 1.433721
Iteration: 490, Loss : 1.455404
Iteration: 500, Loss : 1.477086
```
