# Self-Driving-Car-from-Scratch
Self-driving cars have rapidly become one of the most transformative technologies to emerge. Fuelled by Deep Learning algorithms, they are continuously driving our society forward and creating new opportunities in the mobility sector.

From generating my own custom dataset followed by Lane Detection using OpenCV4.0 and imgaug to network architecture development using Tensorflow and Keras, and testing it on an entirely new track - - - finally succeeded in completing the entire project and it seems to be really stable so far. 

## Libraries Installation
1. OpenCV:
Find the complete installation guide here: https://www.pyimagesearch.com/opencv-tutorials-resources-guides/

2. socketio/eventlet/numpy/flask/keras/pillow/jupyter notebook:

Do : _pip install socketio/eventlet/numpy/flask/keras/pillow/jupyter notebook_

## Project Structure

The goals / steps of this project are the following:

    Use the simulator to collect data of good driving behavior
    Build, a convolution neural network in Keras that predicts steering angles from images
    Train and validate the model with a training and validation set
    Test that the model successfully drives around multiple tracks  without leaving the road
    
## Getting Started

The project includes the following files:

    Model.ipynb containing the script to create and train the model
    drive.py for driving the car in autonomous mode
    model_1.h5 containing a trained convolution neural network
    
Additionally you need to download and unpack the Udacity self-driving car simulator (Version 1 was used). To run the code start the simulator in autonomous mode, open another shell and type:

_python drive.py_

To train the model, first make a directory, drive the car in training mode around the track and save the data to this directory. The model is then trained by executing the _Model.ipynb_ using _jupyter notebook_, where you need to upload the customized dataset, perform data preprocessing, generate and train the model as well as create the validation and testing datasets. After successful training, the trained CNN can be downloaded as a _model_1.h5_ file.
I would suggest use _GOOGLE COLAB_ for faster training and execution.

![graph](https://user-images.githubusercontent.com/29462447/50538264-a00f5180-0b92-11e9-8d40-826e07ff5564.png)

![both](https://user-images.githubusercontent.com/29462447/50538271-c33a0100-0b92-11e9-9e9f-048ac4243e00.png)

You can also download the dataset that I developed for the project by typing: _git clone https://github.com/prateekralhan/track.git_

## General considerations

The simulated car is equipped with three cameras, one to the left, one in the center and one to the right of the driver that provide images from these different view points. The training track has sharp corners, exits, entries, bridges, partially missing lane lines and changing light conditions. An additional test track exists with changing elevations, even sharper turns and bumps. It is thus crucial that the CNN does not merely memorize the first track, but generalizes to unseen data in order to perform well on the test track. The model developed here was trained exclusively on the training track and completes the test track.

The main problem lies in the skew and bias of the data set. Shown below is a histogram of the steering angles recorded while driving in the middle of the road for a few laps. This is also the data used for training. The left-right skew is less problematic and can be eliminated by flipping images and steering angles simultaneously. However, even after balancing left and right angles most of the time the steering angle during normal driving is small or zero and thus introduces a bias towards driving straight. The most important events however are those when the car needs to turn sharply.

image

Without accounting for this bias towards zero, the car leaves the track quickly. One way to counteract this problem is to purposely let the car drift towards the side of the road and to start recovery in the very last moment. However, the correct large steering angles are not easy to generate this way, because even then most of the time the car drives straight, with the exception of the short moment when the driver avoids a crash or the car going off the road.

## Model Architecture

For the network architecture I decided to draw on a CNN that evolved from a previous submission for classfying traffic signs with high accuracy(>95%) given here. 
However,I included some crucial changes.

![archi](https://user-images.githubusercontent.com/29462447/50538200-dac4ba00-0b91-11e9-9bd6-c487c77cb1fd.png)

## Training

All computations were run on an Ubuntu 16.04 system with an Intel i7 7700 processor and an NVIDIA GTX 1050Ti, while the training and the code _model.ipynb_ was executed in _GOOGLE COLAB_.
Due to the problems with generating the important recovery events manually we decided on an augmentation strategy. The raw training data was gathered by driving the car as smoothly as possible right in the middle of the road for 3-4 laps in one direction. We simulated recovery events by transforming (shifts, shears, crops, brightness, flips) the recorded images using library functions from OpenCV with corresponding steering angle changes. The final training images are then generated in batches of 200 on the fly with 20000 images per epoch. A python generator creates new training batches by applying the aforementioned transformations with accordingly corrected steering angles. The operations performed are

A random training example is chosen:
    1.The camera (left,right,center) is chosen randomly
    2.Random shear: the image is sheared horizontally to simulate a bending road
    3.Random crop: we randomly crop a frame out of the image to simulate the car being offset from the middle of the road (also     downsampling the image to 64x64x3 is done in this step)
    4.Random flip: to make sure left and right turns occur just as frequently
    5.Random brightness: to simulate differnt lighting conditions

![aug](https://user-images.githubusercontent.com/29462447/50538216-3f801480-0b92-11e9-9be2-2c40fa2ae36f.png)

![pre](https://user-images.githubusercontent.com/29462447/50538220-44dd5f00-0b92-11e9-8f9c-60458c01ff5e.png)

![prepro](https://user-images.githubusercontent.com/29462447/50538221-4870e600-0b92-11e9-88d6-21f7ee8cf3f6.png)
    
## Epochs and Validation

For validation purposes 10% of the training data (about 1000 images) was held back. Only the center camera imags are used for validation. After few epochs (~10) the validation and training loss settle. The validation loss is consistently about half of the training loss, which indicates underfitting, however with the caveat that training and validation data are not drawn from the same sample: there is no data augmentation for the validation data. A more robust albeit non-automatic metric consists of checking the performance of the network by letting it drive the car on the second track which was not used in training.

We used an Adam optimizer for training. All training was performed at the fastest graphics setting.



## Conclusions

By making consequent use of image augmentation with subsequent steering angle updates we could train a neural network to recover the car from extreme events, like suddenly appearing curves change of lighting conditions by exclusively simulating such events from regular driving data, and the model seemed to be really stable on test tracks as shown:

