# **Behavioral Cloning** 

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* itr_model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* Build Model.ipynb for initial development of the model
* Explore_Data.ipynb to explore the initial recorded datasets and almalgamate multiple runs
* run3.mp4 is the final submission video

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py itr_model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is defined in the `nvidia_paper_model` function. This is based on the nvidia model discussed in the lectures and consists of 5 layers of convolutions with the results fed into four dense layers to produce one output.

All the activation layers are all relus


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of drive in both directions on the circuit and both tracks as training data

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The nvidia behavioural cloning network proved sufficient for the goal of behaviour cloning.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. To look for overfitting, I looked for when the training error deviated from the validation error, particularly when the validation error increased but training error decreased.

The final step was to run the simulator to see how well the car was driving around track one. At first the car didn't stay on the track very well so I looked to diversify the dataset as a first step.

I created training datasets with the car driving in both directions on both circuits in order to create a diversified dataset. This proved sufficient to get the car to follow the basic track successfully.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 

| Layer        | Settings           | Notes | 
| ------------- |:-------------:| ----------- |
| lambda      | input / 255.0 - 0.5 | regularise data |
| Cropping2D | crop image to ((70,25) (0,0)) |  |
| Conv2D | 24 filters, 5x5 conv, 2x2 stride |
| Conv2D | 36 filters, 5x5 conv, 2x2 stride |
| Conv2D | 48 filters, 5x5 conv, 2x2 stride |
| Conv2D | 64 filters, 3x3 conv, 1x1 stride |
| Conv2D | 64 filters, 3x3 conv, 1x1 stride |
| Flatten ||
| Dense | 100 |
| Dense | 50 |
| Dense | 10 |
| Dense | 1 | to get steering angle |


#### 3. Creation of the Training Set & Training Process

To create the dataset I first drove a couple of laps of track one, the proved insufficient so I drove in the opposite direction on track one for a couple of loops.
I also did the same with track two, drove in one direction then the other.

After the collection process, I had 7921 number of data points. 

I finally randomly shuffled the data set and put 20% of the data into a validation set to develop the model. This can be seen the `Build Model.ipynb` notebook. The model developed in this notebook was then recoded into the `model.py` file to give an end to end training routine. I chose an adam optimiser so that I didn't experiment with learning rates.

Based on watching validation loss vs training loss on epochs, I found that 5 epochs were sufficient to get a good enough model to "solve" track 1.
