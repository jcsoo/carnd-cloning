# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

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
* model.h5 containing a trained convolution neural network 
* run1.mp4 showing a run of the model on Track 1
* run2.mp4 showing a run of the model on Track 2
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

This model is based on the Nvidia model described in [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316). It consists of the following layers:

Preprocessing

   - The incoming image is converted to YUV color space.

Cropping

   - The incoming 320x160 image is cropped to remove the sky and the vehicle at the top and bottom.

Normalization
   - The incoming image is converted to a floating point value centered around the midpoint with a range of [-1..1].
   - The V channel of the YUV image is l2 normalized.

Convolutional Layers

   - Three convolutional layers with 5x5 kernels of depths 24, 36, and 38, each with 2x2 subsampling and RELU activation
   - Two convolutional layers with 3x3 kernels of depths 64, each with 2x2 subsampling and RELU activation

Dropout Layer

   - A single dropout layer with 50% dropout

Dense Layers

   - Three dense layers with weights of 100, 50, and 10 and RELU activation

Output Layer

   - A single output layer

#### 2. Attempts to reduce overfitting in the model

The model includes a dropout layer to reduce overfitting. Additionally, sessions from both tracks
in both directions were used to ensure a reasonably large training set.

#### 3. Model parameter tuning

The model uses the Adam optimizer, with the learning rate set to 0.0001.

#### 4. Appropriate training data

Training data was collected from centerline driving of both tracks in both directions, and the example dataset was also used. 

Additionally, some variants were trained with recovery sessions, where the vehicle was aimed away from the centerline and appropriate centering input was provided.

To augment the data, two approaches were taken:

   - Left / Right flipping: An copy of the image was flipped horizontally and added to the data set, along with
   a reflected steering input.
   - Left / Right camera: The Left and Right camera images were optionally added to the data set along with
   an adjusted factor to simulate recovery steering input of +/- 0.2.

Additionally, an optional steering scaling factor was added to teach the model to use stronger steering input. The value of this parameter is discussed later.

Finally, because a significant amount of samples have low steering angle compared to samples that have high
steering angles, a parameter was added to allow excluding samples with an absolute steering angle below a
threshold. This reduced training time and did not seem to impact model performance.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The basic design approach was to use a specific model (simpler at first, then more complex) with a standard training process, using a 80 / 20 split of training / validation samples, varying parameters and options such as the specific training sets used.

After each training session, the model was downloaded to the simulator and tested, initially at low speeds and then at higher speeds if successful. Initially, only sample data from the first track was used and only
that track was used for testing. After successful completion of the first track, models were trained with
sample data from the second track, sometimes exclusively and eventually in combination.

After each iteration, parameters and configurations were modified slightly. After a few iterations, a model similar to the NVIDIA End to End model was used. Different color spaces were tested, as well as different
image sizes and normalization methods. Each iteration was stored in the Git repository along with detailed notes about the validation performance and simulation performance.

#### 2. Final Model Architecture

Please see the discussion in section 1.

#### 3. Creation of the Training Set & Training Process

Please see the discussion in section 4, "Appropriate training data".