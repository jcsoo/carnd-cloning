# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[img_center]: ./examples/center.jpg "Center Image"
[img_left]: ./examples/left.jpg "Left Image"
[img_right]: ./examples/right.jpg "Right Image"
[img_center_flipped]: ./examples/center_flipped.jpg "Center Image Flipped"
[img_center_cropped]: ./examples/center_cropped.jpg "Center Image Flipped"

[img_recovery]: ./examples/recovery.jpg "Recovery Image"


## Rubric Points
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* loader.py containing the data loader and image generators
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run1.mp4 showing a run of the model on Track 1
* run2.mp4 showing a run of the model on Track 2
* writeup_report.md summarizing the results
* NOTES.txt showing all models that were tested as well as their performance
* models/ containing code for all models that were tested.

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

   Original Image

   ![alt text][img_center] 

   Cropped Image (top 80 and bottom 50 pixels removed)
   
   ![alt text][img_center_cropped] 

Normalization
   - The incoming image is converted to a floating point value centered around the midpoint with a range of [-1..1].
   - The V channel of the YUV image is l2 normalized.

Convolutional Layers

   - Three convolutional layers with 5x5 kernels of depths 24, 36, and 38, each with RELU activation. The first two
   layers used 2x2 subsampling.
   - Two convolutional layers with 3x3 kernels of depths 64, each with RELU activation

Dropout Layer

   - A single dropout layer with 0.1 dropout

Dense Layers

   - Three dense layers with weights of 100, 50, and 10 and RELU activation

Output Layer

   - A single output layer with no additional activation.

#### 2. Attempts to reduce overfitting in the model

The model includes a dropout layer to reduce overfitting. Additionally, sessions from both tracks
in both directions were used to ensure a reasonably large training set. Variants with no dropout,
dropout 0.5, and dropout 0.1 were tested. Dropout 0.1 was found to be the most effective.

#### 3. Model parameter tuning

The model uses the Adam optimizer, with the learning rate set to 0.0001.

#### 4. Appropriate training data

Training data was collected from centerline driving of both tracks in both directions, and the example dataset was also used. 

![alt text][img_center] 

Additionally, some variants were trained with recovery sessions, where the vehicle was aimed away from the centerline and appropriate centering input was provided.

![alt text][img_recovery] 

To augment the data, two approaches were taken:

   - Left / Right flipping: An copy of the image was flipped horizontally and added to the data set, along with
   a reflected steering input.

   ![alt text][img_center_flipped] 

   - Left / Right camera: The Left and Right camera images were optionally added to the data set along with
   an adjusted factor to simulate recovery steering input of +/- 0.2.

   ![alt text][img_left] 

   ![alt text][img_right]

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

Please see [NOTES.txt](NOTES.txt) for an overview of the different variations that were implemented and tested.

#### 2. Final Model Architecture

The final architecture was the NVIDIA end-to-end architecture with some differences:

   - Input image size of 50 x 320 instead of 66 x 200
   - Only the first two layers were subsampled 
   - Dropout of 0.1 added after convolutional layers
   - RELU activation for all layers

Please see the discussion in section 1 for additional details.

#### 3. Creation of the Training Set & Training Process

Please see the discussion in section 4, "Appropriate training data".

### Observations

Building a model that could drive the first track was fairly straightforward - this was accomplished on the 8th iteration
by adding left + right camera augmentation. Training a model to drive the second track took more data including recovery
samples. The biggest challenge was to make a model that could drive both tracks successfully.

Part of this difficulty was a mistake that I made when originally implementing the NVIDIA End-to-End model: I forgot to 
specify an activation type for the Dense() layers, so Keras defaulted to linear activation. It was surprising that the
model was able to learn at all! During this phase of development, I resorted to techniques such as adding a multiplier
to the steering angle in order to force the model to steer sharply enough around tight corners. A side effect was a lot
undamped oscillation on shallow curves and straightaways, particularly on the first track.

Adding in RELU activation made a huge difference in the ability of the system to learn. I was immediately able to back off
the steering angle multiplier and remove left + right augmentation and still have a system that could handle the corners
on the second track.

Additionally, normalizing the V channel seemed to help significantly on the second track. Without it, the model would often 
stop recognizing the road edges when transitioning to shadow and run off the road.

Selecting a tighter crop area seemed to have promise but needs more research and experimentation. Theoretically it should
be excluding un-needed data and reducing the number of weights needed, but in this case a side effect of the change was
removing subsampling from some of the convolutional layers. Some tests that added back partial 1x2 subsampling seemed
to show worse performance than expected.

The final model chosen performed reasonably well on both track 1 and track 2.

Run 1 on Track 1: [run1.mp4](Run 1 Video)

Run 2 on Track 2: [run2.mp4](Run 2 Video)
