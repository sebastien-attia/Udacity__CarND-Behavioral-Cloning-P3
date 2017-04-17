# **Behavioral Cloning Project**

[//]: # (Image References)

[image1]: ./report/LeNet/run_LeNet_1.png "LeNet - Overfitting"
[image2]: ./report/center.jpg "Center image"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model's architecture

The model's architecture is the following:
_________________________________________________________________

Layer (type)               | Output Shape         | Param #    
:-------------------------- |: ----------------    :|:----------:
lambda_1 (Lambda)          |  (None, 160, 320, 3)  | 0
cropping2d_1 (Cropping2D)  |  (None, 75, 320, 3)   | 0      
conv2d_1 (Conv2D)          |  (None, 75, 320, 3)   | 1 824     
dropout_1 (Dropout)        |  (None, 36, 158, 24)  | 0
conv2d_2 (Conv2D)          |  (None, 16, 77, 36)   | 21 636
dropout_2 (Dropout)        |  (None, 16, 77, 36)   |     0
conv2d_3 (Conv2D)          |  (None, 6, 37, 48)    | 43 248
dropout_3 (Dropout)        |  (None, 6, 37, 48)    |     0
conv2d_4 (Conv2D)          |  (None, 4, 35, 64)    | 27 712
dropout_4 (Dropout)        |  (None, 4, 35, 64)    |     0
conv2d_5 (Conv2D)          |  (None, 2, 33, 64)    | 36 928
dropout_5 (Dropout)        |  (None, 2, 33, 64)    |     0
flatten_1 (Flatten)        |  (None, 4224)         |     0
dense_1 (Dense)            |  (None, 100)          | 422 500
dropout_6 (Dropout)        |  (None, 100)          |     0
dense_2 (Dense)            |  (None, 50)           |   5 050
dropout_7 (Dropout)        |  (None, 50)           |     0
dense_3 (Dense)            |  (None, 10)           |    510
dropout_8 (Dropout)        |  (None, 10)           |     0
dense_4 (Dense)            |  (None, 1)            |     11


Total params: 559,419.0  
Trainable params: 559,419.0  
Non-trainable params: 0.0  
_________________________________________________________________

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 9).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 146-198). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 41).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
A correction angle of:
- 0.25° is added to the steering for the left side images,
- -0.25° is added to the steering for the right side images,

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My strategy was to start with the standard LeNet architecture.

In order to increase the size of the data set:
- the images were flipped with a steering multiplied by (-1),
- the left and right side images were added too to the data set, with a steering correction_steering of 0.25° added (left images) or substracted (right images).

Then, I randomly shuffled the data set and put:
- 80% of the data into the training set and,
- 20% of the data into the validation set.

##### a- First iteration
I have done a first iteration with the following size of data set:
- 63 859 samples for the training set,
- 15 965 samples for the validation set.

The car goes out of track the first five seconds of the run.

##### b- Second iteration
I reused the previous model and trained it with additional samples, to have at the
end the following size of data set:
- 122 293 (63 859 + 32361 + 26 073)
- 30 575 (15965 + 8091 + 6519)

Now the car was able to drive autonomously around the track without leaving the road, but it was because the model "remembers" the track.
When I traced the mean square error of the training and validation sets, we got the following:

![LeNet - overfitting][image1]

The model clearly overfits.

##### b- Third iteration
I added in this iteration drop outs at each layer of the model, with a dropout of 0.6,
to limit/remove the overfitting.

But it did not help much and I was not able to have a model with the vehicle being able
to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

To avoid or at least limit the overfitting, I tried a more powerful network: [a Convolutional Neural Network used by NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

This CNN (model.py lines 41-74) has more convolutional layers more than the LeNet, more filters and 1 more fully connected layer.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 3 laps on track one using center lane driving. Here is an example image of center lane driving:

![Center image][image2]

I recorded too a laps in clockwise to limit the bias of having only data
of the vehicle driving in the counter-clockwise.

Then I augmented the data set by:
- flipping the images and multiply by (-1) the steering,
- adding the left and right side images, with steering corrected of +/- 0.25°

The size of:
- the training set is 122 293 images and,
- the validation set is 30 575 images








I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
