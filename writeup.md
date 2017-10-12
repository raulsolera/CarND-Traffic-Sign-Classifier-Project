#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/class_counter.png "Train set examples per class"
[image2]: ./writeup_images/label_distribution.png "Histograms of label distribution in the different sets"
[image3]: ./writeup_images/label_distribution_2.png "Class distribution comparison between sets"

[image4]: ./writeup_images/sample-original-0.png "class 0"
[image5]: ./writeup_images/sample-original-1.png "class 1"
[image6]: ./writeup_images/sample-original-2.png "class 2"
[image7]: ./writeup_images/sample-original-3.png "class 3"
[image8]: ./writeup_images/sample-original-4.png "class 4"
[image9]: ./writeup_images/sample-original-5.png "class 5"
[image10]: ./writeup_images/sample-original-6.png "class 6"
[image11]: ./writeup_images/sample-original-7.png "class 7"
[image12]: ./writeup_images/sample-original-8.png "class 8"
[image13]: ./writeup_images/sample-original-9.png "class 9"
[image14]: ./writeup_images/sample-original-10.png "class 10"
[image15]: ./writeup_images/sample-original-11.png "class 11"
[image16]: ./writeup_images/sample-original-12.png "class 12"
[image17]: ./writeup_images/sample-original-13.png "class 13"
[image18]: ./writeup_images/sample-original-14.png "class 14"
[image19]: ./writeup_images/sample-original-15.png "class 15"
[image20]: ./writeup_images/sample-original-16.png "class 16"
[image21]: ./writeup_images/sample-original-17.png "class 17"
[image22]: ./writeup_images/sample-original-18.png "class 18"
[image23]: ./writeup_images/sample-original-19.png "class 19"
[image24]: ./writeup_images/sample-original-20.png "class 20"
[image25]: ./writeup_images/sample-original-21.png "class 21"
[image26]: ./writeup_images/sample-original-22.png "class 22"
[image27]: ./writeup_images/sample-original-23.png "class 23"
[image28]: ./writeup_images/sample-original-24.png "class 24"
[image29]: ./writeup_images/sample-original-25.png "class 25"
[image30]: ./writeup_images/sample-original-26.png "class 26"
[image31]: ./writeup_images/sample-original-27.png "class 27"
[image32]: ./writeup_images/sample-original-28.png "class 28"
[image33]: ./writeup_images/sample-original-29.png "class 29"
[image34]: ./writeup_images/sample-original-30.png "class 30"
[image35]: ./writeup_images/sample-original-31.png "class 31"
[image36]: ./writeup_images/sample-original-32.png "class 32"
[image37]: ./writeup_images/sample-original-33.png "class 33"
[image38]: ./writeup_images/sample-original-34.png "class 34"
[image39]: ./writeup_images/sample-original-35.png "class 35"
[image40]: ./writeup_images/sample-original-36.png "class 36"
[image41]: ./writeup_images/sample-original-37.png "class 37"
[image42]: ./writeup_images/sample-original-38.png "class 38"
[image43]: ./writeup_images/sample-original-39.png "class 39"
[image44]: ./writeup_images/sample-original-40.png "class 40"
[image45]: ./writeup_images/sample-original-41.png "class 41"
[image46]: ./writeup_images/sample-original-42.png "class 42"

[image]: ./writeup_images/sample-original-.png "class "
[image]: ./writeup_images/sample-original-.png "class "








## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

The dataset consists of 3 sets (training, validation and test) of 32x32 RGB color images:
* The size of training set is 34.799
* The size of the validation set is 4.410
* The size of test set is 12.630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

##### Sign classes distribution:
The images are not uniformly distributed among classes, in the train set they range from 180 images of the least represented classes to 2.010 of the most represented class:
![alt text][image1]

However, the distribution among classes is similar in all the training sets:
![alt text][image2]
![alt text][image3]

##### Sample images visualizations:
Samples of ten images of each class:
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]
![alt text][image22]
![alt text][image23]
![alt text][image24]
![alt text][image25]
![alt text][image26]
![alt text][image27]
![alt text][image28]
![alt text][image29]
![alt text][image30]
![alt text][image31]
![alt text][image32]
![alt text][image33]
![alt text][image34]
![alt text][image35]
![alt text][image36]
![alt text][image37]
![alt text][image38]
![alt text][image39]
![alt text][image40]
![alt text][image41]
![alt text][image42]

### Design and Test a Model Architecture

#### 1. Image data preprocessing.

On one hand the colors should not influence in the sign classification as there are no exact signs with different meanings subject to different colors, hence converting to grayscale simplifies the model and improves the efficiency.
On the other hand images differ in terms of brightnes and contrast so it seems a good idea to apply an histogram equalization.
Finally the images will be normalize to 0 to 1 values dividing by 255.

Preprocessing pipeline:


As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


