**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup-images/class_counter.png "Train set examples per class"
[image2]: ./writeup-images/label_distribution.png "Histograms of label distribution in the different sets"
[image3]: ./writeup-images/label_distribution_2.png "Class distribution comparison between sets"
[image4]: ./writeup-images/Sample-original-0.png "class 0"
[image5]: ./writeup-images/Sample-original-1.png "class 1"
[image6]: ./writeup-images/Sample-original-2.png "class 2"
[image7]: ./writeup-images/Sample-original-3.png "class 3"
[image8]: ./writeup-images/Sample-original-4.png "class 4"
[image9]: ./writeup-images/Sample-original-5.png "class 5"
[image10]: ./writeup-images/Sample-original-6.png "class 6"
[image11]: ./writeup-images/Sample-original-7.png "class 7"
[image12]: ./writeup-images/Sample-original-8.png "class 8"
[image13]: ./writeup-images/Sample-original-9.png "class 9"
[image14]: ./writeup-images/Sample-original-10.png "class 10"
[image15]: ./writeup-images/Sample-original-11.png "class 11"
[image16]: ./writeup-images/Sample-original-12.png "class 12"
[image17]: ./writeup-images/Sample-original-13.png "class 13"
[image18]: ./writeup-images/Sample-original-14.png "class 14"
[image19]: ./writeup-images/Sample-original-15.png "class 15"
[image20]: ./writeup-images/Sample-original-16.png "class 16"
[image21]: ./writeup-images/Sample-original-17.png "class 17"
[image22]: ./writeup-images/Sample-original-18.png "class 18"
[image23]: ./writeup-images/Sample-original-19.png "class 19"
[image24]: ./writeup-images/Sample-original-20.png "class 20"
[image25]: ./writeup-images/Sample-original-21.png "class 21"
[image26]: ./writeup-images/Sample-original-22.png "class 22"
[image27]: ./writeup-images/Sample-original-23.png "class 23"
[image28]: ./writeup-images/Sample-original-24.png "class 24"
[image29]: ./writeup-images/Sample-original-25.png "class 25"
[image30]: ./writeup-images/Sample-original-26.png "class 26"
[image31]: ./writeup-images/Sample-original-27.png "class 27"
[image32]: ./writeup-images/Sample-original-28.png "class 28"
[image33]: ./writeup-images/Sample-original-29.png "class 29"
[image34]: ./writeup-images/Sample-original-30.png "class 30"
[image35]: ./writeup-images/Sample-original-31.png "class 31"
[image36]: ./writeup-images/Sample-original-32.png "class 32"
[image37]: ./writeup-images/Sample-original-33.png "class 33"
[image38]: ./writeup-images/Sample-original-34.png "class 34"
[image39]: ./writeup-images/Sample-original-35.png "class 35"
[image40]: ./writeup-images/Sample-original-36.png "class 36"
[image41]: ./writeup-images/Sample-original-37.png "class 37"
[image42]: ./writeup-images/Sample-original-38.png "class 38"
[image43]: ./writeup-images/Sample-original-39.png "class 39"
[image44]: ./writeup-images/Sample-original-40.png "class 40"
[image45]: ./writeup-images/Sample-original-41.png "class 41"
[image46]: ./writeup-images/Sample-original-42.png "class 42"
[image47]: ./writeup-images/preprocessing-pipeline.png "Preprocessing pipeline"
[image48]: ./writeup-images/Sample-converted-0.png "Preprocessing sample class 0"
[image49]: ./writeup-images/Sample-converted-2.png "Preprocessing sample class 2"
[image50]: ./writeup-images/Sample-converted-9.png "Preprocessing sample class 9"
[image51]: ./writeup-images/Sample-converted-13.png "Preprocessing sample class 13"
[image52]: ./writeup-images/Sample-converted-14.png "Preprocessing sample class 14"
[image53]: ./writeup-images/Sample-converted-17.png "Preprocessing sample class 17"
[image54]: ./writeup-images/Sample-converted-27.png "Preprocessing sample class 27"
[image55]: ./writeup-images/Sample-converted-35.png "Preprocessing sample class 35"
[image56]: ./writeup-images/Sample-converted-38.png "Preprocessing sample class 38"
[image57]: ./writeup-images/Sample-converted-39.png "Preprocessing sample class 39"
[image58]: ./writeup-images/augm-transformation.png "Augmentation transformation"
[image59]: ./writeup-images/loss_acc_graph.png "Loss and accuracy graphs"
[image60]: ./writeup-images/loss_acc_augm_graph.png "Loss and accuracy graphs for augmented data"
[image61]: ./writeup-images/new_images.png "New images from the web"
[image62]: ./writeup-images/new_conv_images.png "New images after preprocessig"
[image63]: ./writeup-images/new_images_predictions.png "New images predictions"
[image64]: ./writeup-images/new_images_top_predictions.png "New images top predictions"


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

This images illustrates the preprocessing pipeline:

![alt text][image47]

These samples illustrate the preprocessing results ten random classes:

![alt text][image48]
![alt text][image49]
![alt text][image50]
![alt text][image51]
![alt text][image52]
![alt text][image53]
![alt text][image54]
![alt text][image55]
![alt text][image56]
![alt text][image57]

Finally the images will be normalize to 0 to 1 values dividing by 255.

##### Data augmentation:
First training was done with the original data set reaching validation acuracy close to 96%. To improve the accuracy data augmentation was tested and validation accuracy improved up to 98%.

The data augmentation was done using [imgaug library](https://github.com/aleju/imgaug/blob/master/README.md). The following techniques were used:
* Scale: random scale factor from 80% to 120% in both x and y dimensions
* Rotate: random rotation angle from -30 to 30 degrees
* Shear: shear transformation using random angle from -20 to 20 degrees

The following images illustrates the transformation:

![alt text][image58]

These techniques were applied equally to all training images and added to the training dataset which result in a 139.196 images training set.

Although less frequent classes could have been augmentated in a greater factor to get a more homogeneuos distribution of classes in the training, as both validation and test set suffer from similar bias among classes it was decided to keep the training classes bias.


#### 2. Final model architecture:

Final model was based in LeNet architecture adapted to different size inputs and greater number of classes. Addionaly dropout and L2 regularization were used to contol overfitting. The final model consisted of the following layers:

| Layer					| Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 32x32x1 Grayscale image  						| 
| Convolution 1 5x5		| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 14x14x6					|
| Convolution 2 5x5		| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 5x5x6					|
| Flatten				| outputs 400									|
| Dropout				|												|
| Fully connected 1 	| outputs 120									|
| RELU					|												|
| Dropout				|												|
| Fully connected 2 	| outputs 84									|
| RELU					|												|
| Dropout				|												|
| Fully connected 3 	| outputs 43									|
| Softmax				|												|

#### 3. Model training.
Adam optimizer was used to train the model and the batch size was fix to 128. The following paramaters were tuned to improve validation accuracy:
* Epochs: from 10 to 50 were tested
* Learning rate: from 0.001 to 0.01 were tried
* Learning rate exponential decay factor: values close to 0.9 were found to help stabilizing the training as the optimizer aproximates the optimum
* Dropout keep probability: values from 0.65 to 0.9 were tested
* L2 beta regularization parameter: values in the order of 0.0001 were used

#### 4. Describe the approach.

##### Training in the original data set:
Firts training was done over the original training reaching accuracies over 96% in the validation set after tuning up the training parameters. The following values were found to bring optimal results:
* Epochs: 25
* Learning rate: 0.005
* Learning rate exponential decay factor: 0.88
* Dropout keep probability: 0.65
* L2 beta regularization parameter: 0.0001

The final model trained in the results were:
* training set accuracy of 99.79%
* validation set accuracy of 96.80%
* test set accuracy of 95.42%

The following graph shows the loss and accuracy learning curves:

![alt text][image59]

The following aproach was used to tun the parametes:
1. Try to get the best possible training accuracy using tuning epochs and learning rate parameters.
2. Once adjusted these parameters try to reduce overfitting using dropout and L2 regularization using keep probablity and beta reguralization parameters.

The LeNet architecture was chosen as base model and adjusted to return 43 classes instead of 10 classes from the MNIST data set. LeNet architecture has rechead great results in image recognition in the MNIST data set and hence represented a good start.
The original architecture was improved with dropout and L2 regularization in the fully connected layers to control overffiting
These architecture got consistent accuracy results close to 97% accuracy in the validation set without using complex techniques such as data aumentation, early stopping, pretraining... and in only few minutes using an 8 eight years old Mac Book with 16 GB of RAM and an NVIDIA GeForce 320M 256 MB. 
 
##### Training in the augmented data set:
To improved results the augmented training set was used with the aim to reach 98% accuracy. Computational cost increased as to make training un-practical in desktop laptop so an amazon G2.2 large instance was used.

Using a greater training data set reduces risks of overfitting and hence lower regularization parameters were used. However, although greater accuracy was reached we could only grasp the expected 98%  but not be consistently over. Only a rough 1% higher than the model trained with the original data set and at a much higher computational cost.

Probably a more complex model with one more convolution layer and probably another fully connected layer should be used. It could also be of use a more complex image augmentation model to increase the least represented classes.

The following values were found to bring optimal results in the augmented data set:
* Epochs: 20
* Learning rate: 0.0025
* Learning rate exponential decay factor: 0.85
* Dropout keep probability: 0.75
* L2 beta regularization parameter: 0.0

The final model trained in the results were:
* training set accuracy of 99.28%
* validation set accuracy of 97.73%
* test set accuracy of 96.25%

The following graph shows the loss and accuracy learning curves:

![alt text][image60]


### Test a Model on New Images

#### New German traffic signs.
New 10 images from the web:

![alt text][image61]

Images afer preprocessing:

![alt text][image62]

The 4th image: "Beware of ice/snow" is probably the most difficult to predict as it is the least quality image of the set and after the preprocessing it is difficult to predict manually.

#### Model's predictions

![alt text][image63]

The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100% which compares favorably to the accuracy on the test set of probably because clear images has been chosen.

#### Top 5 prediction for each image:

![alt text][image64]


