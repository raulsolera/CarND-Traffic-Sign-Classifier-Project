## Project: Build a Traffic Sign Recognition Program

Overview
---
The project objective is to build a deep learning model to recognize traffic signs and test it using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) which consists of 39,209 32×32 px color images to use for training, and 12,630 images that will be used for testing.

The model is build using Tensorflow library and following the LeNet architecture with some minor changes. Other libraries used are: pandas, numpy, matplotlib, cv2 (for image processing), sklearn (for data shuffling), pickle (used for saving and uploading data), skimage (for image intensity adjustment) and imgaug (used for image augmentation).

The following files and folders are included in this repository:
- new-examples: folder that contains some new examples found in the web to test the model.
- writeup-images folder that contains images for the writeup archive.
- Traffic_Sign_Classifier_final.ipynb: ipython notebook
- Traffic_Sign_Classifier_final.html: html version of the ipython notebook
- README.md: this file that you are reading right now
- signnames.csv: list of 43 classes and names of german traffic signs
- writeup.md: the file describing the project pipeline.

To run the project (apart from installing the requiered libraries) two folder should be created in the working directory:
- saved-model: folder to save the trained model
- traffic-signs-data: folder that contains the original data and the converted data

And the [german traffic signs](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip) data should be downloaded in 'traffic-signs-data' folder.