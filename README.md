# Face-Recognition
<br>

## Description:
In this project I have written 3 python scripts.
First script collects face data using OpenCV that extracts image features.
Second script identifies faces based on recorded data. Here, I have implemented my own K-nearest-neighbor algorithm to predict or identify faces present in front of webcam.
Third script is for validation and evaluation of face data.

Note : Your Laptop/Desktop must have webcam in order to record and identify faces.

<br>

## Validation and evaluation :
For validation and evaluation purpose, I have done following steps :
1. Split the data into a training and testing set using train_test_split.
2. Apply PCA that transforms 7500 features into 400.
3. Apply feature selection due to high dimensional input.
4. Train a SVM classification model
5. Calculate accuracy on test set.

<br>

## Output :
<img src = "https://github.com/codeboy47/Face-Recognition/blob/master/Images/output.png" />
 
## Result : 
The accuracy for SVM comes out to be 98.89% with a f1 score of 0.99

<img src = "https://github.com/codeboy47/Face-Recognition/blob/master/Images/accuracy.png" />

<br>

## Applications :
1. It can be used as a attendence software that recognizes faces of the employees working in a company.
2. Instead of using passcodes, mobile phones will be accessed via owners’ facial features.
