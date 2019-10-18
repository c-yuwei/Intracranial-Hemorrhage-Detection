# Intracranial-Hemorrhage-Detection
Development of an AI-model for detecting brain intracranial hemorrhage using data from RSNA

# Overview
The objective is to train a 2D multi-label and multi-class classifier for intracranial hemorrhage using CT dicom files provied by open competition RSNA Intracranial Hemorrgage Detection in kaggle. There are five hemorrhage subtypes: intraparenchymal, intraventricular, subarachnoid, subdural and epidural.

# Challenges
Prior run has shown that the overfitting is the major problem in this issue. The validation accuracy hits the wall when we took all the subtypes into consideration. The accuracy of classification on epidural and normal is 75.87% while it is deteriorating as 16% that all the subtype is taken into account.  In addition, the current approach is only working on single class output, but the competition may need to evaluate the predictions on 6 classes individually at single case. Therefore, it is crucial to stack a model which can handle the multi-label task as well as overcome the overfitting problem.

# Current Results
2019/10/18 - L2 regularization cnn model
The CNN.m file creates a model implemented with global and local L2 regularization fators which can yield the 84.87% of accuracy on the classification between epidural and normal.
For the sake of comparison, I adapted this model to perform 6-classes classified tasks, and the confusion matrix is as follows.
We can see that, the class 1 and 2 is well categroized but the class 3 /5, class 1/5 are the most confused pair respectively.

![alt text](https://github.com/maverickyuwei/Intracranial-Hemorrhage-Detection/blob/master/6%20classess%20confusion%20matrix.jpg)

