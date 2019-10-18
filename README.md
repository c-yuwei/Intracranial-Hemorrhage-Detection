# Intracranial-Hemorrhage-Detection
Development of an AI-model for detecting brain intracranial hemorrhage using data from RSNA

# Overview
The objective is to train a 2D multi-label and multi-class classifier for intracranial hemorrhage using CT dicom files provied by open competition RSNA Intracranial Hemorrgage Detection in kaggle. There are five hemorrhage subtypes: intraparenchymal, intraventricular, subarachnoid, subdural and epidural.

# Challenges
Prior run has shown that the overfitting is the major problem in this issue. The validation accuracy hits the wall when we took all the subtypes into consideration. The accuracy of classification on epidural and normal is 84.87% while it is deteriorating as 36% that all the subtype is taken into account.  In addition, the current approach is only working on single class output, but the competition may need to evaluate the predictions on 6 classes individually at single case. Therefore, it is crucial to stack a model which can handle the multi-class task as well as overcome the overfitting problem.
