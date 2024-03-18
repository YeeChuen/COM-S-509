# Comparative analysis of various machine learning method

## Summary
This repo implements various machine-learning models that we have addressed in course COM S 573. These include but are not limited to support-vector machines, and logistic regression. 
For each of the models we implement, we also incorporate various optimization techniques to increase the performance of corresponding models. Some of the techniques we will show include bagging, boosting, feature selection via entropy/information gain, feature space projection, hyperparameter tuning, etc. 
Furthermore, we also make hypotheses on the correlation factor between the model, optimization techniques used and the quality of the data. This repo aims to provide insight into the efficacy of different optimization techniques on various models and show their impact on performance gain. 
The efficacy of these models is assessed through their performance on a diverse range of datasets, encompassing predictions related to breast cancer, Alzheimer’s disease handwriting, spam email classification, and water potability determination.

## scripts
All Logistic Regression's script are in the folder `LR_model`
use
`python main.py`
for all testing using LR model on all 4 dataset

All Support Vector Machine's script are in the folder `SVM`
use
It uses Jupyter Notebook, hence result will already be shown.
Swap the techniques by comment and uncomment in the main.

## Data Visualization
in the main directory
`python plot.py` will plot the result of the performance from these models against the dataset

All result plot can be found in directory `result`

