This repo is the solution to the feature importance question by Smart Steel.

Task description
----------------
The file task_data.csv contains an example data set that has been artificially generated. 
The set consists of 400 samples where for each sample there are 10 different sensor readings available. 
The samples have been divided into two classes where the class label is either 1 or -1. 
The class labels define to what particular class a particular sample belongs.
Your task is to rank the sensors according to their importance/predictive power with respect to the class labels of the samples. 
Your solution should be a Python script or a Jupyter notebook file that generates a ranking of the sensors from the provided CSV file. 
The ranking should be in decreasing order where the first sensor is the most important one.

Additionally, please include an analysis of your method and results, with
possible topics including:

* your process of thought, i.e., how did you come to your solution?
* properties of the artificially generated data set
* strengths of your method: why does it produce a reasonable result?
* weaknesses of your method: when would the method produce inaccurate results?
* scalability of your method with respect to number of features and/or samples
* alternative methods and their respective strengths, weaknesses, scalability


Technical requirements
----------------------
Please submit *one* zip file that contains the following:
    * .txt or .csv of ranked list of sensors
    * .txt or .pdf of analysis
    * Python script (.py) without parameters and/or Jupyer notebook (.ipynb)
    * requirements.txt

Pipeline:
+ Introduction
+ Filter method
+ Wrapper method
+ Tree-based
+ ML method
+ Write OOP code

## Introduction
Why do we need to rank features?

In the ML system design, pre-processing the dataset is one of the most important step, in which, we try to grasp as many as possible insights from the dataset.
Understanding the problem statement of the pipeline, we can get our hands on the feature ranking.
Feature ranking could be seen as a part of feature selection.
The fact is that not all features are useful for the final task, and adding or removing one feature could change the accuracy of the final model. 
Choosing the correct features based on their ranks could bring some advantages:
+ Faster traing model
+ Improve accuracy
+ Save time and space
+ Reduce over fitting

Although tree-based model is the most common among all, we will go through some methods to rank features:
+ Filter method
+ Tree-based or intrinsic method
+ Additional ML method
+ Wrapper method

The main problem of with the dataset is testing the ranked features.
In order to understand the ranking models and to pick the best performed one, one final (neutral) model should try to predict the classes of sensors based only on the ranked sensors.
Therefore, once we have the ranked sensors, it is very difficult to argue that the ranking is totally correct.

## Components

## Filter method 
They the the general methods, which we expect to pick a subset of features. 
Using correlation matrix is the most basic filter method. 
We can see that, all 10 sensors show the independence to each other with small correlation values. 
On the class column, we can see the top three sensors are sensor 8, 4, and 0. 
Naively, this means the value of class column depends on 3 sensors more than other.

// add image of correlation matrix

Advantage and disadvantages

## Tree-based method
It is the most common to-go method for this problem, where we have nummerical features and categorical class.

Based on the dataset and the question, the tree model is the most suitable, for classification 10 sensors, instead of 10 classification models for each sensor.

Choosing the models, the tree models are immune to multicollinearity of the dataset. Tradiditonal way to solve the multicollinearity problem is to remove these features, other ML ways could be using dimension reduction method: PCA, LDA, and auto encoder. This problem could also be seen as feature selections, in which we will try to select the top 3 sensors in set of 10.
