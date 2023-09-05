# Upskilling-Week-1


## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Code](#code)
4. [Data](#data)
5. [Usage](#usage)

## Overview
Repository containing code and data for Week 1 of Trainee Upskilling at Fusemachines. I performed classification of quality of wine based on its features on the [Wine Quality dataset from Kaggle] (https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009). For that, I used Support Vector Machines, Decision Trees, Random Forest and xgBoost.

## Directory Structure

* /code - Directory containing the code for everything done in the project.
* /data - Directory containing the csv files used in the project.



## Code

The [code](code) directory contains 

* decisiontrees.ipynb - Contains the code for training & evaluating the performance of Decision Tree on our data.

* initial-exploration.ipynb - Contains initial EDA of the raw dataset.

* preprocessing.ipynb - Contains code for preprocessing the data before training models on it.

* random_forest_classifier.py - Contains code for random forest classifier.

* svm.py - Contains code for Support Vector Machine, with evaluation on train and test data.

* xg_boost.py - Contains code for training and evaluating xgBoost on the data.


## Data

* winequality-red.csv - This csv file contains the raw data downloaded from Kaggle.
* preprocessed.csv - This csv file contains the preprocessed data, exported from code/preprocessing.ipynb.


