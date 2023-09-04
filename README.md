# Stroke Prediction Using Ensemble Learning

## Project Status: Completed

### Introduction
This project aims to build a predictive model for stroke based on individuals' medical history and demographic information. It explores three ensemble algorithms: Gradient Boosting, Bagging (Random Forest algorithm), and Stacking Generalization. This solo final project in Pattern Recognition focuses on stroke prediction.

### Methods
- Inferential Statistics
- Machine Learning
- Data Visualization
- Predictive Modeling

### Technologies 
- Jupyter Notebook

### Libraries
- Pandas
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Plotly Express

## Project Description
Stroke is a severe medical condition, a leading cause of death in the United States. Early prediction can reduce the disease's impact on individuals and families. This project hypothesizes that ensemble learning can achieve high precision and recall in stroke prediction.

- Dataset: Obtained from Kaggle [link](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
- Dataset Size: 5110 records with 12 features. The 'id' variable was removed for lack of value.
- Insights: Stroke prevalence is higher in older patients, males, married individuals, and smokers, as well as those with hypertension, heart disease, high glucose levels, and high BMI.

### Data Preprocessing
- Missing Values: Treated missing values in 'smoking status' and 'BMI.'
- Outliers: Identified and removed outliers in 'average glucose level' and 'BMI.'
- Feature Conversion: Converted categorical features to numeric for machine learning.
- Dataset Split: Split the dataset into training (60%), validation (20%), and testing (20%) sets.
- Standardization: Standardized all features for consistent scale.
- Imbalanced Dataset: Addressed imbalance using Synthetic Minority Over-Sampling Technique (SMOTE).

### Feature Sets
Created three feature sets: original, reduced original, and clean reduced to assess the impact of data cleaning.

### Model Tuning
- Used Grid Search CV to optimize model parameters.
- For Gradient Boosting: Tuned number of estimators, maximum depth, and learning rate.
- For Random Forest: Tuned number of estimators, maximum depth, and bootstrap.
- For Stacked Generalization: Tuned parameters including number of estimators and regularization (C parameter).

### Model Evaluation
- Best Model: Gradient Boosting on Reduced Raw features.
- Validation Performance: Accuracy (91.4%), Precision (97.3%), Recall (85.2%), F1 Score (90.8%).
- Test Set Performance: Accuracy (93.5%), Precision (98.3%), Recall (88.6%), F1 Score (93.2%).

## Project Needs
- Data exploration/descriptive statistics
- Data processing/cleaning
- Statistical modeling
- Report writing
