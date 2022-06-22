# Stroke Prediction Using Ensemble Learning

#### -- Project Status: [ Completed]

## Project Intro/Objective
The goal of this project is to build a model that can predict stroke in people based on people’s medical history, and demographic information.  In this study, I implemented three ensemble Algorithms: Gradient Boosting, Bagging (Random Forest algorithm), and Stacking generalization(Gradient Boosting and Random Forest as base models, and Logistic Regression as metamodel). This is a final project for my Pattern Recognition class


### Methods Used
* Inferential Statistics
* Machine Learning
* Data Visualization
* Predictive Modeling


### Technologies 
* Jupyter notbook

### Libraries
* Pandas
* scikit-learn
* Numpy
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy
* Ploty.express


## Project Description

A stroke is a serious medical condition that can lead to death or a long-term disability. In the United States of America, stroke is considered a leading cause of death. Predicting the disease early can help in mitigating the burden of the disease on the community in general and on the family of the patients in particular. 
I hypothesize that I can predict stroke, with high  precision, and recall based on people’s medical history, and demographic information using ensemble learning. 

Dataset was retrieved from Kaggle (https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). The dataset consists of 12 features and 5110 records. The variables consist of the individuals’ medical history and some demographic information. The id variable was eliminated later from the dataset. This variable holds no valuable information, and it can degrade the performance of our model. 

From exploring the dataset, I found that the prevalence of stroke is higher in old patients, in males compared to females, among married people in comparison to other marriage categories, and in smokers compared to non-smokers. Stroke occurrence is also significantly higher in patients who have hypertension, heart disease, high glucose level, and high body mass index (BMI). 

The dataset contains missing values. There are 1544 missing values in the smoking status, and 201 missing values in BMI. I checked if the variable smoking status and BMI were missing at random or not; both are missing at random, so no further treatment is needed. Missing values in the variable smoking status were entered as “Unknown”. Pandas’ library in python does not consider unknown as a missing value; only treat “NaN”, “None”, “NaN”, or “NaT” as missing values for numeric array, object array, and DateTime respectively. To fill in the missing values for smoking status, I had to first replace “Unknown” with “NAN”. Then, I filled in the missing values for the smoking status with the most common value 
Smoking status is a categorical variable, so I used mode to obtain the most common value. For, the variable BMI, which is a numerical variable, I filled in missing values with the mean. 

I checked the numerical variables in the dataset for Outliers. Outliers can affect the performance of our model. In this study, I assumed that our dataset follows a normal distribution; so, an observation that has a standard score (z score) of more than 3 is an outlier. 
The two variables: average glucose level and BMI contain outliers. Outliers were removed by clipping them at the 99 percentiles. To inspect if removing outliers would improve the performance of our model, I saved the average glucose level with no outliers as a new variable, called “avg-glucose-level- cap,” and left the old variable that contains the outliers as it is . Same procedure applied to the BMI variable, the newly created variable saved ad “bmi clean capped.” Next, I converted categorical features to numeric as many machine learning algorithms work better with numbers. 

I split the dataset into training set 60%, validation set 20%, and testing set 20%. The training set is used for training the proposed algorithms and generating models, the validation set is used to evaluate the trained models to choose the most accurate model. The best performance model will be tested using the test set. After the dataset’s partitioning, I standardized all features. Machine learning performs better on features that have the same scale. 

The dataset is imbalanced. There are 4861 records for non- stroke patients, and only 249 for stroke patients. This can be problematic as it can affect the performance of the model because it might have a bias toward the majority class. To balance the dataset, Synthetic Minority Over- Sampling Technique (SMOTE) was implemented. SMOTE uses synthetic examples to oversample the minority class.

I created a list of features that I would like to use. I created three sets of features which are original, reduced original, and clean reduced. The goal is to examine if the cleaning that I implemented on the original dataset will improve the performance of my models or not. 

I utilized Grid Search CV to tune my parameters. For Gradient Boosting, I tuned the number of estimators, maximum depth, and learning rate. For Random Forest, I tuned the number of estimators, maximum depth, and bootstrap. For Stacked Generalization,  I tuned the number of estimators for Gradient Boosting, the number of estimators for Random Forest, to pass through or not (True: model fits on the output of the base model and training data, False: fits on the output of the base models only), and the amount of regularization (C parameter) for Logistic Regression  

I aim to predict stroke; hence the best model should have the lowest false negative (highest recall), and highest F1- score. When I evaluated my models on the validation set, Gradient Boosting on Reduced Raw features achieved the best results. It achieved 91.4 % Accuracy, 97.3 % Precision, 85.2 % Recall , and 90.8 % F1 Score. This model then evaluated on test set and it yielded  93.5 % Accuracy,  98.3 % Precision, 88.6 % Recall , and  % 93.2 F1 Score.


## Needs of this project

- data exploration/descriptive statistics
- data processing/cleaning
- statistical modeling
- writeup/reporting
