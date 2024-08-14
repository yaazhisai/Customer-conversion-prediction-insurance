# Customer-conversion-prediction-insurance

In the insurance industry, acquiring new customers and converting leads into sales is crucial for business growth. The dataset provided contains information about a series
of marketing calls made to potential customers by an insurance company. The goal is to predict whether a customer will subscribe to an insurance policy based on various
attributes of the customer and details of the marketing interactions.

To tackle the problem of predicting whether a customer will subscribe to an insurance policy based on marketing interactions and customer attributes, we’ll need to follow a structured approach. Here’s a step-by-step guide that will help you through the process, including data preprocessing, exploratory data analysis (EDA), feature engineering, model building, and evaluation.

This Streamlit application allows users to explore various aspects of a marketing dataset and predict customer conversion based on different features. The application consists of two main sections:

## Analysis: Visualize and analyze the dataset to gain insights into customer demographics, job roles, marital status, call types, and more.


## Prediction: Input customer data and receive predictions on whether the customer will convert based on a pre-trained model.

### UNDERSTANDING THE DATA:
### Features
AGE ANALYSIS: Visualizes the distribution of customer ages and their conversion outcomes.
JOB ANALYSIS: Analyzes the impact of different job roles on customer conversion.
MARITAL ANALYSIS: Examines how marital status affects conversion rates.
CALL TYPE ANALYSIS: Analyzes the influence of call types on conversion.
NUMBER OF CALLS ANALYSIS: Visualizes the distribution of call counts and their effect on conversion.
MONTH ANALYSIS: Examines conversion rates based on the month of the call.
PREVIOUS OUTCOME ANALYSIS: Looks at how previous call outcomes impact conversion rates.
PREDICTION FORM: Allows users to input customer details and get a prediction on conversion likelihood.

## Installation:

Prerequisites
Ensure you have Python 3.7 or later installed. You will also need the following Python packages:
Streamlit
Pandas
NumPy
Seaborn
Matplotlib
Plotly
Scikit-learn
You can install these packages using pip:


##  Data Preprocessing
Handling Missing Values:Identify and handle missing values appropriately.
Encoding Categorical Variables:Convert categorical variables into numerical format using encoding techniques.

## Feature Scaling
Normalize or standardize numerical features if necessary.

## Exploratory Data Analysis (EDA)
DATA Visualization
Use visualizations to understand distributions, relationships, and potential issues in the data.

## Feature Relationships
Explore relationships between features and the target variable.
USE HEATMAP to find the correlation between features and target variables

## Splitting the Data
Split the dataset into training and testing sets.

## Model Building
Train several models and evaluate their performance.

Logistic Regression
Decision Tree
Extratreesclassifier
Randomforest
Adaboost
xgbboost
GradientBoost classifier
KNN classifier

After comparing all the models,i have selected 'ADABOOST CLASSIFIER' based on auccuracy  and classification report.

## Model Evaluation
Compare model performance using metrics like accuracy, precision, recall, and F1-score. Use metrics like ROC-AUC for classification problems.

## Hyperparameter Tuning
Optimize model performance by tuning hyperparameters using techniques like GridSearchCV or RandomizedSearchCV.

## Deployment
Once you have a trained and validated model, you can deploy it to make predictions on new data. Ensure the model is monitored and updated regularly.
