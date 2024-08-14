# Customer-conversion-prediction-insurance

In the insurance industry, acquiring new customers and converting leads into sales is crucial for business growth. The dataset provided contains information about a series
of marketing calls made to potential customers by an insurance company. The goal is to predict whether a customer will subscribe to an insurance policy based on various
attributes of the customer and details of the marketing interactions.

To tackle the problem of predicting whether a customer will subscribe to an insurance policy based on marketing interactions and customer attributes, we’ll need to follow a structured approach. Here’s a step-by-step guide that will help you through the process, including data preprocessing, exploratory data analysis (EDA), feature engineering, model building, and evaluation.

## UNDERSTANDING THE DATA:

1.age: Age of the customer.
2. job: Type of job the customer holds.
3. marital: Marital status of the customer.
4. education_qual: Educational qualification of the customer.
5. call_type: Type of marketing call.
6. day: Day of the month when the call was made.
7. mon: Month when the call was made.
8. dur: Duration of the call in seconds.
9. num_calls: Number of calls made to the customer before this interaction.
10.prev_outcome: Outcome of the previous marketing campaign.
11. y: Whether the customer subscribed to the insurance policy (target variable).

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
