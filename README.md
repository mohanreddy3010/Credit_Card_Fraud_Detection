## Credit_Card_Fraud_Detection

#Overview:

The Credit Card Fraud Detection project aims to detect fraudulent transactions from a dataset of credit card transactions using machine learning algorithms. The dataset consists of anonymized features for each transaction, including transaction time, amount, and several anonymized features (V1 to V28) that represent different characteristics of each transaction. The goal is to predict whether a given transaction is legitimate or fraudulent.

#Data Preprocessing:

The dataset contains both continuous and categorical features. To prepare the data for machine learning models, StandardScaler was applied to normalize the ‘Amount’ and ‘Time’ features. Normalization is crucial as these features have varying scales, which can adversely affect model performance.

The target variable is the ‘Class’ column, where ‘0’ represents non-fraudulent transactions and ‘1’ represents fraudulent ones. To ensure the model can handle the significant imbalance between fraud and non-fraud transactions, stratified splitting was used during the train-test data split to maintain the same proportion of classes in both training and testing sets.

#Model Development:

Several machine learning algorithms were employed to detect fraudulent transactions:
	•	Logistic Regression: A simple yet effective classification model used to establish a baseline for the project. We used class_weight='balanced' to handle the class imbalance, giving more importance to the minority class (fraudulent transactions).
	•	XGBoost: An advanced gradient boosting model known for its high performance in classification tasks. We leveraged the XGBClassifier with the objective='binary:logistic' setting to predict the probability of fraud, optimizing the model’s accuracy and recall.

Both models were trained on the training set and tested on the holdout test set.

#Model Evaluation:

Model performance was evaluated using multiple metrics:
	•	Confusion Matrix: Used to assess the number of true positives, false positives, true negatives, and false negatives.
	•	Classification Report: Provides detailed metrics such as precision, recall, and F1-score to evaluate model performance for each class (fraud and non-fraud).
	•	ROC-AUC Score: The area under the ROC curve was calculated to evaluate the model’s ability to distinguish between fraudulent and non-fraudulent transactions.
	•	Precision-Recall Curve: This curve was used to analyze the trade-off between precision and recall, particularly important given the imbalance in class distribution.

#Exploratory Data Analysis (EDA):

Before building the models, Exploratory Data Analysis (EDA) was performed to understand the structure of the data:
	•	Class Distribution: The dataset has a highly imbalanced class distribution, with fraudulent transactions representing only 0.17% of the total data. We visualized this imbalance using seaborn’s countplot.
	•	Transaction Amount Distribution: We plotted the distribution of transaction amounts for both fraudulent and non-fraudulent transactions using seaborn’s kdeplot to visualize how the amounts vary between the two classes.
	•	Feature Correlation: A correlation heatmap was generated to identify potential relationships between the various features, helping to guide feature engineering.

#Results:

The models achieved high accuracy, with XGBoost performing better than Logistic Regression in terms of recall and ROC-AUC score. Despite the low number of fraudulent transactions, XGBoost was able to detect a significant proportion of fraud cases with a precision of 0.92 and recall of 0.81.

Key Metrics for XGBoost:
	•	Precision: 0.92
	•	Recall: 0.81
	•	F1-score: 0.86
	•	ROC-AUC Score: 0.9743

#Conclusion:

This project successfully demonstrates the application of machine learning techniques, particularly Logistic Regression and XGBoost, to detect fraudulent credit card transactions. The final model provides a solid foundation for real-time fraud detection systems and can be further optimized for production use by implementing additional techniques such as ensemble learning, feature engineering, or deploying the model in a real-time pipeline.

#Dataset Details:

This dataset comes from a real-world collection of credit card transactions and is available on Kaggle. The data includes 284,807 transactions, each with 30 anonymized features, along with the ‘Time’, ‘Amount’, and ‘Class’ columns (where Class 0 represents non-fraudulent and Class 1 represents fraudulent transactions).
	•	Dataset Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data 
	•	Size: 284,807 samples, 31 features
	•	Target Variable: ‘Class’ (0: Non-Fraud, 1: Fraud)

The dataset provides a real-world scenario of financial fraud detection, making it ideal for practicing machine learning, especially in the context of imbalanced datasets.

This comprehensive README includes all the necessary information for GitHub, combining the project overview, dataset details, model development, evaluation metrics, and results.
