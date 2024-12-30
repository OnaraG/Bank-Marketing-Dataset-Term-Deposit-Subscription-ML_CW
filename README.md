##Bank Marketing Dataset Analysis and Prediction##
This project focuses on building predictive models to determine whether a customer will subscribe to a term deposit based on bank marketing data. The project implements two machine learning models, Random Forest and Neural Network, and evaluates their performance using key metrics. The dataset is preprocessed, balanced using SMOTE, and rigorously analyzed through Exploratory Data Analysis (EDA).

Table of Contents
Dataset Overview
Project Workflow
Models Used
Evaluation Metrics
Setup and Installation
Usage
Results
Future Enhancements
Dataset Overview
Source: Bank Marketing Dataset (bank-full.csv)
Rows: 45,211
Columns: 17
Target Variable:
deposit: Indicates whether a customer subscribed to a term deposit (yes or no).
The dataset includes both categorical (e.g., job, marital) and numerical features (e.g., age, balance).

Project Workflow
Exploratory Data Analysis (EDA):

Analyzed categorical and numerical features.
Visualized relationships between features and the target variable.
Identified outliers and correlations.
Data Preprocessing:

Renamed the target variable y to deposit.
Dropped irrelevant columns: default, pdays.
One-hot encoded categorical variables.
Handled outliers in campaign and previous.
Balanced the dataset using SMOTE to address class imbalance.
Model Training:

Implemented Random Forest Classifier and Neural Network (MLPClassifier).
Applied cross-validation for performance evaluation.
Model Evaluation:

Compared models using metrics: Accuracy, Precision, Recall, F1-Score, and ROC AUC.
Visualized results using ROC and Precision-Recall curves.
Models Used
Random Forest Classifier:

Hyperparameters: n_estimators=100, max_depth=5, max_features='sqrt', criterion='gini'.
Neural Network (MLPClassifier):

Hyperparameters:
Hidden Layers: (100, 50)
Activation: relu
Solver: adam
Learning Rate: constant
Regularization: alpha=0.001
Evaluation Metrics
Accuracy: Overall correctness of predictions.
Precision: Proportion of correctly identified positives.
Recall: Ability to identify all positives in the data.
F1-Score: Harmonic mean of Precision and Recall.
ROC AUC: Discriminatory ability of the models.
Setup and Installation
Prerequisites
Python 3.10 or higher.
Libraries:
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
Installation
Clone the repository:
bash
Copy code
git clone <your-git-repo-url>
cd bank-marketing-prediction
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Usage
Run the script:
bash
Copy code
python main.py
Outputs:
Model evaluation metrics (Accuracy, Precision, Recall, F1-Score).
Visualizations: ROC and Precision-Recall curves.
Tabular comparison of Random Forest and Neural Network results.
Results
Random Forest:
High accuracy and precision, making it suitable for balanced predictions.
Neural Network:
Better recall, effective for identifying positive cases.
Visualization:
ROC and Precision-Recall curves highlight strong discriminatory performance for both models.
Future Enhancements
Apply advanced hyperparameter optimization techniques (e.g., Optuna).
Explore other machine learning models like XGBoost or LightGBM.
Develop a real-time prediction dashboard for deployment.
License
This project is licensed under the MIT License.

Acknowledgments
The dataset used is publicly available from UCI Machine Learning Repository.
