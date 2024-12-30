# **Bank Marketing Dataset Analysis and Prediction**

This project focuses on building predictive models to determine whether a customer will subscribe to a term deposit based on bank marketing data. The project implements two machine learning models, **Random Forest** and **Neural Network**, and evaluates their performance using key metrics. The dataset is preprocessed, balanced using SMOTE, and rigorously analyzed through Exploratory Data Analysis (EDA).

---

## **Table of Contents**
1. [Dataset Overview](#dataset-overview)
2. [Project Workflow](#project-workflow)
3. [Models Used](#models-used)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Setup and Installation](#setup-and-installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Future Enhancements](#future-enhancements)

---

## **Dataset Overview**
- **Source**: Bank Marketing Dataset (`bank-full.csv`)
- **Rows**: 45,211  
- **Columns**: 17  
- **Target Variable**:  
  - `deposit`: Indicates whether a customer subscribed to a term deposit (`yes` or `no`).  

The dataset includes both categorical (e.g., `job`, `marital`) and numerical features (e.g., `age`, `balance`).

---

## **Project Workflow**
1. **Exploratory Data Analysis (EDA)**:
   - Analyzed categorical and numerical features.
   - Visualized relationships between features and the target variable.
   - Identified outliers and correlations.

2. **Data Preprocessing**:
   - Renamed the target variable `y` to `deposit`.
   - Dropped irrelevant columns: `default`, `pdays`.
   - One-hot encoded categorical variables.
   - Handled outliers in `campaign` and `previous`.
   - Balanced the dataset using SMOTE to address class imbalance.

3. **Model Training**:
   - Implemented **Random Forest Classifier** and **Neural Network (MLPClassifier)**.
   - Applied cross-validation for performance evaluation.

4. **Model Evaluation**:
   - Compared models using metrics: Accuracy, Precision, Recall, F1-Score, and ROC AUC.
   - Visualized results using ROC and Precision-Recall curves.

---

## **Models Used**
1. **Random Forest Classifier**:
   - Hyperparameters: `n_estimators=100`, `max_depth=5`, `max_features='sqrt'`, `criterion='gini'`.

2. **Neural Network (MLPClassifier)**:
   - Hyperparameters:  
     - Hidden Layers: `(100, 50)`  
     - Activation: `relu`  
     - Solver: `adam`  
     - Learning Rate: `constant`  
     - Regularization: `alpha=0.001`

---

## **Evaluation Metrics**
- **Accuracy**: Overall correctness of predictions.
- **Precision**: Proportion of correctly identified positives.
- **Recall**: Ability to identify all positives in the data.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **ROC AUC**: Discriminatory ability of the models.

---

## **Setup and Installation**

### **Prerequisites**
1. Python 3.10 or higher.
2. Libraries:
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `seaborn`
   - `scikit-learn`
   - `imbalanced-learn`
     
### **Installation**
1. Clone the repository:
   ```bash
   git clone <your-git-repo-url>
   cd bank-marketing-prediction
2. Install the required dependencies:
   pip install -r requirements.txt

---

## **Usage**
1. **Run the script**:
   ```bash
   python main.py
   
## **Outputs**
- **Model Evaluation Metrics**:
  - Accuracy, Precision, Recall, and F1-Score for both models.
- **Visualizations**:
  - ROC and Precision-Recall curves.
- **Tabular Comparison**:
  - Side-by-side comparison of Random Forest and Neural Network results.

---

## **Results**
- **Random Forest**:
  - High accuracy and precision, making it suitable for balanced predictions.
- **Neural Network**:
  - Better recall, effective for identifying positive cases.
- **Visualization**:
  - ROC and Precision-Recall curves highlight strong discriminatory performance for both models.

---

## **Future Enhancements**
- Apply advanced hyperparameter optimization techniques (e.g., Optuna).
- Explore other machine learning models like XGBoost or LightGBM.
- Develop a real-time prediction dashboard for deployment.

---

## **License**
This project is licensed under the MIT License.

---

## **Acknowledgments**
The dataset used is publicly available from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

  
