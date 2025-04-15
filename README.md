# ML Diabetes Readmission Predictor

A machine learning project that analyzes 10 years of patient data from over 130 U.S. hospitals to predict the likelihood of early diabetes-related hospital readmissions.


---

## How It's Made:

**Tech used:** Python, Pandas, Scikit-learn, Imbalanced-learn (SMOTE-ENN), XGBoost, Neural Networks (MLP), Matplotlib, Seaborn, Jupyter Notebooks

The project leverages a large clinical dataset to develop a robust classifier for predicting early readmissions of diabetic patients. The pipeline includes:

### Data Preprocessing

- Loaded 100K+ patient records from CSV files
- Cleaned missing values and standardized categories (e.g. medications, diagnosis codes)
- Merged multi-source hospital data into a unified feature matrix

### Feature Engineering

- Applied encoding techniques for categorical variables
- Evaluated feature importance using mutual information and tree-based models
- Selected top-performing features using variance thresholding and recursive selection

### Class Imbalance Handling

- Applied **SMOTE-ENN** (Synthetic Minority Oversampling with Edited Nearest Neighbors)
- Balanced the dataset to improve minority class prediction performance

###  Models Trained

- **Logistic Regression**, **Support Vector Machines**, **Random Forests**
- **XGBoost** for ensemble modeling
- **Multilayer Perceptron (MLP)** for deeper learning

### Evaluation

- Used **F1-score**, **Precision/Recall**, **Confusion Matrix**, and **AUC**
- Visualized model comparison through bar plots and confusion matrices

---

## Optimizations

- SMOTE-ENN improved minority class recall by over 15%
- Tuning hyperparameters with `GridSearchCV` for SVM and XGBoost
- Compared models across metrics to choose the best trade-off between precision and recall
- Focused on clinical relevance: avoiding false negatives for patient safety

---

##  Lessons Learned

- Hands-on experience working with **real-world imbalanced healthcare datasets**
- Learned the tradeoffs in **oversampling techniques** like SMOTE vs. SMOTE-ENN
- Understood the importance of **model explainability** in medical ML
- Gained insight into **feature selection's impact** on performance and training time
