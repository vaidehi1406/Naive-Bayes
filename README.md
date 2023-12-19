# Naive-Bayes

The code implements a Gaussian Naive Bayes classifier on a dataset of bank customer attributes, achieving an 89.1% accuracy in predicting whether customers will take a personal loan, with a detailed analysis including a confusion matrix and classification report.

---

Naive Bayes is a probabilistic machine learning algorithm that is based on Bayes' theorem. It is particularly suited for classification tasks and is known for its simplicity and efficiency. The "naive" part of Naive Bayes comes from the assumption that features used to describe an observation are conditionally independent, given the class label. Despite this simplifying assumption, Naive Bayes often performs well in practice, especially in text classification and spam filtering.

---

Here's a breakdown of the code you provided:

---

**Data Preparation:**

**1.Loading Data:**

* The code begins by importing necessary libraries and loading a dataset named "Bank_Personal_Loan_Modelling.csv" into a Pandas DataFrame.

**2.Data Exploration:**

* The info() method is used to get information about the dataset, and head() is used to display the first few rows.
  
**3.Feature Encoding:**

* Label encoding is applied to categorical variables using LabelEncoder for each feature in the dataset.

**4.Creating X and y:**

* The dataset is divided into features (X) and the target variable (y), where "Personal Loan" is the target.

---
  
**Model Building:**

**1.Splitting Data:**

* The dataset is split into training and testing sets using train_test_split().

**2.Building the Naive Bayes Model:**

* A Gaussian Naive Bayes classifier is instantiated using GaussianNB().

**3.Training the Model:**

* The model is trained on the training set using the fit() method.

---

**Model Evaluation:**

**1.Model Evaluation on Training Set:**

* The model's accuracy on the training set is calculated using score().

**2.Model Evaluation on Testing Set:**

* The model's accuracy on the testing set is calculated using score().

**3.Making Predictions:**

* The model is used to make predictions on the testing set.

---

**Evaluating Model Accuracy:**

* The accuracy, classification report, and confusion matrix are printed and displayed.

**Visualization:**

* A heatmap of the confusion matrix is plotted using Seaborn.

**Displaying Predictions:**

* A DataFrame is created to display actual vs. predicted values on the testing set.

**Summary:**

* The Naive Bayes model achieves an accuracy of 89.1% on the testing set.
  
* The confusion matrix and classification report provide additional insights into the model's performance.

---

**Result Analysis**

**1. Model Accuracy:**

The model achieved an accuracy of approximately 89.1% on the testing set.

**2. Classification Report:**

Precision, recall, and F1-score are reported for each class (0 and 1) and the overall accuracy.

Class 0 (No Personal Loan):
* Precision: 0.95
* Recall: 0.92
* F1-score: 0.94
  
Class 1 (Personal Loan):
* Precision: 0.49
* Recall: 0.63
* F1-score: 0.55
  
The weighted average F1-score is 0.90.

**3. Confusion Matrix:**

The confusion matrix is as follows:

[[825  70]
 [ 39  66]]
 
Interpretation:
* True Positive (TP): 66
* False Positive (FP): 70
* True Negative (TN): 825
* False Negative (FN): 39
  
**4. Visualization:**

A heatmap of the confusion matrix is plotted for better visualization.

**5. Predictions:**

A DataFrame is created to display actual vs. predicted values on the testing set.

**Summary:**

* The model performs well in predicting Class 0 (No Personal Loan) with high precision, recall, and F1-score.
* However, the model has lower performance in predicting Class 1 (Personal Loan), as indicated by lower precision, recall, and F1-score.
* The confusion matrix provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.

**Recommendations:**

* The model's overall accuracy is good, but the performance disparity between classes suggests further investigation.
* Consider addressing the class imbalance if present.
* Fine-tune the model parameters or explore other classification algorithms for potential improvement.
* Feature engineering or additional data exploration may enhance model performance.
* This analysis provides insights into the strengths and weaknesses of the Naive Bayes model for the specific task of predicting whether a person will take a personal loan.
