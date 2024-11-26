import numpy as np 
import matplotlib.pyplot as plt

# --- Your Task --- #
# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
# --- end of task --- #

# Load an imbalanced data set 
# There are 50 positive class instances 
# There are 500 negative class instances 
data = np.loadtxt('HW4/diabetes_new.csv', delimiter=',', skiprows=1)
[n, p] = np.shape(data)

# Always use last 25% data for testing 
num_test = int(0.25 * n)
sample_test = data[n - num_test:, 0:-1]
label_test = data[n - num_test:, -1]

# Vary the percentage of data for training
num_train_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

acc_base_per = []
auc_base_per = []

acc_yours_per = []
auc_yours_per = []

for per in num_train_per: 

    # Create training data and label
    num_train = int(n * per)
    sample_train = data[0:num_train, 0:-1]
    label_train = data[0:num_train, -1]

    # --- Your Task --- #
    # Implement a baseline method that standardly trains 
    # the model using sample_train and label_train
    model = LogisticRegression(max_iter=1000)
    model.fit(sample_train, label_train)

    # Evaluate model testing accuracy and store it in "acc_base"
    label_pred = model.predict(sample_test)
    acc_base = accuracy_score(label_test, label_pred)
    acc_base_per.append(acc_base)

    # Evaluate model testing AUC score and store it in "auc_base"
    label_pred_prob = model.predict_proba(sample_test)[:, 1]
    auc_base = roc_auc_score(label_test, label_pred_prob)
    auc_base_per.append(auc_base)
    # --- end of task --- #

    # --- Your Task --- #
    # Now, implement your method 
    # Aim to improve AUC score of baseline 
    # while maintaining accuracy as much as possible 

    # Apply SMOTE to balance the training data
    # Count the number of samples in each class
    class_counts = Counter(label_train)
    minority_class_count = min(class_counts.values())
    
    # Adjust k_neighbors based on minority class count
    if minority_class_count > 1:
        k_neighbors = min(5, minority_class_count - 1)
    else:
        k_neighbors = 1  # Minimum possible value

    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    sample_train_res, label_train_res = smote.fit_resample(sample_train, label_train)

    # Train the model on the resampled data using Random Forest
    model_smote = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model_smote.fit(sample_train_res, label_train_res)

    # Evaluate model testing accuracy and store it in "acc_yours"
    label_pred_smote = model_smote.predict(sample_test)
    acc_yours = accuracy_score(label_test, label_pred_smote)
    acc_yours_per.append(acc_yours)

    # Evaluate model testing AUC score and store it in "auc_yours"
    label_pred_prob_smote = model_smote.predict_proba(sample_test)[:, 1]
    auc_yours = roc_auc_score(label_test, label_pred_prob_smote)
    auc_yours_per.append(auc_yours)
    # --- end of task --- #

plt.figure()    
plt.plot(num_train_per, acc_base_per, marker='o', label='Baseline Accuracy')
plt.plot(num_train_per, acc_yours_per, marker='s', label='Improved Method Accuracy')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification Accuracy')
plt.legend()
plt.title('Figure 4: Model Accuracy vs Training Data Size')
plt.grid(True)

plt.figure()
plt.plot(num_train_per, auc_base_per, marker='o', label='Baseline AUC Score')
plt.plot(num_train_per, auc_yours_per, marker='s', label='Improved Method AUC Score')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification AUC Score')
plt.legend()
plt.title('Figure 5: Model AUC Score vs Training Data Size')
plt.grid(True)

plt.show()
