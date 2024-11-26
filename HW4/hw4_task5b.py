import numpy as np
import matplotlib.pyplot as plt

# --- Import Necessary Libraries --- #
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import Counter
# --- End of Imports --- #

# Load the imbalanced dataset
data = np.loadtxt('HW4/diabetes_new.csv', delimiter=',', skiprows=1)
n, p = data.shape

# Always use the last 25% of data for testing
num_test = int(0.25 * n)
sample_test = data[n - num_test:, :-1]
label_test = data[n - num_test:, -1]

# Fixed percentage of data for training (e.g., 60%)
per = 0.6
num_train = int(n * per)
sample_train = data[0:num_train, :-1]
label_train = data[0:num_train, -1]

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

# Calculate scale_pos_weight for XGBoost
counter = Counter(label_train_res)
scale_pos_weight = counter[0] / counter[1]

# Range of max_depth values to test
max_depth_values = [3, 5, 7, 9, 11]

# Store AUC scores for each max_depth
auc_scores = []

for max_depth in max_depth_values:
    # Train the XGBoost model with the current max_depth
    model = XGBClassifier(
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(sample_train_res, label_train_res)

    # Predict probabilities on the test set
    label_pred_prob = model.predict_proba(sample_test)[:, 1]

    # Compute AUC score and store it
    auc = roc_auc_score(label_test, label_pred_prob)
    auc_scores.append(auc)

# Plotting Figure 6: AUC Score vs max_depth
plt.figure()
plt.plot(max_depth_values, auc_scores, marker='o')
plt.xlabel('max_depth')
plt.ylabel('AUC Score')
plt.title('Figure 6: Impact of max_depth on Model AUC Score')
plt.grid(True)
plt.show()
