import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
# ......
# --- end of task --- #

# load a data set for regression
# in array "data", each row represents a community 
# each column represents an attribute of community 
# last column is the continuous label of crime rate in the community
data = np.loadtxt('HW4/crimerate.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

# --- Your Task --- #
# now, pick the percentage of data used for training 
# remember we should be able to observe overfitting with this pick 
# note: maximum percentage is 0.75 
per = 0.75
num_train = int(n*per)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
# --- end of task --- #


# --- Your Task --- #
# We will use a regression model called Ridge. 
# This model has a hyper-parameter alpha. Larger alpha means simpler model. 
# Pick 5 candidate values for alpha (in ascending order)
# Remember we should aim to observe both overfitting and underfitting from these values 
# Suggestion: the first value should be very small and the last should be large 
alpha_vec = [0.001, 0.01, 0.1, 1, 10]
# --- end of task --- #

er_train_alpha = []
er_test_alpha = []
er_valid_alpha = []

# Implement k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for alpha in alpha_vec: 

    # pick ridge model, set its hyperparameter 
    model = Ridge(alpha = alpha)
    
    # --- Your Task --- #
    # now implement k-fold cross validation 
    # on the training set (which means splitting 
    # training set into k-folds) to get the 
    # validation error for each candidate alpha value 
    # store it in "er_valid"
    validation_error = 0
    # Perform k-fold cross-validation
    for train_idx, valid_idx in kf.split(sample_train):
        # Split the training data into k-folds
        X_train_k, X_valid_k = sample_train[train_idx], sample_train[valid_idx]
        y_train_k, y_valid_k = label_train[train_idx], label_train[valid_idx]
        
        # Train the model on the k-th fold
        model.fit(X_train_k, y_train_k)
        
        # Predict on the validation fold and calculate the validation error (MSE)
        valid_predictions = model.predict(X_valid_k)
        validation_error += mean_squared_error(y_valid_k, valid_predictions)
    
    # Average validation error across the folds
    er_valid = validation_error / kf.get_n_splits()
    er_valid_alpha.append(er_valid)
    # --- end of task --- #


# Now you should have obtained a validation error for each alpha value 
# In the homework, you just need to report these values

# Report the validation errors for each alpha value (Table 1)
for alpha, val_err in zip(alpha_vec, er_valid_alpha):
    print(f'Alpha: {alpha}, Validation Error (MSE): {val_err}')

# The following practice is only for your own learning purpose.
# Compare the candidate values and pick the alpha that gives the smallest error 
# set it to "alpha_opt"
alpha_opt = alpha_vec[np.argmin(er_valid_alpha)]

# now retrain your model on the entire training set using alpha_opt 
# then evaluate your model on the testing set 
model = Ridge(alpha = alpha_opt)
model.fit(sample_train, label_train)

# Evaluate training and testing errors
train_predictions = model.predict(sample_train)
test_predictions = model.predict(sample_test)

er_train = mean_squared_error(label_train, train_predictions)
er_test = mean_squared_error(label_test, test_predictions)

# Output the final results
print(f'Optimal Alpha: {alpha_opt}')
print(f'Training Error (MSE): {er_train}')
print(f'Testing Error (MSE): {er_test}')


