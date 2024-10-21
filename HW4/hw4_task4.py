import numpy as np
import matplotlib.pyplot as plt


# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# ......
# --- end of task --- #


# load a data set for classification 
# in array "data", each row represents a patient 
# each column represents an attribute of patients 
# last column is the binary label: 1 means the patient has diabetes, 0 means otherwise
data = np.loadtxt('HW4/diabetes.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]


# --- Your Task --- #
# now, vary the percentage of data used for training 
# pick 8 values for array "num_train_per" e.g., 0.5 means using 50% of the available data for training 
# You should aim to observe overiftting (and normal performance) from these 8 values 
# Note: maximum percentage is 0.75
num_train_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75]
# --- end of task --- #

er_train_per = []
er_test_per = []
for per in num_train_per: 

    # create training data and label 
    num_train = int(n*per)
    sample_train = data[0:num_train,0:-1]
    label_train = data[0:num_train,-1]
    
    # we will use logistic regression model 
    model = LogisticRegression()
    
    # --- Your Task --- #
    # now, training your model using training data 
    model.fit(sample_train, label_train)

    # now, evaluate training error (not MSE) of your model 
    # store it in "er_train"
    train_predictions = model.predict(sample_train)
    er_train = 1 - accuracy_score(label_train, train_predictions)
    er_train_per.append(er_train)
    
    # now, evaluate testing error (not MSE) of your model 
    # store it in "er_test"
    test_predictions = model.predict(sample_test)
    er_test = 1 - accuracy_score(label_test, test_predictions)
    er_test_per.append(er_test)
    # --- end of task --- #
    
plt.figure(figsize=(8, 6))
plt.plot(num_train_per, er_train_per, label='Training Error', color='blue', marker='o')
plt.plot(num_train_per, er_test_per, label='Testing Error', color='red', marker='o')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification Error')
plt.title('Impact of Training Data Size on Classification Performance')
plt.legend()
plt.grid(True)
plt.show()



