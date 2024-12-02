import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = np.loadtxt(r'c:/Users/Administrator/Desktop/OU/AI/HW7/diabetes.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# training data 
num_train = int(0.5*n)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
# testing data 
num_test = int(0.5*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

# ----------------------- #
# --- Hyper-Parameter --- #
# ----------------------- #
fixed_layers = 5
fixed_neurons = 20
activation_functions = ['identity', 'logistic', 'tanh', 'relu']

# List to store results
results = []

# Iterate over activation functions
for activation in activation_functions:
    # Define MLPClassifier with fixed parameters
    mlp = MLPClassifier(hidden_layer_sizes=(fixed_neurons,) * fixed_layers, 
                         activation=activation, max_iter=1000, random_state=0)
    
    # Train the model
    mlp.fit(sample_train, label_train)
    
    # Predict on training and testing data
    pred_train = mlp.predict(sample_train)
    pred_test = mlp.predict(sample_test)
    
    # Compute errors
    train_error = 1 - accuracy_score(label_train, pred_train)
    test_error = 1 - accuracy_score(label_test, pred_test)
    
    # Append results
    results.append([activation, train_error, test_error])

# Display results as a table
print(f"{'Activation Function':<15}{'Training Error':<15}{'Testing Error':<15}")
for activation, train_error, test_error in results:
    print(f"{activation:<15}{train_error:<15.4f}{test_error:<15.4f}")