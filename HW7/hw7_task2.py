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
#k_values = [5, 10, 20, 50, 100]
m_values = [1, 2, 3, 5, 10]
fixed_neurons = 20 

er_train_m = []
er_test_m = []

for m in m_values:
    # Define MLPClassifier with m layers and fixed_neurons neurons per layer
    mlp = MLPClassifier(hidden_layer_sizes=(fixed_neurons,) * m, max_iter=1000, random_state=0)
    
    # Train the model
    mlp.fit(sample_train, label_train)
    
    # Predict on training and testing data
    pred_train = mlp.predict(sample_train)
    pred_test = mlp.predict(sample_test)
    
    # Compute errors
    er_train = 1 - accuracy_score(label_train, pred_train)
    er_test = 1 - accuracy_score(label_test, pred_test)
    
    # Append errors to respective lists
    er_train_m.append(er_train)
    er_test_m.append(er_test)

# Plotting the results
plt.figure()
plt.plot(m_values, er_train_m, label='Training Error', marker='o')
plt.plot(m_values, er_test_m, label='Testing Error', marker='o')
plt.xlabel('Number of Layers (m)')
plt.ylabel('Classification Error')
plt.title('Impact of Layer Depth on Classification Accuracy')
plt.legend()
plt.grid()
plt.show()

