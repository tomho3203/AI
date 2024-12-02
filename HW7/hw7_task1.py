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
k_values = [5, 10, 20, 50, 100]
# m_values = [0,0,0,0,0]

er_train_k = []
er_test_k = []
for k in k_values: 
    
    # Define MLPClassifier with 5 layers and k neurons per layer
    mlp = MLPClassifier(hidden_layer_sizes=(k, k, k, k, k), max_iter=1000, random_state=0)
    
    # Train the model
    mlp.fit(sample_train, label_train)
    
    # Predict on training and testing data
    pred_train = mlp.predict(sample_train)
    pred_test = mlp.predict(sample_test)
    
    # Compute errors
    er_train = 1 - accuracy_score(label_train, pred_train)
    er_test = 1 - accuracy_score(label_test, pred_test)

    er_train_k.append(er_train)
    er_test_k.append(er_test)
   
# Plotting the results
plt.figure()
plt.plot(k_values, er_train_k, label='Training Error', marker='o')
plt.plot(k_values, er_test_k, label='Testing Error', marker='o')
plt.xlabel('Number of Neurons per Layer (k)')
plt.ylabel('Classification Error')
plt.title('Impact of Layer Size on Classification Accuracy')
plt.legend()
plt.grid()
plt.show()

