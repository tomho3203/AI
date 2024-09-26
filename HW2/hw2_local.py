import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data_cleanset
data = pd.read_csv('HW2\CreditCard.csv')
data_clean = data.dropna()

# Encode categorical variables
data_clean.loc[:, 'Gender'] = data_clean['Gender'].map({'M': 1, 'F': 0})
data_clean.loc[:, 'CarOwner'] = data_clean['CarOwner'].map({'Y': 1, 'N': 0})
data_clean.loc[:, 'PropertyOwner'] = data_clean['PropertyOwner'].map({'Y': 1, 'N': 0})

# Features and target
X = data_clean[['Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID']].values
y = data_clean['CreditApprove'].values

# Error function definition
def error_function(w, X, y):
    predictions = np.dot(X, w)
    errors = (predictions - y) ** 2
    return np.mean(errors)

# Function to find adjacent solutions (differ by one element)
def get_adjacent_solutions(w):
    adjacent_solutions = []
    for i in range(len(w)):
        w_copy = w.copy()
        w_copy[i] *= -1  # Flip the ith element
        adjacent_solutions.append(w_copy)
    return adjacent_solutions

# Hill climbing local search
def hill_climbing(X, y, max_rounds=340):
    w = np.array([-1, -1, -1, -1, -1, -1])  # Initial solution
    error_history = []
    
    for round_num in range(max_rounds):
        current_error = error_function(w, X, y)
        error_history.append(current_error)
        
        # Get all adjacent solutions
        adjacent_solutions = get_adjacent_solutions(w)
        best_solution = w
        best_error = current_error
        
        # Find the best adjacent solution
        for adj_w in adjacent_solutions:
            adj_error = error_function(adj_w, X, y)
            if adj_error < best_error:
                best_solution = adj_w
                best_error = adj_error
        
        # If no improvement, break the loop
        if best_error == current_error:
            print(f"No improvement found in round {round_num}, stopping search.")
            break
        
        # Update w to the best solution found
        w = best_solution
    
    print(f"Optimal solution found: w = {w}, error = {best_error}")
    return w, current_error, error_history

# Run hill climbing search
optimal_w, final_error, error_history = hill_climbing(X, y, max_rounds=340)

# Plotting the error vs. rounds
if error_history:
    plt.plot(error_history)
    plt.title('Error vs. Search Round (Hill Climbing)')
    plt.xlabel('Round of Search')
    plt.ylabel('Error')
    plt.grid(True)
    plt.show()
else:
    print("No error history to plot.")