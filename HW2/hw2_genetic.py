import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Load the dataset
data = pd.read_csv('HW2/CreditCard.csv')

# Encode categorical variables
data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})
data['CarOwner'] = data['CarOwner'].map({'Y': 1, 'N': 0})
data['PropertyOwner'] = data['PropertyOwner'].map({'Y': 1, 'N': 0})

# Features and target
X = data[['Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email']].values
y = data['CreditApprove'].values

# Error function definition
def error_function(w, X, y):
    predictions = np.dot(X, w)
    errors = (predictions - y) ** 2
    return np.mean(errors)

# Fitness function (inverse of error function)
def fitness_function(w, X, y):
    return np.exp(-error_function(w, X, y))

# Initialize population with random chromosomes (w vectors)
def initialize_population(pop_size=10):
    return [np.random.choice([-1, 1], size=6) for _ in range(pop_size)]

# Selection based on fitness (roulette wheel selection)
def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    # Select two parents based on their probabilities
    parent_indices = np.random.choice(len(population), size=2, p=selection_probs)
    parent1 = population[parent_indices[0]]
    parent2 = population[parent_indices[1]]
    return parent1, parent2

# Crossover between two parents
def crossover(parent1, parent2):
    crossover_point = len(parent1) // 2
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2

# Mutation (flip a random bit)
def mutate(child, mutation_rate=0.1):
    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] *= -1
    return child

# Genetic algorithm
def genetic_algorithm(X, y, pop_size=10, generations=100, mutation_rate=0.1):
    population = initialize_population(pop_size)
    best_chromosome = None
    best_error = float('inf')
    error_history = []
    
    for gen in range(generations):
        fitnesses = [fitness_function(w, X, y) for w in population]
        next_population = []
        
        # Create next generation
        while len(next_population) < pop_size:
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            next_population.extend([child1, child2])
        
        # Ensure the next population size is exactly pop_size
        population = next_population[:pop_size]
        
        # Check the best chromosome of the generation
        for w in population:
            current_error = error_function(w, X, y)
            if current_error < best_error:
                best_chromosome = w
                best_error = current_error
        
        error_history.append(best_error)

    return best_chromosome, best_error, error_history

# Run the genetic algorithm
optimal_w, final_error, error_history = genetic_algorithm(X, y, pop_size=10, generations=100, mutation_rate=0.1)

# Plotting the error vs. generation
plt.plot(error_history)
plt.title('Error vs. Generation (Genetic Algorithm)')
plt.xlabel('Generation')
plt.ylabel('Error')
plt.grid(True)
plt.show()

# Output the optimal weights and final error
print(f"Optimal w: {optimal_w}")
print(f"Final error: {final_error}")
