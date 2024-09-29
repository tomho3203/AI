import numpy as np
import random

# Fitness function (inverse of error function)
def fitness_function(w, X, y):
    return np.exp(-error_function(w, X, y))

# Selection based on fitness (roulette wheel selection)
def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    
    # Check for NaN or invalid fitness values
    if np.isnan(total_fitness) or total_fitness == 0:
        # If total fitness is NaN or zero, assign equal probability to all
        selection_probs = [1 / len(fitnesses)] * len(fitnesses)
    else:
        # Normalize the fitnesses to get probabilities
        selection_probs = [f / total_fitness if not np.isnan(f) else 0 for f in fitnesses]
    
    # Handle the case where the sum of probabilities is not 1
    selection_probs = np.array(selection_probs)
    selection_probs = selection_probs / selection_probs.sum()

    print(f"Fitnesses: {fitnesses}")  # Debugging
    print(f"Selection probabilities: {selection_probs}")  # Debugging

    # Select two parents based on their probabilities
    parent_indices = np.random.choice(len(population), size=2, p=selection_probs)
    parent1 = population[parent_indices[0]]
    parent2 = population[parent_indices[1]]
    return parent1, parent2

# Initialize population with random chromosomes (w vectors)
def initialize_population(pop_size=10):
    return [np.random.choice([-1, 1], size=6) for _ in range(pop_size)]

# Error function definition (you would define this based on your problem)
def error_function(w, X, y):
    predictions = np.dot(X, w)
    errors = (predictions - y) ** 2
    return np.mean(errors)

# Genetic algorithm implementation (simplified)
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
            # Apply crossover and mutation here (not shown in this snippet)
            # Add offspring to next population

        # Check best chromosome of the generation
        for w in population:
            current_error = error_function(w, X, y)
            if current_error < best_error:
                best_chromosome = w
                best_error = current_error
        
        error_history.append(best_error)

    return best_chromosome, best_error, error_history

# Run the genetic algorithm with your dataset
# You would replace X and y with actual data
# optimal_w, final_error, error_history = genetic_algorithm(X, y)
