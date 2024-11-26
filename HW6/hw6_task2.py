import numpy as np
import pandas as pd

# Define the grid and policy setup
rows, cols = 4, 3
terminal_states = [(4, 2), (4, 3)]
rock_state = (2, 2)
reward = -0.04
gamma = 0.8

# Initial policy
policy = {
    (1, 1): "RIGHT", (1, 2): "UP", (1, 3): "LEFT",
    (2, 1): "UP", (2, 3): "DOWN",
    (3, 1): "LEFT", (3, 2): "LEFT", (3, 3): "RIGHT",
    (4, 1): "LEFT"
}

# Estimated utilities from Task 1
utilities = {
    (1, 1): -0.1996, (1, 2): -0.1992, (1, 3): -0.1992,
    (2, 1): -0.1995, (2, 3): -0.1944,
    (3, 1): -0.1973, (3, 2): -0.1898, (3, 3): -0.1928,
    (4, 1): -0.1845
}

# Transition probabilities
P = {
    "UP": {"UP": 0.6, "LEFT": 0.2, "RIGHT": 0.1, "DOWN": 0.1},
    "DOWN": {"DOWN": 0.6, "LEFT": 0.2, "RIGHT": 0.1, "UP": 0.1},
    "LEFT": {"LEFT": 0.6, "UP": 0.2, "DOWN": 0.1, "RIGHT": 0.1},
    "RIGHT": {"RIGHT": 0.6, "UP": 0.2, "DOWN": 0.1, "LEFT": 0.1},
}

# Define all possible actions
actions = ["UP", "DOWN", "LEFT", "RIGHT"]

# Policy optimization function
def optimize_policy(policy, utilities):
    new_policy = {}
    for state in policy.keys():
        if state in terminal_states:
            new_policy[state] = None  # Terminal states do not require a policy
            continue
        
        max_utility = float("-inf")
        best_action = None
        
        # Evaluate all actions
        for action in actions:
            expected_utility = 0
            for direction, prob in P[action].items():
                row, col = state
                if direction == "UP":
                    next_state = (max(row - 1, 1), col)
                elif direction == "DOWN":
                    next_state = (min(row + 1, rows), col)
                elif direction == "LEFT":
                    next_state = (row, max(col - 1, 1))
                elif direction == "RIGHT":
                    next_state = (row, min(col + 1, cols))
                else:
                    next_state = state
                
                # Handle walls and rocks
                if next_state == rock_state or next_state in terminal_states:
                    next_state = state
                
                expected_utility += prob * utilities.get(next_state, 0)
            
            # Update the best action
            total_utility = reward + gamma * expected_utility
            if total_utility > max_utility:
                max_utility = total_utility
                best_action = action
        
        new_policy[state] = best_action
    
    return new_policy

# Optimize the policy
optimized_policy = optimize_policy(policy, utilities)

# Prepare the updated policy table
policy_table = pd.DataFrame(columns=[1, 2, 3], index=[1, 2, 3, 4])

for state, action in optimized_policy.items():
    row, col = state
    policy_table.loc[row, col] = action

# Display the updated policy table
print("Optimized Policy for Each State:")
for state, action in sorted(optimized_policy.items()):
    print(f"State {state}: Best Action = {action}")

