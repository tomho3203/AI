import numpy as np
from collections import defaultdict

# Grid setup
rows, cols = 4, 3
grid = np.zeros((rows, cols))
terminal_states = [(4, 2), (4, 3)]
rock_state = (2, 2)

# Transition probabilities
P = {
    "UP": {"UP": 0.6, "LEFT": 0.2, "RIGHT": 0.1, "DOWN": 0.1},
    "DOWN": {"DOWN": 0.6, "LEFT": 0.2, "RIGHT": 0.1, "UP": 0.1},
    "LEFT": {"LEFT": 0.6, "UP": 0.2, "DOWN": 0.1, "RIGHT": 0.1},
    "RIGHT": {"RIGHT": 0.6, "UP": 0.2, "DOWN": 0.1, "LEFT": 0.1},
}

# Reward and discount
reward = -0.04
gamma = 0.8

# Policy (initially given as default actions)
policy = {
    (1, 1): "RIGHT", (1, 2): "UP", (1, 3): "LEFT",
    (2, 1): "UP", (2, 3): "DOWN",
    (3, 1): "LEFT", (3, 2): "LEFT", (3, 3): "RIGHT",
    (4, 1): "LEFT"
}

# Generate random movement based on action probabilities
def stochastic_move(state, action):
    row, col = state
    probabilities = P[action]
    move = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))
    
    if move == "UP":
        next_state = (max(row - 1, 1), col)
    elif move == "DOWN":
        next_state = (min(row + 1, rows), col)
    elif move == "LEFT":
        next_state = (row, max(col - 1, 1))
    elif move == "RIGHT":
        next_state = (row, min(col + 1, cols))
    else:
        next_state = state
    
    # Handle walls/rocks
    if next_state == rock_state or next_state[1] > cols or next_state[0] > rows:
        next_state = state
    return next_state

# Monte Carlo simulation
def first_visit_mc(policy, episodes=10):
    utilities = defaultdict(list)
    for _ in range(episodes):
        for state in policy.keys():
            if state in terminal_states:
                continue
            
            current_state = state
            episode = []
            total_reward = 0
            discount = 1.0
            
            # Generate an episode
            while current_state not in terminal_states:
                action = policy[current_state]
                next_state = stochastic_move(current_state, action)
                episode.append((current_state, reward))
                total_reward += discount * reward
                discount *= gamma
                current_state = next_state
            
            # First-visit utility update
            visited_states = set()
            for (s, r) in episode:
                if s not in visited_states:
                    utilities[s].append(total_reward)
                    visited_states.add(s)
    
    # Compute average utilities
    avg_utilities = {s: np.mean(utilities[s]) for s in utilities}
    return avg_utilities

# Run simulation
utilities = first_visit_mc(policy, episodes=10)

# Display results
print("Estimated Utilities:")
for state, utility in sorted(utilities.items()):
    print(f"State {state}: Utility = {utility:.4f}")
