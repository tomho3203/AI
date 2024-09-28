# Problem: Implement the Breadth-First Search (BFS), Depth-First Search (DFS) 
# and Greedy Best-First Search (GBFS) algorithms on the graph from Figure 1 in hw1.pdf.


# Instructions:
# 1. Represent the graph from Figure 1 in any format (e.g. adjacency matrix, adjacency list).
# 2. Each function should take in the starting node as a string. Assume the search is being performed on
#    the graph from Figure 1.
#    It should return a list of all node labels (strings) that were expanded in the order they where expanded.
#    If there is a tie for which node is expanded next, expand the one that comes first in the alphabet.
# 3. You should only modify the graph representation and the function body below where indicated.
# 4. Do not modify the function signature or provided test cases. You may add helper functions. 
# 5. Upload the completed homework to Gradescope, it must be named 'hw1.py'.

# Examples:
#     The test cases below call each search function on node 'S' and node 'A'
# -----------------------------
from collections import deque

adj_matrix = [
    #A,B,C,D,E,F,G,H,I,J,K,L,M,N,P,Q,S
    [0, 4, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],   # A
    [4, 0, 2, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],    # B
    [-1, 2, 0, -1, -1, -1, -1, 4, -1, -1, -1, -1, -1, -1, -1, -1 , 3],   # C
    [-1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, 2], # D
    [1, -1, -1, -1, 0, 3, -1, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1],   # E
    [-1, 2, -1, -1, 3, 0, -1, -1, -1, 6, 4, -1, -1, -1, -1, -1 ,-1],   # F
    [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 4, 4, -1, 10, -1], # G
    [-1, -1, 4, -1, -1, -1, -1, 0, -1, -1, 3, 7, -1, -1, -1, -1, -1],   # H
    [-1, -1, -1, -1, 6, -1, -1, -1, 0, 1, -1, -1, 5, -1, -1, -1, -1], #I
    [-1, -1, -1, -1, -1, 6, -1, -1, 1, 0, 3, -1, -1, 3, -1, -1, -1], #J
    [-1, -1, -1, -1, -1, 4, -1, 3, -1, 3, 0, 9, -1, -1, 3, -1, -1], #K
    [-1, -1, -1, 8, -1, -1, -1, 7, -1, -1, 9, 0, -1, -1, -1, 10, -1], #L
    [-1, -1, -1, -1, -1, -1, 4, -1, 5, -1, -1, -1, 0, -1, -1, -1, -1], #M
    [-1, -1, -1, -1, -1, -1, 4, -1, -1, 3, -1, -1, -1, 0, 2, -1, -1], #N
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, 2, 0, -1, -1], #P
    [-1, -1, -1, -1, -1, -1, 10, -1, -1, -1, -1, 10, -1, -1, -1, 0, -1], #Q
    [-1, -1, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], #S
]

# Map nodes to their index in the adjacency matrix
node_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 
            'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'P': 14, 'Q': 15, 'S': 16}
# Map nodes to their h(n) value
heuristic = {'A': 10, 'B': 9, 'C': 16, 'D': 21, 'E': 13, 'F': 9, 'G': 0, 'H': 12, 'I': 9, 
            'J': 5, 'K': 8, 'L': 18, 'M': 3, 'N': 4, 'P': 6, 'Q': 9, 'S': 17}

# Map indices back to nodes
reverse_node_map = {v: k for k, v in node_map.items()}

def BFS(start: str) -> list:
    start_idx = node_map[start]  # Convert start node to index
    end_idx = node_map['G']      # Convert end node (G) to index
    
    # Initialize BFS structures
    queue = deque([start_idx])
    visited = set([start_idx])
    search_sequence = []
    
    while queue:
        current = queue.popleft()
        search_sequence.append(reverse_node_map[current])
        
        # Stop when we reach the end node G
        if current == end_idx:
            break
        
        # Check if 'G' is a direct neighbor of the current node
        if adj_matrix[current][end_idx] != -1:
            search_sequence.append('G')
            break
        
        # Find all unvisited neighbors, prioritize alphabetically by node name
        neighbors = [(reverse_node_map[i], i) for i, weight in enumerate(adj_matrix[current]) if weight != -1 and i not in visited]
        neighbors.sort()  # Alphabetical sorting based on node name
        
        for _, neighbor_idx in neighbors:
            if neighbor_idx not in visited:
                queue.append(neighbor_idx)
                visited.add(neighbor_idx)
    
    return search_sequence


def DFS(start: str) -> list:
    start_idx = node_map[start]  # Convert start node to index
    end_idx = node_map['G']      # Convert end node (G) to index
    
    # Initialize DFS structures
    stack = [start_idx]
    visited = set()
    search_sequence = []
    
    while stack:
        current = stack.pop()
        
        if current not in visited:
            search_sequence.append(reverse_node_map[current])
            visited.add(current)
        
        # Stop when we reach the end node G
        if current == end_idx:
            break

        # Find all unvisited neighbors
        neighbors = [(reverse_node_map[i], i) for i, weight in enumerate(adj_matrix[current]) if weight != -1 and i not in visited]
        
        # Sort neighbors alphabetically by node name
        neighbors.sort(reverse=True)  # Reverse to prioritize smallest alphabetical order in DFS stack
        
        # Push neighbors onto the stack
        for _, neighbor_idx in neighbors:
            if neighbor_idx not in visited:
                stack.append(neighbor_idx)
    
    return search_sequence



def GBFS(start: str) -> list:
    start_idx = node_map[start]  # Convert start node to index
    end_idx = node_map['G']      # Convert end node (G) to index
    
    # Priority queue for Greedy Best-First Search, with (heuristic value, node index)
    priority_queue = [(heuristic[start], start_idx)]
    visited = set()
    search_sequence = []

    while priority_queue:
        # Sort by heuristic value, then by node name for tie-breaking
        priority_queue.sort(key=lambda x: (x[0], reverse_node_map[x[1]]))
        _, current = priority_queue.pop(0)

        if current in visited:
            continue
        
        search_sequence.append(reverse_node_map[current])
        visited.add(current)

        # Stop when we reach the goal node 'G'
        if current == end_idx:
            break

        # Add all unvisited neighbors to the priority queue
        for i, weight in enumerate(adj_matrix[current]):
            if weight != -1 and i not in visited:
                priority_queue.append((heuristic[reverse_node_map[i]], i))
    
    return search_sequence



# test cases - DO NOT MODIFY THESE
def run_tests():
    # Test case 1: BFS starting from node 'A'
    assert BFS('A') == ['A', 'B', 'E', 'C', 'F', 'I', 'H', 'S', 'J', 'K', 'M', 'G'], "Test case 1 failed"
    
    # Test case 2: BFS starting from node 'S'
    assert BFS('S') == ['S', 'C', 'D', 'B', 'H', 'L', 'A', 'F', 'K', 'Q', 'G'], "Test case 2 failed"

    # Test case 3: DFS starting from node 'A'
    assert DFS('A') == ['A', 'B', 'C', 'H', 'K', 'F', 'E', 'I', 'J', 'N', 'G'], "Test case 3 failed"
    
    # Test case 4: DFS starting from node 'S'
    assert DFS('S') == ['S', 'C', 'B', 'A', 'E', 'F', 'J', 'I', 'M', 'G'], "Test case 4 failed"

    # Test case 5: GBFS starting from node 'A'
    assert GBFS('A') == ['A', 'B', 'F', 'J', 'N', 'G'], "Test case 5 failed"
    
    # Test case 6: GBFS starting from node 'S'
    assert GBFS('S') == ['S', 'C', 'B', 'F', 'J', 'N', 'G'], "Test case 6 failed"

    
    
    print("All test cases passed!")

if __name__ == '__main__':
    run_tests()