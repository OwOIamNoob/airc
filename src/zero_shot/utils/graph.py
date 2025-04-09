import numpy as np
from collections import defaultdict

def dfs_tree(g_mask, inp_mask, start, weight = 1, alpha=0.1, thresh=0.95):
    rows, cols = g_mask.shape[:2]
    stack = PriorityQueue()
    stack.put((0, start))

    # Dijikstra
    status = np.zeros(g_mask.shape, dtype=int)

    # This to indicate strictly incremental path
    g_mask = np.abs(g_mask - 0.001)
    cost = np.full_like(g_mask, 1e4)
    px, py = start
    cost[px, py] = 0
    leaves = set()
    border = set()
    begin = start
    parent = {}
    directions = [(-1, 0),(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)] 
    dfs_tree = defaultdict(list)
    
    while not stack.empty():
        
        state, (x, y) = stack.get()
        i = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and g_mask[nx, ny] > alpha and status[nx, ny] <= status[x, y]:
                # Update on inverse confidence
                if cost[nx, ny] > state + (1 - g_mask[nx, ny]) * weight:
                    # Set cost as uncertainty gain 
                    i += 1
                    cost[nx, ny] = state + (1 - g_mask[nx, ny]) * weight
                    # Start by largest margin
                    stack.put((cost[nx, ny], (nx, ny)))
                    # Erase entry from other branch
                    if (nx, ny) in parent.keys():
                        # print(dfs_tree[parent[(nx, ny)]])
                        dfs_tree[parent[(nx, ny)]].remove((nx, ny))

                    parent[(nx, ny)] = (x, y)
                    # Determine that they have gone out of mask
                    # Odd for out-going, Even for in-going.
                    if (g_mask[nx, ny] - thresh) * (g_mask[x, y] - thresh) < 0 :
                        if status[nx, ny] == 0:
                            border.add((nx, ny))
                        status[nx, ny] = status[x, y] + 1
                    else:
                        status[nx, ny] = status[x, y] 
                    dfs_tree[(x, y)].append((nx, ny))
            
        if i == 0:
            # Force leaves to be sink
            status[x, y] += inp_mask[x, y] % 2
            leaves.add((x, y))
            # Continual of flow
            if inp_mask[x, y] == 1:
                begin = (x, y)
    return {'dfs_tree': dfs_tree, 
            'parent': parent, 
            'cost': cost, 
            'border': border, 
            'leaves': leaves, 
            'status': status, 
            'begin': begin}

# We do post 
def longest_path_branching(tree, start):
    visited = set()
    branches = []
    def dfs(node, path, visited, branches):
        if len(tree[node]) == 0:
            return [node]
        if node in visited:
            return path
        paths = []
        visited.add(node)
        for neighbor in tree[node]:
            if neighbor in visited:
                continue
            paths.append(dfs(neighbor, path[:], visited, branches) + [node])

        if len(paths) == 0:
            max_path = [node]
        else:
            paths = sorted(paths, key= lambda x: -len(x))
            max_path = paths[0]
            branches += paths[1:]
        return max_path
    output = dfs(start, [], visited, branches)
    return  branches + [output] 