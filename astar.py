
# Constants 
MOVES = [(0,-1), (1,0), (0,1), (-1,0)]
COSTS = { '.': 1, '*': 3, '#': 5, '~': 7}

# Returns boolean whether the given location is on the frontier list
def on_frontier(_si, _frontier):
    frontier_locations = [node[-1] for node in _frontier]
    return _si in frontier_locations

# Returns the successors of the current state
def successors(_si, _world):
    successors = []
    for move in MOVES:
        candidate = (_si[0] + move[0], _si[1] + move[1])
        if candidate[0] >= 0 and candidate[0] < len(_world) and \
            candidate[1] >= 0 and candidate[1] < len(_world[0]):
                successors.append(candidate)
    return successors

# Function to initialize the cumulative cost of reaching each node in the world
def initialize_costs(_world):
    costs={}
    for i in range(len(_world)):
        for j in range(len(_world)):
            costs[(i,j)]=float('inf')
    return costs

# Returns the Cost of traversing the given location
def cost(_si, _world):
    if _world[_si[0]][_si[1]] in COSTS.keys():
        return COSTS[_world[_si[0]][_si[1]]]
    else:
        return float('inf')

import math
# change the formal arguments and the return value to be what you need.
def my_heuristic(_si, _world, _goal):
    return math.sqrt((_si[0] - _goal[0])**2 + (_si[1] - _goal[1])**2)

# Re-creates the path from Goal to the Start using explored Map
def backtrack_path(_explored, _current):
    final_path = [_current]
    while _current in _explored.keys():
        _current = _explored[_current]
        final_path.insert(0, _current)
    return final_path

'''
a_star_search

The a_star_search function uses the A* Search algorithm to solve a navigational problem for an agent in a grid world. 
It calculates a path from the start state to the goal state and returns the actions required to get from the start to the goal.

    world is the starting state representation for a navigation problem.
    start is the starting location, (x, y).
    goal is the desired end position, (x, y).
    costs is a Dict of costs for each type of terrain.
    moves is the legal movement model expressed in offsets.
    heuristic is a heuristic function that returns an estimate of the total cost f(x) from the start to the goal 
        through the current node, x. The heuristic function might change with the movement model.

'''
from heapq import heappush, heappop

def a_star_search( world, start, goal, costs, moves, heuristic):
    # Initialize Frontier
    frontier = []
    heappush(frontier, (heuristic(_si=start, _world=world, _goal=goal), start))
    # Expored Path so we can backtrack from GOAL
    explored = {}
    # Initialize the cumaltice costs to reach each node
    cumulative_costs = initialize_costs(_world=world)
    cumulative_costs[start] = 0
    # Initialize the total projected cost to reach the goal throug the given node
    total_costs = initialize_costs(_world=world)
    total_costs[start] = heuristic(_si=start, _world=world, _goal=goal)
    while len(frontier) > 0:
        # Pull the first unexplored node off the frontier
        projected_cost, node_current = heappop(frontier)
        # Check if we've reached our goal
        if node_current == goal:
            return backtrack_path(_explored=explored, _current=node_current)
        # Generare the Successor States from the Current
        node_successors = successors(_si=node_current, _world=world)
        for node_successor in node_successors:
            successor_cumulative_cost = cumulative_costs[node_current] + cost(_si=node_current, _world=world)
            if successor_cumulative_cost < cumulative_costs[node_successor]:
                # We found a more cost-efficient path to the successor!!
                # Update the Path so we can backtrack from the Goal
                explored[node_successor] = node_current
                # Update the Cumulative Cost to reach the Successor State
                cumulative_costs[node_successor] = successor_cumulative_cost
                # Update the Total projected costs to reach the goal through the Succesor State
                total_costs[node_successor] = cumulative_costs[node_successor] + heuristic(_si=node_successor, _world=world, _goal=goal)
                if not on_frontier(_si=node_successor, _frontier=frontier):
                    heappush(frontier, (total_costs[node_successor], node_successor))
    return [] # change to return the real answer

from heapq import heappush, heappop

def a_star_search( world, start, goal, costs, moves, heuristic):
    # Initialize Frontier
    frontier = []
    heappush(frontier, (heuristic(_si=start, _world=world, _goal=goal), start))
    # Expored Path so we can backtrack from GOAL
    explored = {}
    # Initialize the cumaltice costs to reach each node
    cumulative_costs = initialize_costs(_world=world)
    cumulative_costs[start] = 0
    # Initialize the total projected cost to reach the goal throug the given node
    total_costs = initialize_costs(_world=world)
    total_costs[start] = heuristic(_si=start, _world=world, _goal=goal)
    while len(frontier) > 0:
        # Pull the first unexplored node off the frontier
        projected_cost, node_current = heappop(frontier)
        # Check if we've reached our goal
        if node_current == goal:
            return backtrack_path(_explored=explored, _current=node_current)
        # Generare the Successor States from the Current
        node_successors = successors(_si=node_current, _world=world)
        for node_successor in node_successors:
            successor_cumulative_cost = cumulative_costs[node_current] + cost(_si=node_current, _world=world)
            if successor_cumulative_cost < cumulative_costs[node_successor]:
                # We found a more cost-efficient path to the successor!!
                # Update the Path so we can backtrack from the Goal
                explored[node_successor] = node_current
                # Update the Cumulative Cost to reach the Successor State
                cumulative_costs[node_successor] = successor_cumulative_cost
                # Update the Total projected costs to reach the goal through the Succesor State
                total_costs[node_successor] = cumulative_costs[node_successor] + heuristic(_si=node_successor, _world=world, _goal=goal)
                if not on_frontier(_si=node_successor, _frontier=frontier):
                    heappush(frontier, (total_costs[node_successor], node_successor))
    return [] # change to return the real answer

# To print the Correct Character when pretty-printing the path
# (0,-1), (1,0), (0,1), (-1,0)
def determine_move(_si, _sj):
    move = (_sj[0] - _si[0], _sj[1] - _si[1])
    if move == (0, -1):
        return '<'
    elif move == (1,0):
        return 'v'
    elif move == (0, 1):
        return '>'
    elif move == (-1, 0):
        return '^'
    else:
        return '?'

def pretty_print_solution( world, path, start):
    # Replace the World Positions with the Path Charachters
    for i in range(len(path)-1):
        world[path[i][0]][path[i][1]] = determine_move(path[i], path[i+1])
    # Show the Final Goal State
    world[path[-1][0]][path[-1][1]] = 'G'
    for row in world:
        print("".join(row))


def pretty(val):
    COSTS = { '.': 1, '*': 3, '#': 5, '~': 7}

    if val == '.':
        color = 'lightgreen'
    elif val == '*':
        color = 'limegreen'
    elif val == '#':
        color = 'green'
    elif val == '~':
        color = 'darkgreen'
    elif val == 'x':
        color = 'gray'
    elif val == 'G':
        color = 'goldenrod'
    else:
        color = 'gold'
    return 'background-color: %s' % color
