import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import pandas as pd
import copy

# Bring in Code from Notebook
from worlds import *
from astar import * 

# Create a Title
st.title('A* Path-Finding Application')

# Set the START and GOAL positions
start_row = st.number_input("Enter the Row of the Starting Position", min_value=0, max_value=len(full_world)-1)
start_col = st.number_input("Enter the Col of the Starting Position", min_value=0, max_value=len(full_world[0])-1)
goal_row = st.number_input("Enter the Row of the Goal Position", min_value=0, max_value=len(full_world[0])-1)
goal_col = st.number_input("Enter the Col of the Goal Position", min_value=0, max_value=len(full_world[0])-1)

# Initialize the Path
path = a_star_search(full_world, (start_row, start_col), (goal_row, goal_col), COSTS, MOVES, my_heuristic)

# Cache the Dataframe Holding the world 
@st.cache
def get_world():
    global path
    # Get the Standard Layout
    world = copy.deepcopy(full_world)
    
    # Display the Start and the Finish
    world[start_row][start_col] = 'S'
    world[goal_row][goal_col] = 'G'

    if path is not None:
        for i in range(len(path)-1):
            world[path[i][0]][path[i][1]] = determine_move(path[i], path[i+1])

    # Return Dataframe
    df = pd.DataFrame(world)
    return df

st.dataframe(get_world().style.applymap(pretty), 2000, 1000)

