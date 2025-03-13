# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

with open("q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)

def get_state(obs):
    taxi_row,   taxi_col, \
    Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs  
    
    return (obstacle_north, obstacle_south, obstacle_east, obstacle_west)


def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    
    state = get_state(obs)

    return np.argmax(Q_table[state])
    # You can submit this random agent to evaluate the performance of a purely random strategy.

