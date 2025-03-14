# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

def get_action(obs):

    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys.
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    action = [0, 1, 2, 3, 4, 5]
    if obs[10]==1:
        action.remove(1)
    if obs[13]==1:
        action.remove(3)
    if obs[12]==1:
        action.remove(2)
    if obs[11]==1:
        action.remove(0)
    if obs[14]!=1:
        action.remove(4)
    if obs[15]!=1:
        action.remove(5)
    return random.choice(action)
