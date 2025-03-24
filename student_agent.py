# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
from enum import IntEnum


class Action(IntEnum):
    MOVE_DOWN = 0
    MOVE_UP = 1
    MOVE_RIGHT = 2
    MOVE_LEFT = 3
    PICK_UP = 4
    DROP_OFF = 5

    @staticmethod
    def get_space_size():
        return 6

    @staticmethod
    def sample():
        return random.choice(list(Action))

    @staticmethod
    def is_movement(action):
        if action is None:
            return False
        return 0 <= action <= 3

stations = [None] * 4
passenger_pos = None
can_be_passenger = [True] * 4
destination_pos = None
can_be_destination = [True] * 4
carrying = False
num_visited = None

step_count = -1

def get_state(obs, prev_action, should_reset=False):
    # obs = taxi_row,            taxi_col,            self.stations[0][0], self.stations[0][1],
    #       self.stations[1][0], self.stations[1][1], self.stations[2][0], self.stations[2][1],
    #       self.stations[3][0], self.stations[3][1], obstacle_north,      obstacle_south,
    #       obstacle_east,       obstacle_west,       passenger_look,      destination_look

    global stations, passenger_pos, can_be_passenger, destination_pos, can_be_destination, carrying, num_visited, step_count

    taxi_row, taxi_col, _,_,_,_,_,_,_,_, obstacle_north, obstacle_south, \
        obstacle_east, obstacle_west, passenger_look, destination_look = obs

    for i in range(4):
        if (obs[2 * i + 2], obs[2 * i + 3]) != stations[i]:
            # New episode started. Reset everything.
            should_reset = True
            break

    if should_reset:
        stations = [(obs[2 * i + 2], obs[2 * i + 3]) for i in range(4)]
        passenger_pos = None
        can_be_passenger = [True] * 4
        destination_pos = None
        can_be_destination = [True] * 4
        carrying = False
        num_visited = [[0 for j in range(10)] for i in range(10)]
        prev_action = None

        step_count = -1


    rel_positions = [(r - taxi_row, c - taxi_col) for r, c in stations]
    rel_distances = [abs(r) + abs(c) for r, c in rel_positions]

    if passenger_pos is None:
        if passenger_look:
            can_be_passenger = [can_be_passenger[i] and rel_distances[i] <= 1 for i in range(4)]
        else:
            can_be_passenger = [can_be_passenger[i] and rel_distances[i] > 1 for i in range(4)]
    if destination_pos is None:
        if destination_look:
            can_be_destination = [can_be_destination[i] and rel_distances[i] <= 1 for i in range(4)]
        else:
            can_be_destination = [can_be_destination[i] and rel_distances[i] > 1 for i in range(4)]

    if passenger_pos is None and sum(can_be_passenger) == 1:
        index = can_be_passenger.index(True)
        passenger_pos = stations[index]
        can_be_destination[index] = False
    if destination_pos is None and sum(can_be_destination) == 1:
        index = can_be_destination.index(True)
        destination_pos = stations[index]
        can_be_passenger[index] = False
    if passenger_pos is None and sum(can_be_passenger) == 1:
        index = can_be_passenger.index(True)
        passenger_pos = stations[index]


    if prev_action == Action.PICK_UP and (taxi_row, taxi_col) == passenger_pos:
        carrying = True
    elif prev_action == Action.DROP_OFF and carrying:
        carrying = False
        passenger_pos = (taxi_row, taxi_col)
        if (taxi_row, taxi_col) == destination_pos:
            stations[0] = None  # make sure next call to get_state() will reset


    if not carrying and passenger_pos is not None:
        target_rel = (passenger_pos[0] - taxi_row, passenger_pos[1] - taxi_col)
    elif carrying and destination_pos is not None:
        target_rel = (destination_pos[0] - taxi_row, destination_pos[1] - taxi_col)
    else:
        index_closest = None
        for i in range(4):
            if (can_be_destination[i] if carrying else can_be_passenger[i])\
                and (index_closest is None or rel_distances[i] < rel_distances[index_closest]):
                index_closest = i
        target_rel = rel_positions[index_closest]

    target_rel = (np.sign(target_rel[0]), np.sign(target_rel[1]))


    if Action.is_movement(prev_action):
        num_visited[taxi_row][taxi_col] += 1
    num_visited_n = num_visited[taxi_row - 1][taxi_col] // 3 if not obstacle_north else -1
    num_visited_s = num_visited[taxi_row + 1][taxi_col] // 3 if not obstacle_south else -1
    num_visited_e = num_visited[taxi_row][taxi_col + 1] // 3 if not obstacle_east else -1
    num_visited_w = num_visited[taxi_row][taxi_col - 1] // 3 if not obstacle_west else -1

    step_count += 1

    return target_rel, num_visited_n, num_visited_s, num_visited_e, num_visited_w, carrying

action = None
with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

def get_action(obs):

    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys.
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    global action

    epsilon = 0.1  # again
    state = get_state(obs, action)
    if state not in q_table.keys() or np.random.rand() < epsilon:
        action = Action.sample()
    else:
        action = np.argmax(q_table[state])

    return action
