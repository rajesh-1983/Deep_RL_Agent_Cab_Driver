# Import routines

import numpy as np
import math
import random
from gym import spaces
from itertools import product

# Defining hyper parameters
number_of_cities = 5  # number of cities
number_of_hrs_per_day = 24  # number of hours per day
number_of_days = 7  # number of days
fuel_cost_per_hour = 5  # Per hour fuel and other costs
revenue_per_hour = 9  # per hour revenue from a passenger
max_allowed_requests = 15
episode_length_days = 30  # An episode is 30 days long.

# Max request based on locations, lambda for poisson distribution
lambda_loc = [2, 12, 4, 7, 8]

# Totoal state space : 5 * 24 * 7 = 840
"""
There’ll never be requests of the sort where pickup and drop locations are the same. So, the action space A will be: (𝑚−1)∗𝑚 + 1 for m locations. Each action will be a tuple of size 2. You can define action space as below:
• pick up and drop locations (𝑝,𝑞) where p and q both take a value between 1 and m;
• (0,0) tuple that represents ’no-ride’ action.
"""


# helper method to Get next_day for current day
def get_next_hour_and_day(hour, day):
    """
    Get hour and day values from provided hour and day value, if hour > 23 then it return day as next day
    hour will hour-23
    """
    next_hour = hour
    next_day = day
    if hour > (number_of_hrs_per_day - 1):
        next_hour -= (number_of_hrs_per_day - 1)
        if day < 6:
            next_day = day + 1
        else:
            next_day = 0
    return next_hour, next_day


# create action space and add default action
def _create_action_space():
    action_spaces = [(p, q) for p in range[5] for q in range(5) if p != q or (p, q) == (0, 0)]
    return action_spaces


class CabDriver:

    def __init__(self, time_matrix):
        """initialise your state and define your action space and state space"""
        self.location = np.random.choice(range(number_of_cities))
        self.time = np.random.choice(range(number_of_hrs_per_day))
        self.day = np.random.choice(range(number_of_days))
        # XiTjDk (location, hour-of-day and day of week)
        self.state = (self.location, self.time, self.day)
        self.action_space = [(p, q) for p in range(5) for q in range(5) if p != q or (p, q) == (0, 0)]
        self.state_size = number_of_cities + number_of_hrs_per_day + number_of_days
        self.action_size = len(self.action_space)

        self.time_matrix = time_matrix

        # Start the first round
        self.reset()

    def state_encode_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN.
         This method converts a given state into a vector format.
         Hint: The vector is of size m + t + d."""
        vector_size = number_of_cities + number_of_hrs_per_day + number_of_days
        state_vector = np.zeros(vector_size, dtype=int).reshape(1, vector_size)
        # set bits for one hot encoding

        # City will be 0-4, use the state value for city as index
        state_vector[0][state[0]] = 1

        offset = number_of_cities
        # Set the index of time of day to 1
        state_vector[0][offset + state[1]] = 1

        offset += number_of_hrs_per_day
        # Set the index of day of week to 1
        state_vector[0][offset + state[2]] = 1

        return state_vector

    # Use this function if you are using architecture-2
    def state_encode_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given
        state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        vector_size = number_of_cities * 3 + number_of_hrs_per_day + number_of_days
        state_vector = np.zeros(vector_size, dtype=int).reshape(1, vector_size)

        # One hot encoding for state
        # City
        state_vector[0][state[0]] = 1

        # time of day
        offset = number_of_hrs_per_day
        state_vector[0][offset + state[1]] = 1

        # day of week
        offset += number_of_days
        state_vector[0][offset + state[2]] = 1

        # One-hot encoding for action
        # start location
        offset += number_of_cities
        state_vector[0][offset + action[0]] = 1

        # end location
        offset += number_of_cities
        state_vector[0][offset + action[1]] = 1

        return state_vector

    # Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location.
        Use the table specified in the MDP and complete for rest of the locations
        Using location index from 1 to 5 in order avoid confusion with (0, 0) action.
        """
        requests = min(np.random.poisson(lambda_loc[state[0]]), 15)

        # Get possible actions - (0,0) is not considered as customer request
        possible_actions_index = random.sample(range(1,
                                                     ((number_of_cities - 1) * number_of_cities) + 1),
                                               requests)
        actions = [self.action_space[i] for i in possible_actions_index]
        actions.append((0, 0))

        return actions

    def step(self, state, action):
        """
        On every step when particular action taken them compute reward and next_state and
        return those values as tuple.

        @param state: current state
        @param action: current action
        @return: reward, next_state
        """
        reward, trip_duration = self.reward_func(state, action)
        next_state = self.next_state_func(state, action)
        return reward, next_state, trip_duration

    def reward_func(self, state, action):
        """Takes in state, action and Time-matrix and returns the reward"""
        reward = 0
        if action[0] == action[1]:  # This is case of if action (0, 0)
            return -fuel_cost_per_hour, 1

        time_to_pick_up_loc = self.time_matrix[state[0]][action[0]][state[1]][state[2]]
        start_hour = int(state[1] + time_to_pick_up_loc)
        start_day = state[2]
        # If Driver started from current location at 11.00 PM, then he may reach pick location next day.
        start_hour, start_day = get_next_hour_and_day(start_hour, start_day)

        time_to_pickup_to_drop_loc = int(self.time_matrix[action[0]][action[1]][start_hour][start_day])

        reward = revenue_per_hour * time_to_pickup_to_drop_loc - \
                 fuel_cost_per_hour * (time_to_pick_up_loc + time_to_pickup_to_drop_loc)
        return reward, int(time_to_pick_up_loc + time_to_pickup_to_drop_loc)

    def next_state_func(self, state, action):
        """Takes state and action as input and returns next state
        @param state:
        @param action:
        @param time_matrix:
        @return:
        """
        hour = state[1]
        day = state[2]

        if action[0] == action[1]:  # This is case if action (0, 0)
            hour += 1
            # If Driver Moved to Idle state from current location at 11.00 PM,
            # then the next state will be on next day
            next_hour, next_day = get_next_hour_and_day(hour, day)
            return state[0], next_hour, next_day

        time_to_pick_up_loc = self.time_matrix[state[0]][action[0]][state[1]][state[2]]
        hour = int(state[1] + time_to_pick_up_loc)
        day = state[2]
        # If Driver started from current location at 11.00 PM, then he may reach pick location next day.
        hour, day = get_next_hour_and_day(hour, day)

        time_to_pickup_to_drop_loc = int(self.time_matrix[action[0]][action[1]][hour][day])
        # Next state hour and day
        hour += time_to_pickup_to_drop_loc
        # get 'next_hour', 'next_day' on new state after ride completed
        next_hour, next_day = get_next_hour_and_day(hour, day)
        return action[1], next_hour, next_day

    def reset(self):
        self.location = np.random.choice(range(number_of_cities))
        self.time = np.random.choice(range(number_of_hrs_per_day))
        self.day = np.random.choice(range(number_of_days))
        self.state = (self.location, self.time, self.day)
        return self.state
