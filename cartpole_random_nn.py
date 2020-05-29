# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:16:33 2020

@author: ocall
"""

import time
import gym
import numpy as np
from tensorflow import keras

n_inputs = 4 # == env.observation_space.shape[0]

model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid"),
])

def nn_policy(obs):
    left_proba = model.predict(obs.reshape(1, -1))
    action = int(np.random.rand() > left_proba)
    return action

if __name__ == "__main__":  
    env = gym.make("CartPole-v1")
    
    obs = env.reset()
    
    for i in range(250):
        # Show the environment
        env.render()
        
        # Select an action from policy
        action = nn_policy(obs)
        
        # Take action and observe new state
        obs, reward, done, info = env.step(action)
        
        # Make animation slower
        time.sleep(0.03)
        
        # Check termination condition
        if done:
            print(f'Failed after {i} time steps')
            env.close()
            break
