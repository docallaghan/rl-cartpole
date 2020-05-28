# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:37:08 2020

@author: ocall
"""

import time
import gym

def basic_policy(obs):
    # If the pole is tilting to the left, then 
    # push the cart to the left, and vice versa
    angle = obs[2]
    return 0 if angle < 0 else 1

env = gym.make("CartPole-v1")

obs = env.reset()

while True:
    # Show the environment
    env.render()
    
    # Select an action from policy
    action = basic_policy(obs)
    
    # Take action and observe new state
    obs, reward, done, info = env.step(action)
    
    # Make animation slower
    time.sleep(0.03)
    
    # Check termination condition
    if done:
        env.close()
        break
