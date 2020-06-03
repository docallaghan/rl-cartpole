# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:10:04 2020

Author: David O'Callaghan
"""

import time
import gym
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from collections import deque

n_inputs = 4 # == env.observation_space.shape[0]
n_outputs = 2

model = keras.models.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=[n_inputs]),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(n_outputs)
])

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                       (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
def nn_policy(obs):
    probs = model.predict(obs.reshape(1, -1))
    action = np.argmax(probs)
    return action
    
if __name__=="__main__":
    replay_memory = deque(maxlen=1000000)
    
    batch_size = 20
    discount_rate = 0.95
    optimizer = keras.optimizers.Adam(lr=1e-3)
    loss_fn = keras.losses.mean_squared_error
    
    rewards = [] 
    best_score = 0
    
    env = gym.make("CartPole-v1")
    env.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    early_stopping_counter = 0
    
    for episode in range(600):
        obs = env.reset()    
        for step in range(200):
            epsilon = max(1 - episode / 500, 0.01)
            obs, reward, done, info = play_one_step(env, obs, epsilon)
            if done:
                if step >= 190:
                    early_stopping_counter += 1
                else:
                    early_stopping_counter = 0
                break
        rewards.append(step) # Not shown in the book
        if early_stopping_counter >= 3:
            break
        if step > best_score: # Not shown
            best_weights = model.get_weights() # Not shown
            best_score = step # Not shown
        print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(episode, step + 1, epsilon), end="") # Not shown
        if episode > 50:
            training_step(batch_size)
    
    model.set_weights(best_weights)
    
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards",)
    plt.show()
    
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
            print(f'\nFailed after {i} time steps')
            break
    env.close()