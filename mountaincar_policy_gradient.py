# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:09:38 2020

Author: David O'Callaghan
"""

import time
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

n_inputs = 2 # == env.observation_space.shape[0]

model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(3, activation='softmax'),
])

def play_one_step(env, obs, model, loss_fn):
    actions = [0,1,2]
    with tf.GradientTape() as tape:
        probs = model(obs[np.newaxis]) # Pass obs (2D) through model
        action = np.random.choice(actions, p=probs[0].numpy())
        y_target = tf.constant([[0 if i != action else 1 for i in actions]]) # onehot encoding
  
        loss = tf.reduce_mean(loss_fn(y_target, probs)) # Compute loss
    
    grads = tape.gradient(loss, model.trainable_variables) # Compute gradient
    obs, reward, done, info = env.step(action)
    return obs, reward, done, grads

def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    # To store rewards and gradients for all epiosdes
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        # To store rewards and gradients for all steps in an episode
        current_rewards = []
        current_grads = []
    
        obs = env.reset()
        for step in range(n_max_steps):
            # Compute reward and gradient for one step
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    # Return the list of reward lists and list of gradient lists
    return all_rewards, all_grads

def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards)-2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards) # Combine to a single vector
    rewards_mean = np.mean(flat_rewards)
    rewards_std = np.std(flat_rewards)
    return [(discounted_rewards - rewards_mean) / rewards_std for discounted_rewards in all_discounted_rewards]

def nn_policy(obs):
    probs = model.predict(obs[np.newaxis])[0]
    action = np.random.choice([0,1,2], p=probs)
    return action

if __name__ == "__main__": 
    n_iterations = 50
    n_episodes_per_update = 5
    n_max_steps = 250
    discount_factor = 0.95 # reward is half ~14 steps into the future math.log(0.5, 0.95) = 13.5
    
    optimizer = keras.optimizers.Adam(lr=0.03)
    loss_fn = keras.losses.categorical_crossentropy
    
    env = gym.make("MountainCar-v0")
    
    for iteration in range(n_iterations):
        all_rewards, all_grads = play_multiple_episodes(
            env, n_episodes_per_update, n_max_steps, model, loss_fn)
        total_rewards = sum(map(sum, all_rewards))
        print("\rIteration: {}, mean rewards: {:.1f}".format(  
            iteration, total_rewards / n_episodes_per_update), end="")
        all_final_rewards = discount_and_normalize_rewards(all_rewards,
                                                           discount_factor)
        all_mean_grads = []
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [final_reward * all_grads[episode_index][step][var_index]
                 for episode_index, final_rewards in enumerate(all_final_rewards)
                     for step, final_reward in enumerate(final_rewards)], axis=0)
            all_mean_grads.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
    
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