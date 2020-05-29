# -*- coding: utf-8 -*-
"""
Created on Thu May 28 17:26:34 2020

Author: David O'Callaghan
"""

import time
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

n_inputs = 4 # == env.observation_space.shape[0]

model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid"),
])

def play_one_step(env, obs, model, loss_fn):
    
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis]) # Pass obs (2D) through model
        
        rand = tf.random.uniform([1,1]) # Random float between 0 and 1 in a 1x1 tensor
        action = rand > left_proba
        action = tf.cast(action, tf.float32) # 0.0 (Left) if False, 1.0 (Right) if True
        y_target = tf.constant([[1.]]) - action # 1.0 if Left, 0.0 if Right
        
        loss = tf.reduce_mean(loss_fn(y_target, left_proba)) # Compute loss
    
    grads = tape.gradient(loss, model.trainable_variables) # Compute gradient
    a = int(action[0,0].numpy()) # Convert action to int
    obs, reward, done, info = env.step(a)
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
    left_proba = model.predict(obs.reshape(1, -1))
    action = int(0.5 > left_proba)
    return action

if __name__ == "__main__":
    n_iterations = 50
    n_episodes_per_update = 5
    n_max_steps = 250
    discount_factor = 0.95 # reward is half ~14 steps into the future math.log(0.5, 0.95) = 13.5
    
    optimizer = keras.optimizers.Adam(lr=0.03)
    loss_fn = keras.losses.binary_crossentropy
    
    env = gym.make("CartPole-v1")
    
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