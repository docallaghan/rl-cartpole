# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:16:33 2020

@author: ocall
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

def nn_policy(obs):
    left_proba = model.predict(obs.reshape(1, -1))
    action = int(np.random.rand() > left_proba)
    return action

if __name__ == "__main__":
    # Make 50 separate CartPole environments for training
    n_environments = 50
    n_iterations = 5000
    envs = [gym.make("CartPole-v1") for _ in range(n_environments)]
    for index, env in enumerate(envs):
        env.seed(index)
    np.random.seed(42)
    observations = [env.reset() for env in envs]
    
    optimizer = keras.optimizers.RMSprop()
    loss_fn = keras.losses.binary_crossentropy
    
    # Train the network 
    for iteration in range(n_iterations):
        # if angle < 0, we want proba(left) = 1., or else proba(left) = 0.
        target_probas = np.array([([1.] if obs[2] < 0 else [0.])
                                  for obs in observations])
        with tf.GradientTape() as tape:
            left_probas = model(np.array(observations))
            loss = tf.reduce_mean(loss_fn(target_probas, left_probas))
        print("\rIteration: {}, Loss: {:.3f}".format(iteration, loss.numpy()), end="")
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        actions = (np.random.rand(n_environments, 1) > left_probas.numpy()).astype(np.int32)
        
        for env_index, env in enumerate(envs):
            obs, reward, done, info = env.step(actions[env_index][0])
            observations[env_index] = obs if not done else env.reset()
    for env in envs:
        env.close()
    
    # Run an episode
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
            print(f'\nFailed after {i} time steps')
            env.close()
            break