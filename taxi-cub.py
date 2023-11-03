import gym
import time
import numpy as np
import copy

gamma = 0.6
alpha = 0.95
epsilon = 0.1
epsilon_decay = 0.99 #decay factor 

total_epochs =0
episodes = 10000

env = gym.make('Taxi-v3',render_mode='ansi')
Q = np.zeros([env.observation_space.n, env.action_space.n])
state = env.reset()
saved_state = state[0]

#TRAINING
for episode in range(episodes):
    epochs = 0
    reward = 0
    state = env.reset()
    state = state[0]
    epsilon = epsilon * epsilon_decay #decay step

    while reward != 20:           #while dropoff state has not been reached
        

        if np.random.rand() < epsilon:
            #exploration option
            action = env.action_space.sample() 
        else:
            #exploitation option
            action = np.argmax(Q[state])

        #obtain reward and next state resulting from taking action
        next_state, reward, done, info, extra_value = env.step(action)

        #update Q-value for state-action pair
        Q[state, action] = Q[state, action] + alpha * (reward + gamma *   \
                            np.max(Q[next_state]) - Q[state, action])
        #update state
        state = next_state
        epochs+=1
    total_epochs += epochs
print("Average timesteps taken: {}".format(total_epochs/episodes))

reward = 0
env = gym.make('Taxi-v3',render_mode='human')
state = env.reset()
state= state[0]
counter = 0
# PERFORMANCE
while reward != 20:
        action = np.argmax(Q[state])
        #obtain reward and next state resulting from taking action
        next_state, reward, done, info, extra_value = env.step(action)
        #update state
        state = next_state
        print(f"State: {state} - Action: {action} - Reward: {reward}")
        counter+=1

print("Performance: {}".format(counter))        
env.render()
env.close()