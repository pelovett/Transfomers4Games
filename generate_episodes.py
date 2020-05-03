import gym
import torch
import torch.nn as nn
import numpy as np
from math import floor

def generateEpisodes(gym_name='CartPole-v0', score_threshold=50, write_2_disc=False):
    env = gym.make(gym_name)
    scores = []
    training_data = []
    for i_episode in range(10**4):
        observation = env.reset()
        game_memory = [[], []]
        prev_observation = []
        total = 0
        for t in range(500):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            total += reward
            if len(prev_observation) > 0:
                game_memory[0].append(prev_observation)
                game_memory[1].append(action)
            prev_observation = observation
            if done: 
                break
        #if total > score_threshold:
        scores.append(total)
        game_memory[1] = np.array(game_memory[1])
        game_memory[0] = np.array(game_memory[0])
        training_data.append((game_memory, total))
        env.reset()
    env.close()
    training_data.sort(key=lambda x:x[1])
    training_data = training_data[floor(.9*len(training_data)):]
    print(f'Recorded {len(training_data)} episodes')
    print(f'Average accepted score: {np.mean(scores)}')
     
    if write_2_disc:
        print(f'Saving data to file: saved_episodes.npy ...')
        np.save('saved_episodes.npy', training_data)
    print()
    return training_data


if __name__ == "__main__":
    generateEpisodes(write_2_disc=True)
