import time

import torch
import torch.nn as nn
import numpy as np
import gym

from basic_rnn import RNN
from generate_episodes import generateEpisodes

np.random.seed(123)

TEST_LENGTH = 200
NUM_TESTS = 50
NUM_EPOCH = 20
HIDDEN_SIZE = 50    
LEARNING_RATE = 0.003
GYM_ENVIRONMENT = 'Acrobot-v1'

training_data = generateEpisodes(GYM_ENVIRONMENT)
#training_data = np.load('saved_episodes.npy', allow_pickle=True)

env = gym.make(GYM_ENVIRONMENT)
model = RNN(env.observation_space.shape[0], HIDDEN_SIZE, env.action_space.n)
model.double()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.eval()
total_score = 0
for _ in range(NUM_TESTS):
    env.reset()
    score = 0 
    prev_obs = np.zeros(env.observation_space.shape[0])
    model.eval()
    hidden = model.initHidden()
    for t in range(TEST_LENGTH):
        prev_obs = torch.from_numpy(prev_obs).type(torch.DoubleTensor)
        action, hidden = model(prev_obs, hidden)
        prev_obs, reward, done, info = env.step(torch.argmax(action).item())
        score += reward
        #env.render()
        if done:
            break
    total_score += score
print(f'Average score pre-training: {(total_score/NUM_TESTS)}\n')


model.train()
for _ in range(NUM_EPOCH):
    localtime = time.asctime(time.localtime(time.time()))
    print(f'Starting epoch {1+_} at {localtime}')
    model.train()
    i = 0
    total_loss = 0
    for (episode, score) in np.random.permutation(training_data):
        observations = torch.from_numpy(episode[0])
        actions = torch.from_numpy(episode[1])
        estimations = []

        optimizer.zero_grad()
        hidden = model.initHidden()
        for obs in observations:
            out, hidden = model(obs, hidden)
            estimations.append(out.unsqueeze(0))
        final = torch.cat(estimations, dim=0)
        loss = loss_fn(final, actions)
        loss.backward()
        total_loss += loss.detach()
        optimizer.step()
        i += 1
        if i > 10:
            break
    print(f'Finished epoch {1+_} at {localtime}')
    print(f'  Total Loss: {total_loss/len(training_data)}')
    model.eval()
    total_score = 0
    for _ in range(NUM_TESTS):
        env.reset()
        score = 0 
        prev_obs = np.zeros(env.observation_space.shape[0])
        model.eval()
        hidden = model.initHidden()
        for t in range(TEST_LENGTH):
            prev_obs = torch.from_numpy(prev_obs).type(torch.DoubleTensor)
            action, hidden = model(prev_obs, hidden)
            prev_obs, reward, done, info = env.step(torch.argmax(action).item())
            score += reward
            #env.render()
            if done:
                break
        total_score += score
    print(f'  Average score: {total_score/NUM_TESTS}\n') 
    
model.eval()
total_score = 0
for _ in range(NUM_TESTS):
    env.reset()
    score = 0 
    prev_obs = np.zeros(env.observation_space.shape[0])
    model.eval()
    hidden = model.initHidden()
    for t in range(TEST_LENGTH):
        prev_obs = torch.from_numpy(prev_obs).type(torch.DoubleTensor)
        action, hidden = model(prev_obs, hidden)
        prev_obs, reward, done, info = env.step(torch.argmax(action).item())
        score += reward
        #env.render()
        if done:
            break
    total_score += score
print(f'Average score post-training: {(total_score/NUM_TESTS)}')

