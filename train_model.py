import time
from math import floor

import torch
import torch.nn as nn
import numpy as np
import gym

from transformer_model import TransformerModel
from basic_rnn import RNN
from generate_episodes import generateEpisodes

np.random.seed(123)
torch.manual_seed(123)

NUM_EPOCH = 20
HIDDEN_SIZE = 100
NHEAD = 5
LEARNING_RATE = 0.001
GRADIENT_CLIP = 5.0
GYM_ENVIRONMENT = 'CartPole-v0'
SAVE_NAME = 'rnn_save_state_50.torch'
BATCH_SIZE = 2

env = gym.make(GYM_ENVIRONMENT)
obs_size = env.observation_space.shape[0]
model = RNN(obs_size, HIDDEN_SIZE, obs_size)
#model = TransformerModel(obs_size, obs_size, NHEAD, HIDDEN_SIZE, nlayers=4, dropout=0)
model.double()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

all_data = np.load('saved_episodes.npy', allow_pickle=True)
train_split = floor(all_data.shape[0]*.8)

mask = torch.zeros((len(all_data), 200, env.observation_space.shape[0]))
padded_data = []
lengths = []
for i in range(len(all_data)):
    data = np.array(all_data[i])
    temp = np.full((200, env.observation_space.shape[0]), -999, dtype='float64')
    temp[:len(data), :] = data
    padded_data.append(temp)
    mask[i, :len(data), :] = torch.ones((len(data), env.observation_space.shape[0]))
    lengths.append(len(all_data[i]))
lengths = torch.tensor(lengths)
training_data = padded_data[:train_split]
test_data = padded_data[train_split:]


for _ in range(NUM_EPOCH):
    localtime = time.asctime(time.localtime(time.time()))
    print(f'Starting epoch {1+_} at {localtime}')
    model.train()
    
    epoch_loss = 0
    indices = np.random.permutation(train_split)
    for i in range(floor(len(indices)/BATCH_SIZE)): 
        batch_indices = indices[i:i+BATCH_SIZE]   
        episode = torch.tensor(np.take(training_data, batch_indices, 0))        

        optimizer.zero_grad()
        out = model(episode, lengths[batch_indices])
        #out = model(episode)
        batch_mask = mask[batch_indices, :out.shape[1], :]
        target = episode[:, :out.shape[1], :]
        masked_out = (out * batch_mask)[:, :-1, :]
        target =  (target * batch_mask)[:, 1:, :]
        loss = loss_fn(masked_out, target)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        epoch_loss += loss.detach()
        optimizer.step()
    
    model.eval()
    total_score = 0
    for i, sample in enumerate(test_data): 
        episode = torch.tensor(sample).unsqueeze(0)
        single_length = torch.tensor([lengths[i]])
        out = model(episode, single_length)
        #out = model(episode)
        batch_mask = mask[i, :out.shape[1], :]
        target = episode[:, :out.shape[1], :]
        masked_out = (out * batch_mask)[:, :-1, :]
        target =  (target * batch_mask)[:, 1:, :]
        loss = loss_fn(masked_out, target)
        total_score += loss.detach()
    print(f'Finished epoch {1+_} at {time.asctime(time.localtime(time.time()))}')
    print(f'  Train Loss: {epoch_loss/len(training_data)}')
    print(f'  Test  Loss: {total_score/len(test_data)}\n')

model.eval()
total_score = 0
for i, sample in enumerate(test_data): 
    episode = torch.tensor(sample).unsqueeze(0)
    single_length = torch.tensor([lengths[i]])
    out = model(episode, single_length)
    #out = model(episode)
    batch_mask = mask[i, :out.shape[1], :]
    target = episode[:, :out.shape[1], :]
    masked_out = (out * batch_mask)[:, :-1, :]
    target =  (target * batch_mask)[:, 1:, :]
    loss = loss_fn(masked_out, target)
    total_score += loss.detach()
print(f'Finished evaluation.\nTest Loss: {total_score/len(test_data)}')


torch.save(model.state_dict(), SAVE_NAME)
print(f'Saved model under: {SAVE_NAME}')
