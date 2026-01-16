import torch
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
from models.dqn import DQN

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def act(self, state, exploit=False):
        if not exploit and random.random() < self.epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            q = self.model(torch.tensor(state).unsqueeze(0))
        return torch.argmax(q).item()

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for s, a, r, ns, d in batch:
            target = r
            if not d:
                target += self.gamma * torch.max(self.model(torch.tensor(ns).unsqueeze(0))).item()

            pred = self.model(torch.tensor(s).unsqueeze(0))
            target_f = pred.clone().detach()
            target_f[0][a] = target

            loss = self.loss_fn(pred, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="model.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="model.pth"):
        self.model.load_state_dict(torch.load(path))
