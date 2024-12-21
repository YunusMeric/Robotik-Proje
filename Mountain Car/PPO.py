import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Hiperparametreler
seed = 123
gamma = 0.99
learning_rate = 1e-3
num_episodes = 2000
max_steps_per_episode = 200
batch_size = 64
update_interval = 10

# Ortamı başlat ve seed ayarla
env = gym.make("MountainCar-v0")
env.action_space.seed(seed)
env.observation_space.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# PPO için politika ağı
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# PPO için değer ağı
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Ağları ve optimizasyonu tanımla
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = PolicyNetwork(obs_dim, action_dim)
value_net = ValueNetwork(obs_dim)

policy_optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)

# PPO için kayıp fonksiyonu
def compute_ppo_loss(old_probs, new_probs, advantages, epsilon=0.2):
    ratio = new_probs / old_probs
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    return -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

# Eğitim döngüsü
for episode in range(num_episodes):
    obs, _ = env.reset()  # Yeni Gym API'sine uygun reset
    log_probs = []
    values = []
    rewards = []
    dones = []
    states = []
    actions = []  # Seçilen eylemleri saklamak için liste

    for t in range(max_steps_per_episode):  # Maksimum adım sayısı
        state = torch.tensor(np.array(obs), dtype=torch.float32)
        states.append(state)

        action_probs = policy_net(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()

        log_prob = action_dist.log_prob(action)
        log_probs.append(log_prob.unsqueeze(0))  # Boyut ekledik
        values.append(value_net(state))
        actions.append(action.item())  # Seçilen eylemi ekle

        obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        rewards.append(reward)
        dones.append(done)

        if done:
            break

    # Ödülleri hesapla
    returns = []
    R = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            R = 0
        R = reward + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.cat(values).squeeze()
    log_probs = torch.cat(log_probs)  # Artık sorun çıkmamalı
    actions = torch.tensor(actions, dtype=torch.long)  # Tam sayı tensor

    advantages = returns - values.detach()

    # Politika ağı güncelleme
    new_action_probs = policy_net(torch.stack(states))
    new_action_dist = torch.distributions.Categorical(new_action_probs)
    new_log_probs = new_action_dist.log_prob(actions)  # Eylemleri kullan

    policy_loss = compute_ppo_loss(
        old_probs=torch.exp(log_probs),
        new_probs=torch.exp(new_log_probs),
        advantages=advantages
    )

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # Değer ağı güncelleme
    value_loss = nn.MSELoss()(values, returns)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    # Bölüm ilerlemesini yazdır
    if episode % update_interval == 0:
        print(f"Episode {episode}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")

env.close()
