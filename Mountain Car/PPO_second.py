import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt  # Grafik çizimi için eklendi

# Hiperparametreler
seed = 123
gamma = 0.99
learning_rate = 1e-3
num_episodes = 5000  # 5000 episode
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

# Performans verileri için liste
episode_rewards = []
average_rewards = []
policy_losses = []
value_losses = []

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
    total_reward = sum(rewards)
    episode_rewards.append(total_reward)

    if episode >= 100:
        avg_reward = np.mean(episode_rewards[-100:])
        average_rewards.append(avg_reward)
    else:
        average_rewards.append(total_reward)

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
    policy_losses.append(policy_loss.item())

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # Değer ağı güncelleme
    value_loss = nn.MSELoss()(values, returns)
    value_losses.append(value_loss.item())

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    # Bölüm ilerlemesini yazdır
    if episode % update_interval == 0:
        print(f"Episode {episode}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Total Reward: {total_reward:.2f}")

# Eğitim performansını görselleştir
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# Ortalama ödülleri çiz
ax[0].plot(average_rewards, label="Average Rewards (100 Episodes)")
ax[0].set_xlabel("Episode")
ax[0].set_ylabel("Average Reward")
ax[0].set_title("PPO Training - Average Rewards")
ax[0].legend()

# Kayıpları çiz
ax[1].plot(policy_losses, label="Policy Loss")
ax[1].plot(value_losses, label="Value Loss")
ax[1].set_xlabel("Episode")
ax[1].set_ylabel("Loss")
ax[1].set_title("PPO Training - Losses")
ax[1].legend()

plt.tight_layout()
plt.show()

env.close()
