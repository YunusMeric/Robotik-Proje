import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
import matplotlib.pyplot as plt
import warnings

# Uyarıları bastırıyoruz
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hiperparametreler
parser = argparse.ArgumentParser(description='PyTorch Advantage Actor Critic Example')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='Discount factor (default: 0.98)')  # Discount factor'ı biraz daha düşük yapmak
parser.add_argument('--num_episodes', type=int, default=5000, metavar='NU',
                    help='Number of episodes (default: 1500)')  # Daha fazla bölümde eğitim yapmak
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='Random seed (default: 42)')  # Daha sağlam ve tekrarlanabilir sonuçlar için seed değeri
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='Interval between training status logs (default: 20)')  # Daha az sıklıkla durum loglaması yapmak
args = parser.parse_args()

# Ortamı başlatma
env = gym.make('MountainCar-v0')
state, _ = env.reset(seed=args.seed)  # Yeni Gym API'sine uygun hale getirildi
torch.manual_seed(args.seed)
num_inputs = 2
epsilon = 0.99
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Epsilon değeri hesaplama fonksiyonu
def epsilon_value(epsilon):
    eps = max(0.01, 0.995 * epsilon)  # Daha yavaş düşüş ile epsilon decay
    return eps

# Actor-Critic Ağı
class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()
        self.Linear1 = nn.Linear(num_inputs, 64)
        nn.init.xavier_uniform_(self.Linear1.weight)
        self.Linear2 = nn.Linear(64, 128)
        nn.init.xavier_uniform_(self.Linear2.weight)
        self.Linear3 = nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.Linear3.weight)
        num_actions = env.action_space.n

        self.actor_head = nn.Linear(64, num_actions)
        self.critic_head = nn.Linear(64, 1)
        nn.init.xavier_uniform_(self.critic_head.weight)
        self.action_history = []
        self.rewards_achieved = []

    def forward(self, state_inputs):
        x = F.relu(self.Linear1(state_inputs))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))
        return self.critic_head(x), x

    def act(self, state_inputs, eps):
        value, x = self(state_inputs)
        x = F.softmax(self.actor_head(x), dim=-1)
        m = Categorical(x)
        e_greedy = random.random()
        if e_greedy > eps:
            action = m.sample()
        else:
            action = m.sample((3,))  # Düzeltildi
            pick = random.randint(-1, 2)
            action = action[pick]
        return value, action, m.log_prob(action)


# Model, optimizer ve loss fonksiyonu
model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# Performans güncellemeleri
def perform_updates():
    r = 0
    saved_actions = model.action_history
    returns = []
    rewards = model.rewards_achieved
    policy_losses = []
    critic_losses = []

    for i in rewards[::-1]:
        r = args.gamma * r + i
        returns.insert(0, r)
    returns = torch.tensor(returns)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # Policy loss hesaplama
        policy_losses.append(-log_prob * advantage)

        # Value loss hesaplama
        critic_losses.append(F.mse_loss(value, torch.tensor([R])))
    optimizer.zero_grad()

    # Kümülatif kayıp hesaplama
    loss = torch.stack(policy_losses).sum() + torch.stack(critic_losses).sum()
    loss.backward()
    optimizer.step()
    
    # Action history ve rewards sıfırlanıyor
    del model.rewards_achieved[:]
    del model.action_history[:]
    return loss.item()


# Ana eğitim fonksiyonu
def main():
    eps = epsilon_value(epsilon)
    losses = []
    counters = []
    plot_rewards = []

    for i_episode in range(1, args.num_episodes + 1):
        counter = 0
        state, _ = env.reset(seed=args.seed)  # Yeni Gym API'sine uygun hale getirildi
        ep_reward = 0
        done = False

        while not done:

            # State'i unroll etme ve aksiyon alma
            state = torch.from_numpy(state).float()
            value, action, ac_log_prob = model.act(state, eps)
            model.action_history.append(SavedAction(ac_log_prob, value))
            # Ajan aksiyonu alır
            state, reward, terminated, truncated, _ = env.step(action.item())  # Dönüş değeri düzenlendi
            done = terminated or truncated  # `done` değişkeni oluşturuldu

            model.rewards_achieved.append(reward)
            ep_reward += reward
            counter += 1
            if counter % 5 == 0:
                loss = perform_updates()
            eps = epsilon_value(eps)

        # Reward ve bölüm kaydını yazdırma
        print(f"Episode: {i_episode}, Reward: {ep_reward}")

        # Kayıplar, zaman adımları ve ödülleri kaydetme
        if i_episode % args.log_interval == 0:
            losses.append(loss)
            counters.append(counter)
            plot_rewards.append(ep_reward)

    # Kayıp grafiği çizme
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.savefig('loss1.png')

    # Zaman adımları grafiği çizme
    plt.clf()
    plt.xlabel('Episodes')
    plt.ylabel('timesteps')
    plt.plot(counters)
    plt.savefig('timestep.png')

    # Toplam ödül grafiği çizme
    plt.clf()
    plt.xlabel('Episodes')
    plt.ylabel('rewards')
    plt.plot(plot_rewards)
    plt.savefig('rewards.png')


if __name__ == '__main__':
    main()
