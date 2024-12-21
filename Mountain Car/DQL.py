
# Gerekli kütüphanelerin yüklenmesi
import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
import random
import matplotlib.pyplot as plt
from matplotlib import rc
from IPython.display import HTML
import matplotlib.animation as animation
from IPython.display import display



# Cuda varsa GPU kullanımı, yoksa CPU kullanımı
torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Q-Ağının (Deep Q-Network) modelinin tanımlanması
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Girdi boyutuna göre 3 katmanlı tam bağlantılı bir ağ
        self.fc1 = nn.Linear(input_dim, 256)  # İlk katman
        self.fc2 = nn.Linear(256, 128)        # İkinci katman
        self.fc3 = nn.Linear(128, output_dim) # Çıkış katmanı
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = F.relu(self.fc1(x.to(self.device)))  # İlk katman
        x = F.relu(self.fc2(x.to(self.device)))  # İkinci katman
        x = self.fc3(x.to(self.device))         # Çıkış katmanı
        return x



# Deneyim Belleği (Replay Buffer)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # Yeni bir deneyimi ekleme
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # Deneyimlerden rastgele bir örnek seçme
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    # Bellekteki toplam deneyim sayısını alma
    def __len__(self):
        return len(self.buffer)


# DQN Ajanı
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim  # Durum boyutu
        self.action_dim = action_dim  # Eylem boyutu
        self.gamma = gamma  # Gelecekteki ödüllerin indirim oranı
        self.epsilon = epsilon  # Keşif oranı (epsilon-greedy)
        self.epsilon_decay = epsilon_decay  # Epsilon değerinin azalması
        self.epsilon_min = epsilon_min  # Epsilon için minimum değer
        self.replay_buffer = ReplayBuffer(20000)  # Deneyim belleği
        self.batch_size = 64  # Eğitim için minibatch boyutu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Cuda desteği

        # Model ve hedef modelin oluşturulması
        self.model = DQN(state_dim, action_dim)  # Q ağı
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.target_model = DQN(state_dim, action_dim)  # Hedef Q ağı
        self.target_model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = Adam(self.model.parameters(), lr=lr)  # Adam optimizasyonu
        self.update_target_model()

    # Hedef modelini güncelleme
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Eylem seçimi
    def select_action(self, state):
        # Eğer keşif yapma oranı (epsilon) ile seçim yapılıyorsa, rastgele eylem seç
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Durumu tensöre dönüştür
        with torch.no_grad():
            q_values = self.model(state)  # Modelden Q-değerlerini al
        return q_values.argmax().item()  # En yüksek Q-değerine sahip eylemi seç

    # Eğitim işlemi
    def train(self):
        # Eğer yeterli deneyim yoksa eğitim yapılmaz
        if len(self.replay_buffer) < self.batch_size:
            return

        # Deneyimlerden rastgele bir minibatch seçme
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Q-değerlerinin hesaplanması
        q_values = self.model(state)
        next_q_values = self.target_model(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        # Kayıp fonksiyonu (Mean Squared Error)
        loss = F.mse_loss(q_value, expected_q_value.detach())

        # Geri yayılım (backpropagation) adımları
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon değeri zamanla azalır
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



# Ajanı eğitme
env = gym.make("MountainCar-v0")  # MountainCar ortamını oluşturma
state_dim = env.observation_space.shape[0]  # Durum boyutu
action_dim = env.action_space.n  # Eylem sayısı
agent = DQNAgent(state_dim, action_dim)  # DQN ajanı

episodes = 5000  # Ajanı 5000 bölüm boyunca eğit
episode_rewards = []  # Ödülleri saklama

for episode in range(episodes):
    state, _ = env.reset()  # Başlangıç durumu
    episode_reward = 0  # Bölüm ödülü
    done = False  # Bölümün bitip bitmediğini kontrol et

    while not done:
        action = agent.select_action(state)  # Ajanın eylemi seçmesi
        next_state, reward, done, trunc, _ = env.step(action)  # Ortamdan çıktı al
        done = done or trunc  # Bitiş durumu kontrolü
        agent.replay_buffer.push(state, action, reward, next_state, done)  # Deneyimi belleğe ekle
        state = next_state  # Durumu güncelle
        episode_reward += reward  # Ödülü güncelle

        agent.train()  # Ajanı eğit

    episode_rewards.append(episode_reward)  # Bölüm ödülünü listeye ekle
    agent.update_target_model()  # Hedef modelini güncelle
    print(f"Episode {episode + 1}: {episode_reward}")  # Bölüm sonucunu yazdır

# Ödül grafiğini çizme
plt.plot(episode_rewards)
plt.xlabel("Bölümler")
plt.ylabel("Ödüller")
plt.show()

# Video kaydı ve görüntüleme (grafiksel render)
env = gym.make("MountainCar-v0", render_mode="rgb_array")
frames = []
state, _ = env.reset()
done = False
while not done:
    action = agent.select_action(state)
    next_state, reward, done, trunc, _ = env.step(action)
    frames.append(env.render())
    state = next_state

# Video animasyonu oluşturma
fig = plt.figure()
ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
display(HTML(ani.to_html5_video()))

