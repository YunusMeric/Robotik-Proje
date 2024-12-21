import gym

env = gym.make('MountainCar-v0')
print(f"Observation Space: {env.observation_space}")
print(f"Action Space: {env.action_space}")
