from PIL import Image
import gymnasium as gym
import gym_pacman
import time

env = gym.make('BerkeleyPacmanPO-v0')

# env.seed(1)
env.reset(seed=1)


done = False

while True:
    done = False
    env.reset()
    i = 0
    while i < 5:
        i += 1
        s_, r, done, info = env.step(env.action_space.sample())
        env.render()
        