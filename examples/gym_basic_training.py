from collections import defaultdict
import functools
import operator as op
from typing import assert_never

import gymnasium as gym
from gymnasium import Env
from matplotlib import pyplot as plt
import numpy as np
from tqdm import trange


def starfilter(f, xs):
    return filter(lambda x: f(*x), xs)


def fdict(d, ks):
    return {k: d[k] for k in ks}


def make_dispatcher(env):
    def dispatcher(name):
        return env[name]

    for var, meaning in starfilter(lambda k, v: not k.startswith("_"), env.items()):
        setattr(dispatcher, var, meaning)

    return dispatcher


def policy(obs, qs, eps, pred, act, rand):
    if pred(eps):
        return rand(obs, qs)
    else:
        return act(obs, qs)


def policy_eps_greedy(obs, qs, eps, act, rand):
    return policy(obs, qs, eps, lambda eps: np.random.rand() < eps, act, rand)


def decay(eps: float, dec: float, flr: float):
    return max(flr, eps - dec)


def loss(x2, x1, bias=0):
    return (x2 - x1) + bias


def loss_td(q_new, q_old, reward, discount_factor):
    return loss(discount_factor * q_new, q_old, reward)


def opt(x, obj, lr):
    return x - lr * obj


def opt_td(q, td, lr):
    return opt(q, td, -lr)


def repeatedly(fn, it):
    return [fn(i) for i in (range(it) if isinstance(it, int) else it)]


def step(obs, env, agent):
    action = agent('get_action')(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    agent('update')(obs, action, reward, terminated, next_obs)

    return next_obs, terminated or truncated, info


def play(env, agent):
    obs, _ = env.reset()
    done = False
    while not done:
        obs, done, _ = step(obs, env, agent)


def train_step(env, agent):
    play(env, agent)
    agent("decay_epsilon")()


def train(env, agent, n_episodes):
    repeatedly(lambda _: train_step(env, agent), trange(n_episodes))


def visualize(env, agent):
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))

    axs[0].plot(np.convolve(env.return_queue, np.ones(100)))
    axs[0].set_title("Episode Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")

    axs[1].plot(np.convolve(env.length_queue, np.ones(100)))
    axs[1].set_title("Episode Length")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Length")

    axs[2].plot(np.convolve(agent('training_error'), np.ones(100)))
    axs[2].set_title("Training Error")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Temporal Difference")

    plt.tight_layout()
    plt.show()


def pprint_env(env, n=0):
    name = env.spec.id
    print(f"-- {name} --")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    if hasattr(env, "reward_range"):
        print(f"Reward Range: {env.reward_range}")
    print(f"Metadata: {env.metadata}")
    print(f"Spec: {env.spec}")

    for _ in range(n):
        print()


def make_blackjack_agent(_env, _lr, _eps, _eps_decay, _final_eps, _discount_factor=0.95):
    _qs = defaultdict(lambda: np.zeros(_env.action_space.n))
    training_error = []

    def get_action(obs):
        return policy_eps_greedy(
            obs,
            _qs,
            _eps,
            lambda obs, qs: int(np.argmax(qs[obs])),
            lambda obs, qs: _env.action_space.sample(),
        )

    def update(obs, action, reward, terminated, next_obs):
        q_new = (not terminated) * np.max(_qs[next_obs])
        q_old = _qs[obs][action]

        l = loss_td(q_new, q_old, reward, _discount_factor)
        _qs[obs][action] = opt_td(q_old, l, _lr)

        training_error.append(l)

    def decay_epsilon():
        nonlocal _eps
        _eps = decay(_eps, _eps_decay, _final_eps)

    def dispatch(symbol):
        if symbol == "get_action":
            return get_action
        elif symbol == "update":
            return update
        elif symbol == "decay_epsilon":
            return decay_epsilon
        elif symbol == "training_error":
            return training_error
        else:
            assert_never(symbol)
    return dispatch


def main():
    learning_rate = 0.01
    n_episodes = 100_000
    initial_epsilon = 1.0
    epsilon_decay = initial_epsilon / (n_episodes / 2)  # 0.00002
    final_epsilon = 0.1

    env = gym.make("Blackjack-v1", sab=False)

    agent = make_blackjack_agent(
        env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon
    )

    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    train(env, agent, n_episodes)
    visualize(env, agent)


if __name__ == "__main__":
    main()
