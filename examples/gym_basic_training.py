from collections import defaultdict
import functools
from functools import reduce
import operator as op
from typing import assert_never

import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import RecordEpisodeStatistics
from matplotlib import pyplot as plt
import numpy as np
from tqdm import trange


def starfilter(f, xs):
    return filter(lambda x: f(*x), xs)


def foreach(f, xs):
    [f(*x) for x in xs]


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


def repeat(fn, it):
    return [fn() for i in (range(it) if isinstance(it, int) else it)]


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
    repeat(lambda: train_step(env, agent), trange(n_episodes))


def plot(ax, xs, title=None, xlabel=None, ylabel=None):
    ax.plot(xs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def viz1(ax, xs):
    plot(ax, *(xs if isinstance(xs, tuple) else (xs,)))


def comp(*fns):
    def comp2(f, g):
        return lambda *a, **kw: f(g(*a, **kw))
    return reduce(comp2, fns)


def apply(f, *args, **kwargs):
    return f(*args, **kwargs)


def applyn(f, x, n):
    assert n >= 0
    return comp(*([f] * n))(x) if n > 0 else x


def wrap(x, t=list):
    if t is list:
        return [x]
    elif t is tuple:
        return (x,)
    elif t is np.ndarray:
        return np.array(wrap(x, list))
    else:
        assert_never(t)


# Note: assumes all elements have the same shape
def nesting(xs):
    if isinstance(xs, (list, tuple, np.ndarray)):
        return 1 + nesting(xs[0])
    else:
        return 0


def nest(xss, level=1):
    return applyn(wrap, xss, n=max(0, level - nesting(xss)))


def subplots(nrows, ncols, *args, **kwargs):
    fig, axs = plt.subplots(nrows, ncols, *args, **kwargs)
    return fig, nest(axs, level=1)


def _viz(xss, figsize=(12, 8)):
    _, axs = subplots(1, len(xss), figsize=figsize)
    foreach(viz1, zip(axs, xss))
    plt.tight_layout()
    plt.show()


def viz(xss, figsize=(12, 8)):
    return _viz(nest(xss, 2), figsize)


def rsum(xs, w=100):
    return np.convolve(xs, np.ones(w)) if w != 1 else xs


def viz_rl(env, agent, window=100):
    ms = map(rsum ,[env.return_queue, env.length_queue, agent("training_error")])
    ls = [("Episode Rewards", "Episode", "Reward"), 
          ("Episode Length", "Episode", "Length"), 
          ("Training Error", "Episode", "Temporal Difference")]

    viz([(m, *l) for m, l in zip(ms, ls)])


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
    learning_rate = 1e-2
    n_episodes = int(1e5)
    initial_epsilon = 1.0
    epsilon_decay = initial_epsilon / (n_episodes / 2)  # 0.00002
    final_epsilon = 0.1

    env = gym.make("Blackjack-v1", sab=False)
    env = RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = make_blackjack_agent(
        env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon
    )

    train(env, agent, n_episodes)
    viz_rl(env, agent)



if __name__ == "__main__":
    main()
