from ddpg_torch import Agent
import gym
import numpy as np
import matplotlib.pyplot as plt


def plot_curve(x, scores, epsilons):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2")

    # this is the epsilon plot
    ax.plot(x, epsilons)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Epsilon")
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2.scatter(x, running_avg, color='C4')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color='C3')
