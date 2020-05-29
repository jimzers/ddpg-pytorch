from ddpg_agent import Agent
import gym
import numpy as np
import matplotlib.pyplot as plt


def plot_curve(x, scores, taus):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2")

    # this is the tau plot
    ax.plot(x, taus)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Tau")
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
    plt.show()


env = gym.make('LunarLanderContinuous-v2')

agent = Agent(alpha=0.000025, beta = 0.00025, input_dims=[8], tau=0.001, env=env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

episodes = 1000

np.random.seed(42)

tau_hist = []
score_hist = []
for i in range(episodes):
    done = False
    score = 0
    state = env.reset()
    while not done:
        act = agent.choose_action(state)
        next_state, reward, done, _ = env.step(act)
        agent.store(state, act, reward, next_state, int(done))
        agent.learn()
        score += reward
        state = next_state

    agent.save_models()
    score_hist.append(score)
    tau_hist.append(agent.tau)
    avg_score = np.mean(score_hist[-100:])
    print('episode ' + str(i + 1) + 'score %.2f' % score +
              'average score %.2f' % avg_score)

episodes = np.arange(1, episodes + 1)
plot_curve(episodes, score_hist, tau_hist)
