import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # lets you call on the class
        # noise = OUActionNoise()
        # noise()
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * \
                self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_mem = np.zeros((self.mem_size, *input_shape))
        self.new_state_mem = np.zeros((self.mem_size, *input_shape))
        self.action_mem = np.zeros((self.mem_size, n_actions))
        self.reward_mem = np.zeros(self.mem_size)
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        idx = self.mem_cntr % self.mem_size # wrap around
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.new_state_mem[idx] = new_state
        self.terminal_mem[idx] = 1 - done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size) # get batch indices

        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        new_states  = self.new_state_mem[batch]
        done = self.terminal_mem[batch]

        return states, actions, rewards, new_states, done

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super().__init__()

        self.beta = beta
        self.n_actions = n_actions
        self.filename = os.path.join(chkpt_dir, name)

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        # init the values of the weights of first lin connected layer
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        # normalization layer
        self.norm1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # init the values of the weights
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        # normalization layer
        self.norm2 = nn.LayerNorm(self.fc2_dims)

        self.action_val = nn.Linear(self.n_actions, fc2_dims)
        f3 = 0.003 # idk why this is constant LOL
        self.q = nn.Linear(self.fc2_dims, 1)
        # do the weight initialization
        torch.nn.init.uniform_(self.q.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # LOL good luck training in CPU HAHAHHAhA

        self.to(self.device) # send off to GPU. or cpu if u wanna finish by 22nd century

    def forward(self, state, action):
        x = self.fc1(state)
        x = self.norm1(x)
        x = F.relu(x) # relu everything AFTER batch norming... so you don't chop off the negative batch norms

        x = self.fc2(x)
        x = self.norm2(x)

        # x => state val

        action_val = self.action_val(action) # option to relu first before feeding in...
        state_action_val = F.relu(torch.add(x, action_val))
        state_action_val = self.q(state_action_val)

        return state_action_val

    def save_checkpoint(self):
        print('------ save chkpt -----------')
        torch.save(self.state_dict(), self.filename)

    def load_checkpoint(self):
        print('------------ loading chkpt ----------------')
        self.load_state_dict(torch.load(self.filename))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super().__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.filename = os.path.join(chkpt_dir, name+'_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.norm1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.norm2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = torch.tanh(self.mu(x))

        return x

    def save_checkpoint(self):
        print('----- saving checkpoint -------')
        torch.save(self.state_dict(), self.filename)

    def load_checkpoint(self):
        print('-------- loading checkpoint --------')
        self.load_state_dict(torch.load(self.filename))

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2, max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        # add clipping values for high and low
        self.low = env.action_space.low
        self.high = env.action_space.high

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='Actor')
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='TargetActor')

        self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='Critic')
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='TargetCritic')

        self.noise = OUActionNoise(mu = np.zeros(n_actions))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval() # set network to evaluation mode, but not train mode yet
        observation = torch.tensor(observation, dtype=torch.float).to(self.actor.device) # send the observation to GPU
        # get action from actor
        mu = self.actor(observation).to(self.actor.device)
        mu_prime = mu + torch.tensor(self.noise(), dtype = torch.float).to(self.actor.device)
        self.actor.train()
        # detach values
        mu_prime = mu_prime.cpu().detach().numpy() # LOL get numpy value in cpu mode to give to gym
        #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        #print(mu_prime.shape)
        return np.clip(mu_prime, self.low, self.high) # clip the action


    def store(self, state, action, reward, state_new, done):
        self.memory.store_transition(state, action, reward, state_new, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return # don't learn if the replay buffer doesn't have enough memories

        state, action, reward, state_new, done = self.memory.sample_buffer(self.batch_size)
        #send off the stuff to the gpu
        state = torch.tensor(state, dtype=torch.float).to(self.critic.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic.device)
        state_new = torch.tensor(state_new, dtype=torch.float).to(self.critic.device)
        done = torch.tensor(done, dtype=torch.float).to(self.critic.device)

        # set networks to eval mode
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        # get target actions to use on target critic network
        target_actions = self.target_actor.forward(state_new)
        critic_val_new = self.target_critic.forward(state_new, target_actions)
        critic_val = self.critic.forward(state, action)

        target = []
        # target stores all the y_i stuff
        for j in range(self.batch_size):
            # this is y_i
            target.append(reward[j] + self.gamma * critic_val_new[j] * done[j])
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1) # resize to shape (batch_size, 1)

        self.critic.train() # NOW we set to training mode
        self.critic.optimizer.zero_grad() # zero the gradients
        critic_loss = F.mse_loss(target, critic_val)
        critic_loss.backward() # backprop the critic loss woo hoo
        self.critic.optimizer.step()

        self.critic.eval() # freeze the weights again

        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()

        policy_loss = -self.critic.forward(state, mu).mean() # do mean b/c batch stuff

        policy_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()


