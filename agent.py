
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from networks import Val_net, Qval_net, Policy_net

class SAC():
    def __init__(self, env_name, state_dim, action_dim, memory, batch_size, gamma, alpha, lr, action_bound, reward_scale, device):
        self.cur_dir = os.path.dirname( os.path.abspath( __file__ ))
        self.net_save_dir = self.cur_dir + "/network_file/"
        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.action_bounds = action_bound
        self.reward_scale = reward_scale
        self.device = device
  
        #make networks
        self.policy_net = Policy_net(input_dim=self.state_dim, output_dim=self.action_dim, action_bounds = self.action_bounds).to(self.device)
        self.qval1_net = Qval_net(input_dim=self.state_dim, output_dim=self.action_dim).to(self.device)
        self.qval2_net = Qval_net(input_dim=self.state_dim, output_dim=self.action_dim).to(self.device)
        self.val_net = Val_net(input_dim=self.state_dim).to(self.device)
        self.val_target_net = Val_net(input_dim=self.state_dim).to(self.device)

        self.policy_net.train()
        self.qval1_net.train()
        self.qval2_net.train()
        self.val_net.train()

        self.val_target_net.load_state_dict(self.val_net.state_dict())
        self.val_target_net.eval()

        #loss
        self.value_loss = nn.MSELoss()
        self.qval_loss = nn.MSELoss()

        #optimizer
        self.policy_optim = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.qval1_optim = optim.Adam(self.qval1_net.parameters(), lr=self.lr)
        self.qval2_optim = optim.Adam(self.qval2_net.parameters(), lr=self.lr)
        self.val_optim = optim.Adam(self.val_net.parameters(), lr=self.lr)

    def push(self, state, action, reward, done, next_state):
        self.memory.push(state, action, reward, done, next_state)
    
    def train(self):
        if len(self.memory.buffer) < self.batch_size:
            return 0, 0, 0
        else:
            states, actions, rewards, dones, next_states = self.memory.sample(self.batch_size, self.state_dim, self.action_dim)
            # compute value target
            reparam_actions, log_probs = self.policy_net.sample_likelihood(states=states)
            q1 = self.qval1_net(states, reparam_actions)
            q2 = self.qval2_net(states, reparam_actions)
            q = torch.min(q1, q2)
            
            # V(s) = E[Q(s, a) - logπ(a|s)]
            target_val = q.detach() - self.alpha * log_probs.detach() 

            value = self.val_net(states)
            # J(ψ) = E[0.5 * (V_ψ(s) - E[Q_θ(s, a) - logπ_φ(a|s])**2]]
            value_loss = self.value_loss(value, target_val) 

            # calculate q-value target
            with torch.no_grad():
                # target Q(s, a) = r(s, a) + γE[V_ψ(s_t+1)]
                target_q = self.reward_scale * rewards + self.gamma * self.val_target_net(next_states) * (1 - dones)  
            q1 = self.qval1_net(states, actions)
            q2 = self.qval2_net(states, actions)

            # J_q(θ) = [.5 * (Q_θ(s, a) - target Q(s, a))**2]
            q1_loss = self.qval_loss(q1, target_q)
            q2_loss = self.qval_loss(q2, target_q)

            # J_π(φ) = E[logπ_φ(f_φ(ε;s)|s) - Q_θ(s, f_φ(ε;s))]
            policy_loss = (self.alpha * log_probs - q).mean() 

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            self.val_optim.zero_grad()
            value_loss.backward()
            self.val_optim.step()

            self.qval1_optim.zero_grad()
            q1_loss.backward()
            self.qval1_optim.step()

            self.qval2_optim.zero_grad()
            q2_loss.backward()
            self.qval2_optim.step()

            self.soft_update_target_net(self.val_net, self.val_target_net)
            return value_loss.item(), .5 * (q1_loss+q2_loss).item(), policy_loss.item()
        
    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = torch.from_numpy(states).float().to(self.device)
        action, _ = self.policy_net.sample_likelihood(states)
        return action.detach().cpu().numpy()[0]
    
    def soft_update_target_net(self, value_net, value_target_net):
        tau = 0.005
        for val_target_param, val_param in zip(value_target_net.parameters(), value_net.parameters()):
            val_target_param.data.copy_(tau * val_param.data + (1 - tau) * val_target_param.data)

    def save_network(self):
        self.createDirectory(self.net_save_dir)
        torch.save(self.policy_net.state_dict(), self.net_save_dir + "{}.pt".format(self.env_name))

    def load_network(self):
        self.policy_net.load_state_dict(torch.load(self.net_save_dir + str(self.env_name) + '.pt'))

    def eval_mode(self):
        self.policy_net.eval()

    def createDirectory(dir_name):
        try:
            if not os.path.exists(dir_name):
                print(f"Create {dir_name} direcoty")
                os.makedirs(dir_name)
        except OSError: print("Error: Failed to create the directory.")