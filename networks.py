import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# actor
class Policy_net(nn.Module):
    def __init__(self, input_dim, output_dim, action_bounds):
        super(Policy_net, self).__init__()
        self.action_bounds = action_bounds
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, output_dim)
        self.log_std = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = torch.distributions.Normal(mu, std)
        return dist
    
    def sample_likelihood(self, states):
        dist = self(states)
        # Reparameterization trick
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(value=u)
        # Enforcing action bounds
        log_prob -= torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return (action * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1]), log_prob

# critic
class Val_net(nn.Module):
    def __init__(self, input_dim):
        super(Val_net, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)
        

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class Qval_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Qval_net, self).__init__()
        self.layer1 = nn.Linear(input_dim + output_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)
        

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
