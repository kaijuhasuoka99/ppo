import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

to_np = lambda x: x.detach().cpu().numpy()

class Pi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(config.embed_dim, 512), nn.ReLU())
        self.pi_mean = nn.Linear(config.embed_dim, config.n_cns_action)
        self.pi_logvar = nn.Linear(config.embed_dim, config.n_cns_action)

    def forward(self, x):
        x = self.pi(x)
        mean = self.pi_mean(x)
        logvar = self.pi_logvar(x)
        std = torch.exp(0.5 * logvar)
        
        return mean, std # policy network outputs mean and standard deviation.
    
class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(config.n_obs, config.embed_dim), nn.ReLU())
        self.pi = Pi(config)
        self.v = nn.Sequential(nn.Linear(config.embed_dim, 512), nn.ReLU(),
                                nn.Linear(config.embed_dim, 1))

        self.device = config.device

    def forward(self, x):
        x = self.state_encoder(x)
        mu, sigma = self.pi(x)
        v = self.v(x).squeeze(-1) # (b, 1, 1) => (b, 1)

        return (mu, sigma), v
    
    def infer(self, state, batch=False):
        assert isinstance(state, np.ndarray)
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device)
            x = torch.arctan(x) # normalize
            if not batch:
                x = x.unsqueeze(0)
            (mu, sigma), v = self.forward(x)
            if not batch:
                mu = mu.squeeze(0)
                sigma = sigma.squeeze(0)
                v = v.squeeze(0)
            mu = to_np(mu)
            sigma = to_np(sigma)
            v = to_np(v)
            return (mu, sigma), v


class ActorCriticConfig:
    def __init__(self):
        self.embed_dim = 512
        self.n_cns_action = None
        self.n_obs = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    B, E = 64, 512
    O, A = 10, 5
    config = ActorCriticConfig()
    config.n_obs = O
    config.n_cns_action = A
    config.device = 'cpu'

    x = torch.randn(B, O)

    ac = ActorCritic(config)

    (mu, mean), v = ac(x)

    print(mu.shape, mean.shape, v.shape)

    x = np.random.randn(O)

    (mu, mean), v = ac.infer(x)
    print(mu, mean, v)