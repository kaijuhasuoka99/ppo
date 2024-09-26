import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

to_np = lambda x: x.detach().cpu().numpy()

class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(), # (7, 7, 64)
                                 nn.Flatten(), nn.Linear(3136, config.embed_dim), nn.ReLU())

        self.pi = nn.Sequential(nn.Linear(config.embed_dim, 512), nn.GELU(),
                                nn.Linear(config.embed_dim, config.n_action), nn.Softmax(dim=-1))

        self.v = nn.Sequential(nn.Linear(config.embed_dim, 512), nn.GELU(),
                                nn.Linear(config.embed_dim, 1))
        self.device = config.device

    def forward(self, x):
        x = self.state_encoder(x)
        p = self.pi(x)
        v = self.v(x).squeeze(-1) # (b, 1, 1) => (b, 1)
        return p, v
    
    def infer(self, state, batch=False):
        assert isinstance(state, np.ndarray)
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device) / 255.0
            if not batch:
                x = x.unsqueeze(0)
            p, v = self.forward(x)
            if not batch:
                p = p.squeeze(0)
                v = v.squeeze(0)
            p = to_np(p)
            v = to_np(v)
            return p, v
        
class ActorCriticConfig:
    def __init__(self):
        self.embed_dim = 512
        self.n_action = 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

if __name__ == "__main__":
    config = ActorCriticConfig()
    config.device = 'cpu'
    B, E = 6, 512

    x = torch.rand(B, 1, 84, 84)

    ac = ActorCritic(config)

    p, v = ac(x)
    print(p.shape, v.shape)

    N = 12
    x = np.random.randint(0, 255, (N, 1, 84, 84))
    a = np.random.randint(0, config.n_action, (N,))
    p, v = ac.infer(x, batch=True)
    print(p.shape, v.shape)

    print('infer')
    x = np.random.randint(0, 255, (1, 84, 84))

    print(x.shape)
    for i in range(5):
        p, v = ac.infer(x)
        print(p.shape)


