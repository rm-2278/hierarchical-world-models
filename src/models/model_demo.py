import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class SimpleEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)
    

class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).reshape(z.size(0), 128, 8, 8)
        return self.deconv(h)

class SimpleActor(nn.Module):
    def __init__(self, latent_dim=128, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Agent:
    """Minimal agent: agent(obs_numpy) -> (action_numpy, recon_numpy)"""
    def __init__(self, device=None, latent_dim=128, action_dim=4):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SimpleEncoder(latent_dim).to(self.device)
        self.decoder = SimpleDecoder(latent_dim).to(self.device)
        self.actor = SimpleActor(latent_dim, action_dim).to(self.device)

    def save(self, path):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "actor": self.actor.state_dict(),
        }, path)

    @staticmethod
    def load(path, device=None, latent_dim=128, action_dim=4):
        device = device or torch.device("cpu")
        checkpoint = torch.load(path, map_location=device)
        agent = Agent(device=device, latent_dim=latent_dim, action_dim=action_dim)
        agent.encoder.load_state_dict(checkpoint["encoder"])
        agent.decoder.load_state_dict(checkpoint["decoder"])
        agent.actor.load_state_dict(checkpoint["actor"])
        return agent

    def __call__(self, obs: np.ndarray, eval: bool = True):
        was_uint8 = obs.dtype == np.uint8
        x = obs.astype(np.float32) / (255.0 if was_uint8 else 1.0)
        x = torch.as_tensor(x, device=self.device).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
        with torch.no_grad():
            z = self.encoder(x)
            action = self.actor(z)
            recon = self.decoder(z)
        action_np = action.squeeze(0).cpu().numpy()
        recon_np = recon.squeeze(0).permute(1,2,0).cpu().numpy()
        return action_np, recon_np