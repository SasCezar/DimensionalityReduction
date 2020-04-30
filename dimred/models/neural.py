import torch
from torch import nn


def _init_modules(hidden_sizes, activation_function, reverse):
    layers = []
    if reverse:
        hidden_sizes = list(reversed(hidden_sizes))
    for i, l_in, l_out in enumerate(zip(hidden_sizes, hidden_sizes[1:])):
        layers.append(nn.Linear(l_in, l_out))
        layers.append(activation_function[i])
    layers.append(activation_function[-1])
    return layers


class AE(nn.Module):
    def __init__(self, hidden_sizes, activation_functions):
        super().__init__()
        self.encoder = nn.Sequential(*_init_modules(hidden_sizes, activation_functions, reverse=False))
        self.decoder = nn.Sequential(*_init_modules(hidden_sizes, activation_functions, reverse=True))
        self.latent_size = hidden_sizes[-1]

    def forward(self, x):
        z = self.encoder(x)
        xHat = self.decoder(z)

        return xHat, z


class VAE(nn.Module):
    def __init__(self, hidden_sizes, activation_function=nn.ReLU):
        super().__init__()
        hidden_sizes_input = hidden_sizes
        hidden_sizes_input[-1] = hidden_sizes_input[-1] * 2
        self.encoder = nn.Sequential(*_init_modules(hidden_sizes_input, activation_function, reverse=False))
        self.decoder = nn.Sequential(*_init_modules(hidden_sizes, activation_function, reverse=True))
        self.latent_size = hidden_sizes[-1]

    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self._reparametrize(mu, logvar)
        xHat = self.decoder(z)

        return xHat, mu, logvar


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass