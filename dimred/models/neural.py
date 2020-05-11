import copy
from typing import Iterable, Callable

import torch
from torch import nn


class AbstractAE(nn.Module):
    def __init__(self, hidden_sizes: Iterable, activation_function: Callable = nn.ReLU(),
                 out_function: Callable = nn.Sigmoid()):
        super().__init__()
        self.encoder_hidden_sizes = []
        self.decoder_hidden_sizes = []
        self.activation_function = activation_function
        self.out_function = out_function
        self.init_sizes(hidden_sizes)
        self.encoder = nn.Sequential(*self.init_module(self.encoder_hidden_sizes, self.activation_function,
                                                       self.out_function, reverse=False))
        self.decoder = nn.Sequential(*self.init_module(self.decoder_hidden_sizes, self.activation_function,
                                                       self.out_function, reverse=True))

    def init_sizes(self, hidden_sizes):
        self.encoder_hidden_sizes = copy.deepcopy(hidden_sizes)
        self.decoder_hidden_sizes = copy.deepcopy(hidden_sizes)

    def init_module(self, hidden_sizes: Iterable, activation_function: Callable, out_activation: Callable,
                    reverse: bool):
        if reverse:
            hidden_sizes = list(reversed(hidden_sizes))
        layers = self.init_layers(hidden_sizes, activation_function)
        if reverse:
            layers[-1] = out_activation
        return layers

    @staticmethod
    def init_layers(hidden_sizes: Iterable, activation_function: Callable):
        layers = []
        for i, (l_in, l_out) in enumerate(zip(hidden_sizes, hidden_sizes[1:])):
            layers.append(nn.Linear(l_in, l_out))
            layers.append(activation_function)

        return layers


class AE(AbstractAE):
    def forward(self, x):
        z = self.encoder(x)
        xHat = self.decoder(z)

        return xHat, z


class VAE(AbstractAE):
    def init_sizes(self, hidden_sizes):
        super().init_sizes(hidden_sizes)
        self.encoder_hidden_sizes[-1] *= 2

    @staticmethod
    def _reparametrize(mu, logvar):
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
