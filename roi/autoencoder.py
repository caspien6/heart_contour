import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, n_inp, n_hidden):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(n_inp, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_inp)
        self.n_inp = n_inp
        self.n_hidden = n_hidden

    def forward(self, x):
        encoded = F.sigmoid(self.encoder(x))
        decoded = F.sigmoid(self.decoder(encoded))
        return encoded, decoded
