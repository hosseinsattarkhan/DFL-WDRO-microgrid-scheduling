import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class TemporalBlock(nn.Module):
    """TCN block with causal convolution and residual connection."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = nn.ConstantPad1d((padding, -padding), 0) if padding > 0 else nn.Identity()
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = nn.ConstantPad1d((padding, -padding), 0) if padding > 0 else nn.Identity()
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if out.size(2) != res.size(2):
            out = out[:, :, :res.size(2)]
        return self.relu(out + res)

class TCN_GMM(nn.Module):
    """TCN network with GMM output (mean, variance, weight)."""
    def __init__(self, input_size=3, num_channels=[32, 32, 32], kernel_size=3, dropout=0.2, n_components=3):
        super(TCN_GMM, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        
        self.tcn = nn.Sequential(*layers)
        self.n_components = n_components
        
        # GMM output layer (each component: mean + variance + weight)
        self.fc_mu = nn.Linear(num_channels[-1], input_size * n_components)
        self.fc_var = nn.Linear(num_channels[-1], input_size * n_components)
        self.fc_pi = nn.Linear(num_channels[-1], n_components)
        
    def forward(self, x):
        # x: (batch, input_size, seq_len)
        y = self.tcn(x) 
        y = y[:, :, -1]
        
        mu = self.fc_mu(y).view(-1, self.n_components, 3)
        log_var = self.fc_var(y).view(-1, self.n_components, 3)
        var = torch.exp(log_var) + 1e-6
        pi = F.softmax(self.fc_pi(y), dim=-1)
        
        return mu, var, pi

    def sample(self, x, n_samples=10):
        """Sample from the GMM distribution for each input."""
        mu, var, pi = self.forward(x)
        batch_size = x.size(0)
        samples = []
        
        for _ in range(n_samples):
            comp_idx = torch.multinomial(pi, 1).squeeze(-1)
            
            batch_indices = torch.arange(batch_size, device=x.device)
            selected_mu = mu[batch_indices, comp_idx]
            selected_var = var[batch_indices, comp_idx]
            
            eps = torch.randn_like(selected_mu)
            sample = selected_mu + torch.sqrt(selected_var) * eps
            samples.append(sample)
            
        return torch.stack(samples, dim=1)
