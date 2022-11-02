import torch
import torch.nn as nn
import torch.nn.functional as F

class Core(nn.Module):
    '''
    Inputs:
        prev_state - (batch_size, embedding_dim)
        embedded_entity - (batch_size, d_model)
        embedded_spatial(deprecated)
        embedded_scalar(deprecated)
    Outputs:
        next_state - The LSTM state for the next step
        lstm_output - The output of the LSTM
    '''

    def __init__(self, embedding_dim=256, hidden_dim=128, batch_size=128, n_layers=1, drop_prob=0.0):
        super(Core, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
                            dropout=drop_prob, batch_first=True)
        self.batch_size = batch_size

    def forward(self, embedded_entity, hidden_state):
        input_tensor = torch.cat([embedded_entity], dim=-1)
        lstm_output, hidden_state = (input_tensor, hidden_state)
        return lstm_output, hidden_state

    def init_hidden_state(self, batch_size=1):
        device = next(self.parameters()).device
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))
        return hidden
