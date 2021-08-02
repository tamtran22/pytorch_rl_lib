import os
import torch
import torch.nn as nn
from utils.utils import prod


#----------------------------------------------------------------------------
# base network
#----------------------------------------------------------------------------
class BaseNetwork(nn.Module):
    def __init__(self, device, name, chkpt) -> None:
        super().__init__()
        self.name = name
        self.checkpoint_file = os.path.join(chkpt, name)
        self.device = device
    def save(self):
        torch.save(self.state_dict(), self.checkpoint_file)
    def load(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    def create_input_layer(self, input_shape, hidden_size):
        self.input = nn.Sequential(
            nn.Linear(prod(input_shape), hidden_size),
            nn.ReLU
        )
    def create_output_layer(self, hidden_size, output_shape):
        self.output = nn.Sequential(
            nn.Linear(hidden_size, prod(output_shape))
        )
    def create_hidden_layers(self, n_hiddens, hidden_size):
        hidden_list = []
        for _ in range(n_hiddens - 1):
            hidden_list.append(nn.Linear(hidden_size, hidden_size))
            hidden_list.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden_list)


