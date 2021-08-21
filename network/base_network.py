import os
import torch
import torch.nn as nn
from utils.utils import prod


#----------------------------------------------------------------------------
# base network/SiLU activation for function approximation
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

    def create_input_layer(self, input_shape, hidden_size, activation=nn.ReLU()):
        layer_list = []
        layer_list.append(nn.Linear(prod(input_shape), hidden_size))
        if activation!=None:
            layer_list.append(activation)
        self.input = nn.Sequential(*layer_list)

    def create_output_layer(self, hidden_size, output_shape, activation=None):
        layer_list = []
        layer_list.append(nn.Linear(hidden_size, prod(output_shape)))
        if activation!=None:
            layer_list.append(activation)
        self.output = nn.Sequential(*layer_list)

    def create_hidden_layer(self, n_hiddens, hidden_size, activation=nn.ReLU()):
        layer_list = []
        for _ in range(n_hiddens - 1):
            layer_list.append(nn.Linear(hidden_size, hidden_size))
            layer_list.append(activation)
        self.hidden = nn.Sequential(*layer_list)


if __name__ == '__main__':
    net = BaseNetwork(
        device=torch.device('cpu'),
        name='test',
        chkpt='./'
    )



