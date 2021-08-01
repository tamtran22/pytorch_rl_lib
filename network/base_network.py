import os
import torch
import torch.nn as nn



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