import torch

class BaseAgent():
    def __init__(self, env=None, device=None) -> None:
        if device!=None:
            self.device = device
        else:
            self.device = torch.device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        if env != None:
            self.load_env(env)
            self.create_memory()
            self.create_network()
    
    def load_env(self, env):
        self.env = env
        # Discrete/continuous state space
        if self.env.observation_space.shape == ():
            self.state_shape = (self.env.observation_space.n,)
            self.state_type = 'discrete'
        else:
            self.state_shape = self.env.observation_space.shape
            self.state_type = 'continuous'
        # Discrete/continuous action space
        if env.action_space.shape == ():
            self.action_shape = (self.env.action_space.n,)
            self.action_type = 'discrete'
        else:
            self.action_shape = self.env.env.action_space.shape
            self.action_type = 'continuous'
    
    def create_memory(self, *args, **kwargs):
        self.memory = None
    
    def create_network(self, *args, **kwargs):
        self.network = []
    
    def save_network(self):
        print("Saving networks...")
        for net in self.network:
            net.save()
    
    def load_network(self):
        print("Saving network...")
        for net in self.network:
            net.load()
    