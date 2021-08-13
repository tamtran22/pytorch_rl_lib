import torch
import torch.distributions as distributions
from network.distribution_network import *


#----------------------------------------------------------------------------
# Fixed Policies
#----------------------------------------------------------------------------
def EpsilonGreedy(input_tensor, epsilon=1e-2):
    pass

#----------------------------------------------------------------------------
# Network Policies
#----------------------------------------------------------------------------
class DiscreteTDActor(CategoricalDistributionNetwork):
    def __init__(self, lr, state_shape, n_actions, device, n_hiddens=2, 
            hidden_size=256, name='actor_TD', chkpt='./temp') -> None:
        super().__init__(lr, state_shape, n_actions, device, 
            n_hiddens=n_hiddens, hidden_size=hidden_size, 
            name=name, chkpt=chkpt)
    def act(self, state):
        prob = self.forward(state)
        dist = distributions.Categorical(prob)
    def evaluate(self, state, action):
        pass

class ContinuousTDActor(GaussianDistributionNetwork):
    def __init__(self, lr, state_shape, action_shape, device, n_hiddens=2, 
            hidden_size=256, name='actor_TD', chkpt='./temp') -> None:
        super().__init__(lr, state_shape, action_shape, device, 
            n_hiddens=n_hiddens, hidden_size=hidden_size, reshape_action=False,
            name=name, chkpt=chkpt)
    def act(self, state):
        # Feeding 1 state at a time.
        mu, sigma = self.forward(state)
        mu = mu.squeeze(0)
        sigma = sigma.squeeze(0)
        dist = distributions.Normal(mu, sigma)
        pred_action = dist.sample()
        logprob = dist.log_prob(pred_action)
        # reshape output to action shape
        pred_action = torch.reshape(pred_action, (state.shape[0],) + self.action_shape)
        logprob = torch.reshape(logprob, (state.shape[0],) + self.action_shape)
        return pred_action, logprob
    def evaluate(self, state, action):
        action = torch.flatten(action, start_dim=1).to(self.device)
        logprob = []
        entropy = []
        mu, sigma = self.forward(state)
        for i in range(len(action)):
            dist_i = distributions.Normal(mu[i], sigma[i])
            logprob_i = dist_i.log_prob(action[i])
            logprob.append(logprob_i)
            entropy.append(dist_i.entropy())
        # reshape output to action shape
        logprob = torch.stack(logprob)
        logprob = torch.reshape(logprob, (state.shape[0],) + self.action_shape)
        entropy = torch.stack(entropy)
        entropy = torch.reshape(entropy, (state.shape[0],) + self.action_shape)
        return logprob, entropy

if __name__=='__main__':
    actor = TDActor(
        lr=1e-3,
        state_shape=(4,),
        action_shape=(2,),
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    state = torch.rand((4,4))
    action = torch.rand((4,2))
    # x = action_value_net.forward(state, action)
    # print(x, x.shape)
    # y = state_value_net.forward(state)
    # print(y, y.shape)
    # mu, sigma = gauss.forward(state)
    # print(mu, sigma)
    # pred_action, old_logprob = actor.act(state)
    # print(pred_action.shape, old_logprob.shape)
    new_logprob, entropy = actor.evaluate(state, action)
    print(new_logprob.shape, entropy.shape)