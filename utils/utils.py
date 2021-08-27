import numpy as np

def prod(tup):
    output = 1.
    for d in list(tup):
        output *= d
    return int(output)

def cal_discount_culmulative_reward_1(reward, done, gamma):
    culmulative_reward = np.zeros(reward.shape)
    count = 0
    discount_reward = 0
    for i in range(culmulative_reward.shape[0]):
        discount_reward += (gamma ** count) * reward[i]
        count += 1
        culmulative_reward[i] = discount_reward
        if done[i]:
            count = 0
            discount_reward = 0
    return (culmulative_reward - \
        culmulative_reward.mean())/(culmulative_reward.std() + 1e-6)

def cal_discount_culmulative_reward(reward, done, gamma):
    culmulative_reward = []
    discounted_reward = 0
    for reward_i, done_i in zip(reversed(reward), reversed(done)):
        if done_i:
            discounted_reward = 0
        discounted_reward = reward_i + (gamma * discounted_reward)
        culmulative_reward.insert(0, discounted_reward)
    culmulative_reward =  np.array(culmulative_reward)
    return (culmulative_reward - \
        culmulative_reward.mean())/(culmulative_reward.std() + 1e-6)

def cal_gae(reward, done, value, next_value, gamma, gae_lambda):
    value = np.append(value, np.array([[next_value]]), axis=0)
    gae = 0
    culmulative_reward = []
    for i in reversed(range(len(reward))):
        delta = reward[i] + gamma * value[i+1]*done[i] - value[i]
        gae = delta + gamma * gae_lambda * done[i] * gae
        culmulative_reward.insert(0, gae + value[i])
    culmulative_reward = np.array(culmulative_reward)
    return (culmulative_reward - \
        culmulative_reward.mean())/(culmulative_reward.std() + 1e-6)

# def BaselineAdvantage()

if __name__=='__main__':
    # tup = (1,)
    # print(prod(tup))
    reward = np.array([1,1,1,1,1])
    done = [0,0,0,1,0]
    returns = cal_discount_culmulative_reward(reward, done, 0.9)
    print(returns)