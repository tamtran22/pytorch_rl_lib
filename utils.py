import numpy as np

def DiscountCulmulativeReward(reward, done, gamma):
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

# def BaselineAdvantage()