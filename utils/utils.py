import numpy as np

<<<<<<< HEAD:utils.py
=======
def prod(_tuple):
    product = 1
    for element in list(_tuple):
        product *= element
    return int(product)

>>>>>>> 759116c068149adb24c65fcf877c5db5db0234f8:utils/utils.py
def cal_discount_culmulative_reward(reward, done, gamma):
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