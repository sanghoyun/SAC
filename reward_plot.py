import os
cur_dir = os.path.dirname( os.path.abspath( __file__ ))
print(cur_dir)

import numpy as np
import pandas as pd

import copy

import matplotlib.pyplot as plt
while True:
    env_name_list= ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4', 'Humanoid-v4', 'Swimmer-v4']
    plt.suptitle("Reward", fontsize=20)
    done_list = []
    for i, env_name in enumerate(env_name_list):
    
        file_dir = cur_dir + '/training_result/{}_reward.txt'.format(env_name)
    
        plt.subplot(320+i+1)
        plt.title(env_name)
        if os.path.exists(file_dir):

            reward_list = np.loadtxt(file_dir)
            df = pd.DataFrame(copy.deepcopy(reward_list))
            mean_list = df.rolling(50).mean().dropna()
            # mean_list = reward_list
            plt.plot(mean_list)
            if len(reward_list) >= 6999:
                done_list.append(True)
        plt.grid()
    if all(done_list) and len(done_list) == len(env_name_list):
        break
    plt.show(block=False)
    plt.pause(300)
    plt.close()


env_name_list= ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4', 'Humanoid-v4', 'Swimmer-v4']
plt.suptitle("Reward", fontsize=20)
done_list = []
for i, env_name in enumerate(env_name_list):

    file_dir = cur_dir + '/training_result/{}_reward.txt'.format(env_name)

    plt.subplot(320+i+1)
    plt.title(env_name)
    if os.path.exists(file_dir):

        reward_list = np.loadtxt(file_dir)
        df = pd.DataFrame(copy.deepcopy(reward_list))
        mean_list = df.rolling(50).mean().dropna()
        # mean_list = reward_list
        plt.plot(mean_list)
        if len(reward_list) >= 6999:
            done_list.append(True)
    plt.grid()
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
plt.show()