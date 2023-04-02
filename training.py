
import gymnasium as gym
from agent import SAC
import torch
from replay_memory import ReplayMemory
import time
import numpy as np
import os
import argparse

cur_dir = os.path.dirname( os.path.abspath( __file__ ))
reward_list = []
value_loss_list, q_loss_list, policy_loss_list = [], [], []

def log(env_name, epi, value_loss, q_loss, policy_loss, epi_reward):
    value_loss_list.append(value_loss)
    q_loss_list.append(q_loss)
    policy_loss_list.append(policy_loss)
    reward_list.append(epi_reward)
    if epi % 10 == 0:
        np.savetxt(cur_dir + '/training_result/{}_reward.txt'.format(env_name), reward_list)
        np.savetxt(cur_dir + '/training_result/{}_q_loss.txt'.format(env_name), q_loss_list)
        np.savetxt(cur_dir + '/training_result/{}_policy_loss.txt'.format(env_name), policy_loss_list)
        np.savetxt(cur_dir + '/training_result/{}_value_loss.txt'.format(env_name), value_loss_list)

# env_name_list= ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4', 'Humanoid-v4', 'Swimmer-v4']

def main(args):
    print(args)
    env_name = args.env_name
    EPSODES = args.episodes
    BATCH_SIZE = args.batch
    GAMMA = args.gamma
    UPDATE_STEP = args.update_step
    CAPACITY = args.capacity
    lr = args.lr
    alpha = args.alpha
    visual = args.visual 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    file_dir = cur_dir + '/traing_result/{}_reward.txt'.format(env_name)

    if visual == True:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name, render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]
    
    if env_name == 'Humanoid-v4':
        reward_scale = 20
    else:
        reward_scale = 5
    
    memory = ReplayMemory(CAPACITY, device)

    agent = SAC(env_name,
                state_dim, 
                action_dim, 
                memory, 
                BATCH_SIZE,
                GAMMA, 
                alpha, 
                lr, 
                action_bounds, 
                reward_scale,
                device)
    agent.createDirectory(file_dir)
    
    print("------------------------------")
    print("ENV : {}".format(env_name))
    print("State : {}".format(state_dim))
    print("Action : {}".format(action_dim))
    print("Action Bounds : {}".format(action_bounds))
    print("Device : {}".format(device))
    print("------------------------------")

    state, info = env.reset()
    start_time = time.time()
    for epi in range(1, EPSODES+1):
        state, info = env.reset()
        epi_reward = 0
        done = 0
        truncation = 0
        while not done and not truncation:
            action = agent.choose_action(state)
    
            next_state, reward, done, truncation, info = env.step(action)
            agent.push(torch.Tensor(state),
            torch.Tensor(np.array([action])), 
            torch.Tensor([reward]), 
            torch.Tensor([done]), 
            torch.Tensor(next_state))
            
            value_loss, q_loss, policy_loss = agent.train()
            
            if epi % UPDATE_STEP == 0:
                agent.save_network()
            epi_reward += reward
            state = next_state  
        
        print("====================================")
        print("EPISODE : {}     Reward : {}     Memory : {}%".format(epi, epi_reward, int(100*len(agent.memory)/CAPACITY)))
        curr_time = time.time() - start_time
        hours = curr_time // 3600
        s = curr_time - hours*3600
        m = s // 60
        print("Running Time : {} hours {} minutes {} seconds".format(hours, m, int(s) - 60*m))
        log(env_name, epi, value_loss, q_loss, policy_loss, epi_reward)
        
        print("====================================")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", help='The name of the environment you want to create.')
    parser.add_argument("--episodes", type = int, default = 7000,help='Number of episodes.')
    parser.add_argument("--batch", type = int, default = 256, help='Batch size.')
    parser.add_argument("--update_step", type = int, default = 50, help='Update step')
    parser.add_argument("--capacity", type = int, default = 1e6, help='Memory capacity')
    parser.add_argument("--alpha", type = int, default = 1, help='temperture parameter of entropy term')
    parser.add_argument("--lr", type = float, default = 3e-4, help='Learning rate')
    parser.add_argument("--gamma", type = float, default = 0.99, help='Discount factor')
    parser.add_argument("--visual", type = bool, default = 0.99, help='Using visualization')
    args = parser.parse_args()
    main(args)
