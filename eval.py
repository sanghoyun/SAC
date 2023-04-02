
import gymnasium as gym
from agent import SAC
import torch
from replay_memory import ReplayMemory
import time
import argparse

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

    print("------------------------------")
    print("ENV : {}".format(env_name))
    print("State : {}".format(state_dim))
    print("Action : {}".format(action_dim))
    print("Action Bounds : {}".format(action_bounds))
    print("Device : {}".format(device))
    print("------------------------------")

    agent.load_network()
    agent.eval_mode()


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
            
            epi_reward += reward
            state = next_state

        print(done, truncation)
        print("====================================")
        print("EPISODE : {}     Reward : {}     Memory : {}%".format(epi, epi_reward, int(100*len(agent.memory)/CAPACITY)))
        curr_time = time.time() - start_time
        hours = curr_time // 3600
        s = curr_time - hours*3600
        m = s // 60
        print("Running Time : {} hours {} minutes {} seconds".format(hours, m, int(s) - 60*m))
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
