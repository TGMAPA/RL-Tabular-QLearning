from __future__ import annotations
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Floor
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import matplotlib.pyplot as plt
from minigrid.wrappers import RGBImgObsWrapper,RGBImgPartialObsWrapper
import numpy as np
import random
import os
from minigrid_simple_env import SimpleEnv
from scipy.special import softmax

def calc_state(agent_pos, width, height, direction):
    return (((agent_pos[1]-1) * (height-2) + (agent_pos[0]-1)) * 10) + direction

max_steps=1

def get_action(steps, state, q_table, env, exploitation = False):
    global max_steps

    eps_end = 0.05
    eps_start = 0.9
    eps_decay=1000
    eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1 * steps/eps_decay)

    random_sample=np.random.random()
    print("eps_thresh",steps, random_sample,">", eps_threshold, random_sample>eps_threshold)

    if( exploitation):
        if(exploitation):
            print(q_table[state])
        return q_table[state].argmax()
    else:
        print("random", softmax(q_table[state]))
        return np.random.choice([0,1,2],size=1, p=softmax(q_table[state]))[0]

def main():
    lr = 0.01
    df = 0.99

    width = 19
    height = 19
    env = SimpleEnv(render_mode="human", max_steps=15)
    env=RGBImgObsWrapper(env)

    action_space = 3
    total_states =( width-2) * ( height-2) * 10

    learning=False
    load_model=True
    if(load_model and os.path.exists("q_table.npy")):
        print("Loading model...")
        q_table = np.load("q_table.npy")
        print("Model loaded...")
        
    else:
        q_table = np.zeros((total_states,action_space))



    ep=0
    step=0
    total_episodes=100000
    save_interval=100
    while(ep<total_episodes):
        print("Episode: ", ep)
        obs, _ = env.reset()
        print(env.agent_pos, calc_state(env.agent_pos, width, height, obs["direction"]))
        prev_state=calc_state(env.agent_pos, width, height, obs["direction"])
        terminated = False
        truncated = False
        while(not truncated and not terminated):
            step+=1
            action2take=get_action(step, prev_state, q_table, env, exploitation=not learning)
            print(action2take)
            
            obs = env.step(action2take)
            
            # exit()

            reward = obs[1]

            current_state=calc_state(env.agent_pos, width, height, obs[0]["direction"])

            if(learning):
                print("step", step)
                print("reward",reward)
                print("prev",q_table[prev_state],q_table[current_state])
                q_table[prev_state, action2take] += lr * (reward + df*(q_table[current_state].max()) - q_table[prev_state, action2take])
                # print(q_table[current_state].argmax(),df*q_table[current_state].argmax(),q_table[current_state].max(),df*q_table[current_state].max())
                print("current",q_table[prev_state],q_table[current_state])
                # print("q_values prev state",prev_state,q_table[prev_state])

            prev_state=current_state

            terminated = obs[2]
            truncated = obs[3]
        ep+=1

        if(ep % save_interval == 0 and learning):
            np.save("q_table.npy",q_table)

if __name__ == "__main__":
    main()