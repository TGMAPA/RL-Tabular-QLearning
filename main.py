from minigrid.wrappers import RGBImgObsWrapper
from minigrid_simple_env import SimpleEnv
import numpy as np

def init_Q_Table(size, n_actions):
    q_table = {}
    for i in range(size):
        for j in range(size):
            q_table[(i,j)] = np.zeros(shape=n_actions)
    return q_table

def q_learning_eq(lr, reward, discount_factor, Qk, maxQ):
    return Qk + lr*(reward + discount_factor * maxQ - Qk)

def Qmax_state(Qtable:dict, current_pos:tuple):
    max = None
    max_idx = None
    for i in range(len(Qtable[current_pos])):
        if max == None and max_idx == None:
            max_idx = i
            max = Qtable[current_pos][i]
        else:
            # Compare
            if Qtable[current_pos][i] > max:
                max = Qtable[current_pos][i]
                max_idx = i

    return max, max_idx

def main(): 
    SIZE = 19
    ACTIONS = 4
    EPISODES = 300
    DISCOUNT_FACTOR = 0.95
    LR = 0.01

    # render_mode="human" activa la ventana de Pygame
    env = SimpleEnv(size=SIZE, render_mode="human")

    env=RGBImgObsWrapper(env)
    obs, _ = env.reset()

    # Init Qtable
    qTable = init_Q_Table(SIZE, ACTIONS)

    exploration = True

    for episode in range(1, EPISODES+1):
        # Get current position
        current_state = env.unwrapped.agent_pos
        
        
        if exploration:
            # Random action selection
            action = np.random.randint(0, ACTIONS)
        else:
            # Get max_state
            _ , action = Qmax_state(qTable, current_state)

        # Execute step
        obs = env.step(action)

        # Get next state
        next_state = env.unwrapped.agent_pos

        # Get Q max for next state (updated by taken action)
        qmax = np.max(qTable[next_state])

        # Get obtained reward
        reward = obs[1]

        # Apply Qlearning equation
        qTable[current_state][action] = q_learning_eq(LR, reward, DISCOUNT_FACTOR, Qk=qTable[current_state][action], maxQ=qmax)
        


if __name__ == "__main__":
    main()
