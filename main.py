from minigrid.wrappers import RGBImgObsWrapper
from minigrid_simple_env import SimpleEnv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

def smooth_curve(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def train(env, qTable, EPISODES, STEPS, ACTIONS, LR, DISCOUNT_FACTOR):
    
    exploration = True

    # Exploration parameters
    max_epsilon = 1.0           
    min_epsilon = 0.05           
    decay_rate = 0.001

    success_count = 0

    rewards_per_episode = []
    success_per_episode = []

    for episode in range(1, EPISODES + 1):
        total_reward = 0
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        obs, _ = env.reset()
        terminated = False

        for step in range(STEPS):
            # Get current position
            current_state = tuple(env.unwrapped.agent_pos)

            exploration = np.random.uniform(0, 1) < epsilon
            #print("Exploration: ", exploration)

            if exploration:
                # Random action selection
                action = np.random.randint(ACTIONS)
            else:
                # Get max_state
                _, action = Qmax_state(qTable, current_state)

            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)

            reward -= 0.001

            total_reward += reward

            # Get next state
            next_state = tuple(env.unwrapped.agent_pos)

            # Get Q max for next state (updated by taken action)
            qmax = np.max(qTable[next_state])

            # Apply Qlearning equation
            qTable[current_state][action] = q_learning_eq(LR, reward, DISCOUNT_FACTOR, Qk=qTable[current_state][action], maxQ=qmax)

            if terminated or truncated:
                if terminated:
                    success_count += 1
                    
                break

        success_per_episode.append(success_count)
        rewards_per_episode.append(total_reward)

        print(f"--- Episode {episode} finished — ε={epsilon:.3f}, success={terminated}, totalReward={total_reward}")

    # Logs

    print(f"Success rate: {success_count}/{EPISODES}")

    return qTable

def test(env, qTable, STEPS, SIZE, EPISODES):
    print("\nEvaluating trained agent without exploration...\n")

    # Crear un nuevo entorno para visualización
    env = SimpleEnv(size=SIZE, render_mode="human")
    env = RGBImgObsWrapper(env)

    # Mapeo de Q-table a acciones reales de Minigrid
    action_map = {
        0: env.actions.left,    # Girar a la izquierda
        1: env.actions.right,   # Girar a la derecha
        2: env.actions.forward  # Avanzar
    }

    actions_names = ["Left", "Right", "Forward"]
    success_count = 0

    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        terminated = False
        total_reward = 0
        steps = 0

        while not terminated and steps < STEPS:
            current_state = tuple(env.unwrapped.agent_pos)

            # Escoger acción con Q máximo
            current_action_map = qTable[current_state]
            _, q_action = Qmax_state(qTable, current_state)
            real_action = action_map[q_action]

            print(f"Step {steps}: pos={current_state}, action={actions_names[q_action]}, actionmap= {current_action_map}")

            # Ejecutar acción
            obs, reward, terminated, truncated, info = env.step(real_action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        if terminated:
            success_count += 1

        print(f"Test Episode {episode} — Success={terminated}, Total Reward={total_reward:.2f}, Steps={steps}")

    print(f"\nSuccess rate during test: {success_count}/{EPISODES}")



def main(): 

    SIZE = 7
    ACTIONS = 3
    EPISODES = 2000
    STEPS = 350
    DISCOUNT_FACTOR = 0.98
    LR = 0.03

    # render_mode="human" activa la ventana de Pygame
    #env = SimpleEnv(size=SIZE, render_mode="human")
    env = SimpleEnv(size=SIZE, render_mode=None)

    env=RGBImgObsWrapper(env)

    # Init Qtable
    qTable = init_Q_Table(SIZE, ACTIONS)

    qTable = train(env, qTable, EPISODES, STEPS, ACTIONS, LR, DISCOUNT_FACTOR)

    # Plot Q table
    q_values = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            q_values[i, j] = np.max(qTable[(i, j)])

    plt.figure(figsize=(6,5))
    sns.heatmap(q_values, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Max Q-value per state")
    plt.show()

    test(env, qTable, STEPS, SIZE, 10)



if __name__ == "__main__":
    main()