from minigrid.wrappers import RGBImgObsWrapper
from minigrid_simple_env import SimpleEnv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def init_Q_Table(size, n_actions):
    q_table = {}
    for i in range(size):
        for j in range(size):
            for d in range(4):  # 4 direcciones
                q_table[(i, j, d)] = np.zeros(shape=n_actions)
    return q_table


def q_learning_eq(lr, reward, discount_factor, Qk, maxQ):
    return Qk + lr * (reward + discount_factor * maxQ - Qk)


def Qmax_state(Qtable: dict, current_state: tuple):
    q_values = Qtable[current_state]
    max_idx = np.argmax(q_values)
    max_val = q_values[max_idx]
    return max_val, max_idx


def smooth_curve(data, window=50):
    return np.convolve(data, np.ones(window) / window, mode='valid')


def train(env, qTable, EPISODES, STEPS, ACTIONS, LR, DISCOUNT_FACTOR):
    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.001

    success_count = 0
    rewards_per_episode = []

    for episode in range(1, EPISODES + 1):
        total_reward = 0
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        obs, _ = env.reset()
        terminated = False

        for step in range(STEPS):
            pos = tuple(env.unwrapped.agent_pos)
            dir = env.unwrapped.agent_dir
            current_state = (pos[0], pos[1], dir)

            # Política epsilon-greedy
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(ACTIONS)
            else:
                _, action = Qmax_state(qTable, current_state)

            # Ejecutar acción
            obs, reward, terminated, truncated, info = env.step(action)

            reward -= 0.001  # Pequeña penalización por paso
            total_reward += reward

            # Nuevo estado
            new_pos = tuple(env.unwrapped.agent_pos)
            new_dir = env.unwrapped.agent_dir
            next_state = (new_pos[0], new_pos[1], new_dir)

            # Actualización Q-learning
            qmax = np.max(qTable[next_state])
            qTable[current_state][action] = q_learning_eq(LR, reward, DISCOUNT_FACTOR, qTable[current_state][action], qmax)

            if terminated or truncated:
                if terminated:
                    success_count += 1
                break

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode}/{EPISODES} — ε={epsilon:.3f} — Reward={total_reward:.2f}")

    print(f"\nSuccess rate: {success_count}/{EPISODES}")
    return qTable


def test(env, qTable, STEPS, SIZE, EPISODES):
    print("\nEvaluando agente entrenado...\n")

    env = SimpleEnv(size=SIZE, render_mode="human")
    env = RGBImgObsWrapper(env)

    action_map = {
        0: env.actions.left,    # Girar izquierda
        1: env.actions.right,   # Girar derecha
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
            pos = tuple(env.unwrapped.agent_pos)
            dir = env.unwrapped.agent_dir
            current_state = (pos[0], pos[1], dir)

            _, q_action = Qmax_state(qTable, current_state)
            real_action = action_map[q_action]

            print(f"Step {steps}: pos={current_state}, action={actions_names[q_action]}")
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
    SIZE = 9
    ACTIONS = 3
    EPISODES = 2000
    STEPS = 250
    DISCOUNT_FACTOR = 0.98
    LR = 0.03

    env = SimpleEnv(size=SIZE, render_mode=None)
    env = RGBImgObsWrapper(env)

    qTable = init_Q_Table(SIZE, ACTIONS)
    qTable = train(env, qTable, EPISODES, STEPS, ACTIONS, LR, DISCOUNT_FACTOR)

    # Visualizar Q-table (promedio por posición)
    q_values = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            vals = []
            for d in range(4):
                vals.append(np.max(qTable[(i, j, d)]))
            q_values[i, j] = np.mean(vals)

    plt.figure(figsize=(6, 5))
    sns.heatmap(q_values, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Promedio de Q-values por celda (promediando dirección)")
    plt.show()

    test(env, qTable, STEPS, SIZE, 10)


if __name__ == "__main__":
    main()
