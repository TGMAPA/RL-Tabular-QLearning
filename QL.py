# Importación de librerias
from minigrid.wrappers import RGBImgObsWrapper
from minigrid_simple_env import SimpleEnv
import numpy as np
import matplotlib.pyplot as plt

# Función que Inicializa la tabla Q con ceros para todos los estados y acciones posibles
def init_Q_Table(size, n_actions):
    q_table = {}
    for i in range(size):
        for j in range(size):
            for d in range(4): # Direcciones posibles
                q_table[(i, j, d)] = np.zeros(shape=n_actions) # Vector Q por cada acción
    return q_table

# Ecuación de actualización de Q-Learning
# Q(s,a) ← Q(s,a) + α * [r + γ * max(Q(s’,a’)) - Q(s,a)]
def q_learning_eq(lr, reward, discount_factor, Qk, maxQ):
    return Qk + lr * (reward + discount_factor * maxQ - Qk)

# Devuelve el valor máximo de Q y la acción asociada para un estado dado
def Qmax_state(Qtable, current_state):
    q_values = Qtable[current_state]
    max_idx = np.argmax(q_values)
    max_val = q_values[max_idx]
    return max_val, max_idx

# Entrenamiento del agente mediante Q-Learning
def train(env, qTable, EPISODES, STEPS, ACTIONS, LR, DISCOUNT_FACTOR):
    print("\nEntrenamiento del agente...\n")

    # Parámetros de exploración 
    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.001

    # Contador de episodios exitosos
    success_count = 0

    # Registro de recompensas por episodio
    rewards_per_episode = []

    # Entrenamiento
    for episode in range(1, EPISODES + 1):
        total_reward = 0

        # Calculo de epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        
        # Reiniciar el entorno
        obs, _ = env.reset()
        terminated = False
        
        # Iteración de los pasos definidos
        for step in range(STEPS):
            # Get current position and direction 
            pos = tuple(env.unwrapped.agent_pos)
            dir = env.unwrapped.agent_dir
            current_state = (pos[0], pos[1], dir)

            # Selección de acción 
            if np.random.uniform(0, 1) < epsilon:
                # Acción aleatoria (exploración)
                action = np.random.randint(ACTIONS)
            else:
                # Mejor acción según la Q-table (explotación)
                _, action = Qmax_state(qTable, current_state)

            # Ejecutar la acción en el entorno
            obs, reward, terminated, truncated, info = env.step(action)

            # Agregar ligera penalización para recompensar soluciones más rápidas
            reward -= 0.001  
            total_reward += reward

            # Obtener nuevo estado después de la acción
            new_pos = tuple(env.unwrapped.agent_pos)
            new_dir = env.unwrapped.agent_dir
            next_state = (new_pos[0], new_pos[1], new_dir)

            # Actualizar valor Q usando la ecuación de Q-Learninge
            qmax = np.max(qTable[next_state])
            qTable[current_state][action] = q_learning_eq(LR, reward, DISCOUNT_FACTOR, qTable[current_state][action], qmax)

            # Episodio finalizado
            if terminated or truncated:
                if terminated:
                    # Si se llego a la meta, agregar a la cuenta de episodios exitosos
                    success_count += 1
                break
            
        # Guardar recompensa total del episodio
        rewards_per_episode.append(total_reward)

        # Log
        if episode % 100 == 0:
            print(f"Episode {episode}/{EPISODES} — ε={epsilon:.3f} — Reward={total_reward:.2f}")
    
    # Mostrar tasa de éxito final
    print(f"\nSuccess rate: {success_count}/{EPISODES}")

    return qTable, rewards_per_episode

# Evaluación del agente entrenado sin exploración
def test(env, qTable, STEPS, SIZE, EPISODES):
    print("\nTest del agente entrenado...\n")

    # Crear entorno con renderizado visual
    env = SimpleEnv(size=SIZE, render_mode="human")
    env = RGBImgObsWrapper(env)

    success_count = 0

    # Ejecutar episodios de prueba
    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        terminated = False
        total_reward = 0
        steps = 0

        # Ejecutar pasos hasta que el episodio termine
        while not terminated and steps < STEPS:
            # Obtener el estado actual
            pos = tuple(env.unwrapped.agent_pos)
            dir = env.unwrapped.agent_dir
            current_state = (pos[0], pos[1], dir)

            # Seleccionar la mejor acción según la Q-table
            _, q_action = Qmax_state(qTable, current_state)

            # Ejecutar acción seleccionada
            obs, reward, terminated, truncated, info = env.step(q_action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                # Si llega al objetivo o se termina el episodio cortar el ciclo
                break

        if terminated:
            success_count += 1

        print(f"- Test Episode {episode} — Success={terminated}, Total Reward={total_reward:.2f}, Steps={steps}")

    # Mostrar resultados finales
    print(f"\nSuccess rate during test: {success_count}/{EPISODES}")
