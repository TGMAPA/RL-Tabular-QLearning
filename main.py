# Importación de librerias y modulos
import matplotlib.pyplot as plt
import seaborn as sns
from QL import *

# Función main 
def main():
    # Tamaño del grid para el entorno
    SIZE = 10 
    # Número de acciones posibles
    ACTIONS = 3
    # Número de episodios para el entrenamiento
    EPISODES = 5000
    # Número de pasos por episodio
    STEPS = 400
    # Factor para el descuento y propagación del Q-value en la tabla
    DISCOUNT_FACTOR = 0.98
    # Learning rate
    LR = 0.03

    # Instancia del entorno
    env = SimpleEnv(size=SIZE, render_mode=None)
    env = RGBImgObsWrapper(env)

    # Inicializar Q table
    qTable = init_Q_Table(SIZE, ACTIONS)

    # Entrenamiento del agente
    qTable, rewards_per_episode = train(env, qTable, EPISODES, STEPS, ACTIONS, LR, DISCOUNT_FACTOR)

    # Mostrar evolución de las recompensas por episodio
    plt.plot(rewards_per_episode)
    plt.grid(True)
    plt.title("Rewards per episode")
    plt.xlabel("Episode")
    plt.ylabel("T_Reward")
    plt.show()

    # Graficar un heatmap con los Q-value propagados en la tabla de estados
    q_values = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            vals = []
            for d in range(4):
                vals.append(np.max(qTable[(i, j, d)]))
            q_values[i, j] = np.mean(vals)

    plt.figure(figsize=(6, 5))
    sns.heatmap(q_values, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Q-values mean per state")
    plt.show()

    # Probar el aprendizaje del agente por 10 episodios
    test(env, qTable, STEPS, SIZE, 10)

main()