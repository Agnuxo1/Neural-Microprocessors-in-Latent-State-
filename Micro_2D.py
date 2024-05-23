import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Configuración del tamaño de la esfera
SPHERE_RADIUS = 10  # Radio de la esfera
NUM_NEURONS = 1000  # Número de neuronas en la esfera

# Estados de las neuronas
NEURON_STATES = [-1, 0, 1]
NEURON_COLORS = ['red', 'green', 'blue']  # Colores correspondientes a los estados

# Inicialización de las neuronas
neuron_positions = np.random.uniform(-SPHERE_RADIUS, SPHERE_RADIUS, size=(NUM_NEURONS, 3))
neuron_states = np.random.choice(NEURON_STATES, size=NUM_NEURONS)

# Configuración de la figura y los ejes 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-SPHERE_RADIUS, SPHERE_RADIUS])
ax.set_ylim([-SPHERE_RADIUS, SPHERE_RADIUS])
ax.set_zlim([-SPHERE_RADIUS, SPHERE_RADIUS])
ax.set_title('Simulación de Procesador Neuronal Esférico')

# Inicialización de la representación de las neuronas
scatter = ax.scatter([], [], [], c=[], s=5)

# Función de actualización de la animación
def update_animation(frame):
    # Actualización de los estados de las neuronas (ejemplo simple)
    neuron_states = np.roll(neuron_states, 1)

    # Actualización de las posiciones de las neuronas en la esfera
    neuron_positions[:, 0] = SPHERE_RADIUS * np.cos(frame / 100 * np.pi) * np.sin(neuron_positions[:, 1])
    neuron_positions[:, 1] = SPHERE_RADIUS * np.sin(frame / 100 * np.pi) * np.sin(neuron_positions[:, 2])
    neuron_positions[:, 2] = SPHERE_RADIUS * np.cos(neuron_positions[:, 2])

    # Actualización de la representación de las neuronas
    scatter._offsets3d = (neuron_positions[:, 0], neuron_positions[:, 1], neuron_positions[:, 2])
    scatter._facecolors = [NEURON_COLORS[state] for state in neuron_states]

    return scatter,

# Creación de la animación
animation = FuncAnimation(fig, update_animation, frames=200, interval=50, blit=True)

# Visualización de la animación
plt.show()