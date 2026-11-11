import gymnasium as gym
import gymnasium_robotics

# Registrar los entornos de gymnasium_robotics
gym.register_envs(gymnasium_robotics)

# Crear el entorno FetchPickAndPlace-v3 con renderizado
env = gym.make("FetchPickAndPlace-v3", render_mode="human")

# Reiniciar el entorno
observation, info = env.reset(seed=42)

# Simulación por 1000 pasos
for _ in range(1000):
    # Tomar una acción aleatoria
    action = env.action_space.sample()

    # Ejecutar la acción en el entorno
    observation, reward, terminated, truncated, info = env.step(action)

    # Si el episodio termina o es truncado, reiniciar el entorno
    if terminated or truncated:
        observation, info = env.reset()

# Cerrar el entorno
env.close()
