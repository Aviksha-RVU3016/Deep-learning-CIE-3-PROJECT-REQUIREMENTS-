from drone_env import DroneEnv
from stable_baselines3 import PPO

# Drone1 environment
# Drone1 environment
env1 = DroneEnv(vehicle_name="Drone1")
model1 = PPO(
    "MlpPolicy",
    env1,
    verbose=1,
    n_steps=512,
    batch_size=64,
    learning_rate=0.0003,
    gamma=0.99
)

# Drone2 environment
env2 = DroneEnv(vehicle_name="Drone2")
model2 = PPO(
    "MlpPolicy",
    env2,
    verbose=1,
    n_steps=512,
    batch_size=64,
    learning_rate=0.0003,
    gamma=0.99
)

print("Training Drone1 🚀")
model1.learn(total_timesteps=20)

print("Training Drone2 🚀")
model2.learn(total_timesteps=20)

model1.save("drone1_model")
model2.save("drone2_model")

print("Both drones trained 🔥")

