from drone_env import DroneEnv
from stable_baselines3 import PPO

env = DroneEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=512,      # smaller for faster logs
    batch_size=64,
    gamma=0.99
)

model.learn(total_timesteps=500, log_interval=1)

model.save("drone_model")

print("Training complete 🚀")
