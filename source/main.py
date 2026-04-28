# main.py

import time
import torch

from config import *
from formation import compute_v_positions
from agent import DronePolicy
from federated import federated_average
from resilience import simulate_dropout
from environment import AirSimEnv

# Initialize environment
env = AirSimEnv()

# Drone names
drones = [f"Drone{i+1}" for i in range(NUM_DRONES)]

# Initialize policies
models = [DronePolicy() for _ in range(NUM_DRONES)]

# Arm & takeoff all drones
for drone in drones:
    env.arm_and_takeoff(drone)

print("All drones airborne 🚀")

# Initial target position
Tx = 0
Ty = -50
# --------------------------
# Formation Test Loop
# --------------------------

for step in range(100):

    # Simulate moving target forward
    Ty += 0.5

    positions = compute_v_positions(Tx, Ty, D_FORWARD, D_SIDE)

    for i, drone in enumerate(drones):
        x, y = positions[i]
        env.move_drone(drone, x, y, ALTITUDE)

    time.sleep(0.5)

print("Formation test complete.")