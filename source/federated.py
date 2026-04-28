import torch
from stable_baselines3 import PPO

print("Federated Learning Aggregation 🚀")

# ---------------- LOAD MODELS ----------------
model1 = PPO.load("drone1_model", device="cpu")
model2 = PPO.load("drone2_model", device="cpu")

# ---------------- EXTRACT PARAMETERS ----------------
params1 = model1.policy.state_dict()
params2 = model2.policy.state_dict()

# ---------------- FEDERATED AVERAGING ----------------
global_params = {}

for key in params1:
    global_params[key] = (params1[key] + params2[key]) / 2

print("Aggregation done ✅")

# ---------------- SAFE PRINT ----------------
sample_key = list(params1.keys())[0]

def safe_print(tensor):
    if tensor.dim() == 0:
        return tensor.item()
    elif tensor.dim() == 1:
        return tensor[:5]
    else:
        return tensor[0][:5]

print("\nSample weight BEFORE aggregation:")
print(safe_print(params1[sample_key]))

print("\nSample weight AFTER aggregation:")
print(safe_print(global_params[sample_key]))

# ---------------- LOAD INTO EXISTING MODEL (FIX) ----------------
model1.policy.load_state_dict(global_params)

# ---------------- SAVE GLOBAL MODEL ----------------
model1.save("global_drone_model")

print("\nGlobal model created successfully ✅")