import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()

drones = ["Drone1", "Drone2", "Drone3", "Drone4", "Drone5"]

# ---------------- RESET ----------------
client.reset()
time.sleep(2)

# ---------------- TELEPORT SAFE ----------------
for i, drone in enumerate(drones):
    client.simSetVehiclePose(
        airsim.Pose(
            airsim.Vector3r(100 + i*5, i*5, -2),
            airsim.to_quaternion(0, 0, 0)
        ),
        True,
        vehicle_name=drone
    )

time.sleep(2)

# ---------------- TAKEOFF ----------------
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)
    client.takeoffAsync(vehicle_name=drone).join()
    time.sleep(1)

print("Airborne 🚀")
time.sleep(2)

# ---------------- PERFECT V FORMATION (DIRECT) ----------------

center_x = 120
center_y = 0
z = -10
spacing = 6

formation = {
    "Drone1": (0, 0),
    "Drone2": (-spacing, -spacing),
    "Drone3": (-spacing, spacing),
    "Drone4": (-2*spacing, -2*spacing),
    "Drone5": (-2*spacing, 2*spacing),
}

# MOVE DIRECTLY TO FINAL POSITIONS
for drone in drones:
    dx, dy = formation[drone]

    client.moveToPositionAsync(
        center_x + dx,
        center_y + dy,
        z,
        velocity=2,
        vehicle_name=drone
    ).join()

    time.sleep(1)

print("V formation done ✈️")

# ---------------- STABILIZE ----------------
for drone in drones:
    client.hoverAsync(vehicle_name=drone)

print("Stable formation 🎯")