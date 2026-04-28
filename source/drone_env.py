import airsim
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time


class DroneEnv(gym.Env):

    def __init__(self, vehicle_name="Drone1"):
        super(DroneEnv, self).__init__()

        self.vehicle_name = vehicle_name

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # Goal position
        self.goal = np.array([8, 0, -2], dtype=np.float32)

        # Action: velocity (x, y, z)
        self.action_space = spaces.Box(
            low=np.array([-3, -3, -3], dtype=np.float32),
            high=np.array([3, 3, 3], dtype=np.float32),
            dtype=np.float32
        )

        # Observation: position (x, y, z)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Only control THIS drone
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)

        # Move to starting position
        self.client.simSetVehiclePose(
            airsim.Pose(
                airsim.Vector3r(0, 0, -2),
                airsim.to_quaternion(0, 0, 0)
            ),
            ignore_collision=True,
            vehicle_name=self.vehicle_name
        )

        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()

        time.sleep(0.5)

        return self._get_obs(), {}

    def step(self, action):

        # Move drone
        self.client.moveByVelocityAsync(
            float(action[0]),
            float(action[1]),
            float(action[2]),
            1,
            vehicle_name=self.vehicle_name
        ).join()

        obs = self._get_obs()

        # Distance to goal
        distance = np.linalg.norm(obs - self.goal)
        reward = -distance

        # Collision check
        collision = self.client.simGetCollisionInfo(
            vehicle_name=self.vehicle_name
        ).has_collided

        terminated = False

        if collision:
            reward = -100
            terminated = True

        if distance < 1.5:
            reward = 100
            terminated = True
            print(f"{self.vehicle_name} reached goal 🚀")

        return obs, reward, terminated, False, {}

    def _get_obs(self):
        state = self.client.getMultirotorState(
            vehicle_name=self.vehicle_name
        )
        pos = state.kinematics_estimated.position

        return np.array(
            [pos.x_val, pos.y_val, pos.z_val],
            dtype=np.float32
        )
