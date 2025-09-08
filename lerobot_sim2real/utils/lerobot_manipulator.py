"""
Fixed version of ManiSkill's LeRobot manipulator with updated import paths.
Based on https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/lerobot/manipulator.py
but with corrected import paths for newer lerobot versions.
"""

import time
from typing import List, Optional

import numpy as np
import torch

from mani_skill.agents.base_real_agent import BaseRealAgent
from mani_skill.utils import common
from mani_skill.utils.structs.types import Array

try:
    # Updated import paths without 'common' in them
    from lerobot.cameras.camera import Camera
    from lerobot.motors.motors_bus import MotorNormMode
    from lerobot.robots.robot import Robot
except ImportError as e:
    print(f"Warning: Failed to import lerobot components: {e}")
    # Define placeholder classes if imports fail
    class Robot:
        pass
    class Camera:
        pass
    class MotorNormMode:
        pass

# busy_wait doesn't exist in newer lerobot versions
# Define a simple busy wait function that just sleeps briefly
def busy_wait(robot, duration=0.01):
    """Simple busy wait implementation for newer lerobot versions."""
    time.sleep(duration)


class LeRobotRealAgent(BaseRealAgent):
    """
    LeRobotRealAgent is a general class for controlling real robots via the LeRobot system. You simply just pass in the Robot instance you create via LeRobot and pass it here to make it work with ManiSkill Sim2Real environment interfaces.

    Args:
        robot (Robot): The Robot instance you create via LeRobot.
        use_cached_qpos (bool): Whether to cache the fetched qpos values. If True, the qpos will be
            read from the cache instead of the real robot when possible. This cache is only invalidated when
            set_target_qpos or set_target_qvel is called. This can be useful if you want to easily have higher frequency (> 30Hz) control since qpos reading from the robot is
            currently the slowest part of LeRobot for some of the supported motors.
    """

    def __init__(self, robot: 'Robot', use_cached_qpos: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._captured_sensor_data = None
        self.real_robot = robot
        self.use_cached_qpos = use_cached_qpos
        self._cached_qpos = None
        self._cached_qvel = None

    def capture_sensor_data(self, names: Optional[List[str]] = None):
        """
        Capture sensor data from the real robot. The captured data is stored internally and will be
        used later when `get_sensor_data` is called.
        
        Args:
            names: Optional list of sensor names to capture. If None, captures all available sensors.
        """
        start = time.time()
        
        # Get observation from robot which includes camera data
        observation = self.real_robot.get_observation()
        
        # Extract camera/sensor data from the observation
        self._captured_sensor_data = {}
        
        # Process camera data for each requested camera
        if names is not None:
            for name in names:
                # Look for camera data in the observation
                # The key format is usually "camera_name.image" for lerobot
                image_key = f"{name}.image"
                if image_key in observation:
                    # Convert image data to the expected format
                    # ManiSkill expects camera data as a dict with 'rgb' key
                    image_data = observation[image_key]
                    if isinstance(image_data, torch.Tensor):
                        image_data = image_data.cpu().numpy()
                    # Store as a dict with 'rgb' key for compatibility
                    self._captured_sensor_data[name] = {"rgb": image_data}
                elif name in observation:
                    # Direct match
                    data = observation[name]
                    if isinstance(data, torch.Tensor):
                        data = data.cpu().numpy()
                    self._captured_sensor_data[name] = {"rgb": data}
        else:
            # Capture all camera data
            for key, value in observation.items():
                if '.image' in key:
                    # Extract camera name from key (e.g., "base_camera.image" -> "base_camera")
                    camera_name = key.replace('.image', '')
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    self._captured_sensor_data[camera_name] = {"rgb": value}
        
        self.capture_time = time.time() - start

    def get_sensor_data(self, names: Optional[List[str]] = None) -> dict:
        """
        Get the captured sensor data. Use the `names` argument to filter the data by sensor names.
        Args:
            names: A list of sensor names to filter the data.
        Returns:
            A dictionary where the key is the sensor name and the value is a dict with 'rgb' key containing the image data.
        """
        if self._captured_sensor_data is None:
            raise ValueError(
                "Sensor data has not been captured. Please call capture_sensor_data first."
            )
        
        if names is None:
            # Return all captured data
            return self._captured_sensor_data
        
        # Return only requested sensors
        ret = {}
        for name in names:
            if name in self._captured_sensor_data:
                # Data should already be in the correct format from capture_sensor_data
                ret[name] = self._captured_sensor_data[name]
            else:
                # If sensor not found, return empty dict with rgb key to avoid errors
                print(f"Warning: Sensor '{name}' not found in captured data")
                ret[name] = {"rgb": np.zeros((480, 640, 3), dtype=np.uint8)}  # Default empty image
        return ret

    def get_qpos(self, refresh=False) -> Array:
        """
        Get the current joint positions of the real robot.

        Args:
            refresh (bool): If True, the qpos will be read from the real robot again. Otherwise, the cached value will be returned.
        Returns:
            A numpy array of the current joint positions.
        """
        if not self.use_cached_qpos or self._cached_qpos is None or refresh:
            # Get current joint positions from robot
            observation = self.real_robot.get_observation()
            
            if hasattr(self.real_robot, 'observation_features'):
                # For SO101 and similar robots that return observations as a dictionary
                qpos_list = []
                qvel_list = []
                
                # Extract position and velocity values from observation
                for key, value in observation.items():
                    if key.endswith('.pos'):
                        qpos_list.append(value)
                    elif key.endswith('.vel'):
                        qvel_list.append(value)
                
                self._cached_qpos = np.array(qpos_list)
                self._cached_qvel = np.array(qvel_list) if qvel_list else np.zeros_like(self._cached_qpos)
            elif hasattr(self.real_robot, 'motor_names'):
                # For robots with motor_names attribute
                motor_names = self.real_robot.motor_names
                qpos_list = []
                qvel_list = []
                
                for motor_name in motor_names:
                    pos_key = f"{motor_name}.pos"
                    vel_key = f"{motor_name}.vel"
                    if pos_key in observation:
                        qpos_list.append(observation[pos_key])
                    if vel_key in observation:
                        qvel_list.append(observation[vel_key])
                
                self._cached_qpos = np.array(qpos_list)
                self._cached_qvel = np.array(qvel_list) if qvel_list else np.zeros_like(self._cached_qpos)
            else:
                # Fallback for robots that return tensors
                qpos, qvel = self.real_robot.read_motors()
                self._cached_qpos = common.to_numpy(qpos[0])
                self._cached_qvel = common.to_numpy(qvel[0])
        return self._cached_qpos

    def get_qvel(self, refresh=False) -> Array:
        """
        Get the current joint velocities of the real robot.

        Args:
            refresh (bool): If True, the qvel will be read from the real robot again. Otherwise, the cached value will be returned.
        Returns:
            A numpy array of the current joint velocities.
        """
        if not self.use_cached_qpos or self._cached_qvel is None or refresh:
            # Use get_qpos to refresh both qpos and qvel
            self.get_qpos(refresh=True)
        return self._cached_qvel

    def set_target_qpos(self, qpos: Array):
        """
        Set the target joint positions for the real robot. In LeRobot, target q-positions
        are immediately sent to the motors, but the robot will take time to reach them. You
        can use the `busy_wait` function to block until the robot reaches the target positions.

        Args:
            qpos: A numpy array of the target joint positions.
        """
        self._cached_qpos = None
        self._cached_qvel = None
        
        # Convert qpos array to the format expected by the robot
        # Check if robot has action_features (SO101 and similar robots)
        if hasattr(self.real_robot, 'action_features'):
            # Create action dictionary with joint position keys
            action_dict = {}
            action_features = self.real_robot.action_features
            qpos_array = np.array(qpos).flatten()
            
            # Get motor names from action_features keys
            motor_keys = [k for k in action_features.keys() if k.endswith('.pos')]
            
            for i, motor_key in enumerate(motor_keys):
                if i < len(qpos_array):
                    action_dict[motor_key] = qpos_array[i]
            
            # Send the action dictionary to the robot
            self.real_robot.send_action(action_dict)
        elif hasattr(self.real_robot, 'motor_names'):
            # For robots with motor_names attribute
            action_dict = {}
            motor_names = self.real_robot.motor_names
            qpos_array = np.array(qpos).flatten()
            
            for i, motor_name in enumerate(motor_names):
                if i < len(qpos_array):
                    action_dict[f"{motor_name}.pos"] = qpos_array[i]
            
            # Send the action dictionary to the robot
            self.real_robot.send_action(action_dict)
        else:
            # Fallback to tensor format (for other robot types)
            qpos_tensor = torch.from_numpy(np.array(qpos)).float().unsqueeze(0)
            self.real_robot.send_action(qpos_tensor)
        
        busy_wait(self.real_robot)

    def set_target_qvel(self, qvel: Array):
        """
        Set the target joint velocities for the real robot. Note that not all LeRobot robots
        support velocity control.

        Args:
            qvel: A numpy array of the target joint velocities.
        """
        raise NotImplementedError(
            "LeRobot does not currently provide a velocity control interface."
        )

    def release(self):
        """
        Release the real robot. This function currently does nothing in LeRobot.
        """
        pass
    
    def reset(self, qpos: Array = None):
        """
        Reset the real robot to a given joint position or home position.
        
        Args:
            qpos: Target joint positions to reset to. If None, resets to home position.
        """
        if qpos is not None:
            # Move robot to the specified position
            self.set_target_qpos(qpos)
        else:
            # If no position specified, we could move to a default home position
            # For now, just stay at current position
            pass
        
        # Clear cached sensor data
        self._captured_sensor_data = None