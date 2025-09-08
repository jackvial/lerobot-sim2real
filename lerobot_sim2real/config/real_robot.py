from pathlib import Path
import gymnasium as gym
from lerobot.robots.robot import Robot
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.utils import make_robot_from_config
import numpy as np
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


def create_real_robot(uid: str = "so100") -> Robot:
    """Wrapper function to map string UIDS to real robot configurations. Primarily for saving a bit of code for users when they fork the repository. They can just edit the camera, id etc. settings in this one file."""
    if uid == "so100":
        robot_config = SO101FollowerConfig(
            port="/dev/ttyACM0",
            use_degrees=True,
            # for phone camera users you can use the commented out setting below
            cameras={
                "base_camera": OpenCVCameraConfig(
                    index_or_path="/dev/video4",
                    fps=30,
                    width=640,
                    height=480
                ),
                # Add more cameras as needed:
                # "front": OpenCVCameraConfig(
                #     index_or_path="/dev/video2",
                #     fps=30,
                #     width=640,
                #     height=480
                # ),
            },
            # for intel realsense camera users you need to modify the serial number or name for your own hardware
            # cameras={
            #     "base_camera": RealSenseCameraConfig(serial_number_or_name="146322070293", fps=30, width=640, height=480)
            # },
            id="stone_home",
        )
        real_robot = make_robot_from_config(robot_config)
        return real_robot