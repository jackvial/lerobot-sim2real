"""
Patch for Sim2RealEnv to fix numpy array handling in preprocess_sensor_data
"""
import numpy as np
import torch
from typing import Dict, List, Optional
from mani_skill.envs.sim2real_env import Sim2RealEnv as BaseSim2RealEnv
from mani_skill.sensors.camera import CameraConfig


class PatchedSim2RealEnv(BaseSim2RealEnv):
    def preprocess_sensor_data(
        self, sensor_data: Dict, sensor_names: Optional[List[str]] = None
    ):
        import cv2

        if sensor_names is None:
            sensor_names = list(sensor_data.keys())
        for sensor_name in sensor_names:
            sim_sensor_cfg = self.base_sim_env._sensor_configs[sensor_name]
            assert isinstance(sim_sensor_cfg, CameraConfig)
            target_h, target_w = sim_sensor_cfg.height, sim_sensor_cfg.width
            real_sensor_data = sensor_data[sensor_name]

            # crop to same aspect ratio
            for key in ["rgb", "depth"]:
                if key in real_sensor_data:
                    # Get the data (expecting shape [1, H, W, C] or [1, H, W] for depth)
                    data = real_sensor_data[key]
                    was_tensor = torch.is_tensor(data)
                    
                    if was_tensor:
                        # It's a tensor, get the first element and convert to numpy for processing
                        img = data[0].cpu().numpy()
                        original_device = data.device
                    else:
                        # It's a numpy array, check if it has batch dimension
                        if len(data.shape) == 4 or (len(data.shape) == 3 and key == "depth"):
                            img = data[0]
                        else:
                            img = data
                    
                    xy_res = img.shape[:2]
                    crop_res = np.min(xy_res)
                    cutoff = (np.max(xy_res) - crop_res) // 2
                    if xy_res[0] == xy_res[1]:
                        pass
                    elif np.argmax(xy_res) == 0:
                        # Handle both 2D (depth) and 3D (rgb) arrays
                        if len(img.shape) == 2:
                            img = img[cutoff:-cutoff, :]
                        else:
                            img = img[cutoff:-cutoff, :, :]
                    else:
                        # Handle both 2D (depth) and 3D (rgb) arrays
                        if len(img.shape) == 2:
                            img = img[:, cutoff:-cutoff]
                        else:
                            img = img[:, cutoff:-cutoff, :]

                    # resize
                    img = cv2.resize(img, (target_w, target_h))
                    
                    # Convert back to proper format
                    if was_tensor:
                        # Convert back to tensor and add batch dimension
                        img = torch.from_numpy(img).to(original_device)
                        if key == "depth" and len(img.shape) == 2:
                            img = img.unsqueeze(-1)  # Add channel dimension for depth
                        img = img.unsqueeze(0)  # Add batch dimension
                        real_sensor_data[key] = img
                    else:
                        # Keep as numpy but ensure proper shape
                        if key == "depth" and len(img.shape) == 2:
                            img = img[..., None]  # Add channel dimension for depth
                        # Add batch dimension if needed
                        if len(data.shape) == 4 or (len(data.shape) == 3 and key == "depth"):
                            img = img[None, ...]  # Add batch dimension
                        real_sensor_data[key] = img
        return sensor_data