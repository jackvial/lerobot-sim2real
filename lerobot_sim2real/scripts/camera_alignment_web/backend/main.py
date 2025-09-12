import json
import asyncio
import base64
from typing import Optional, Dict
from dataclasses import dataclass, asdict
import gymnasium as gym
import torch
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import time
import os

from lerobot_sim2real.utils.safety import setup_safe_exit
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from lerobot_sim2real.config.real_robot import create_real_robot
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils import sapien_utils

@dataclass
class CameraConfig:
    camera_offset: list
    fov_offset: float
    timestamp: str = ""

@dataclass
class SimulationState:
    camera_position: list
    fov: float
    frame_count: int
    is_running: bool

app = FastAPI()

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CameraAlignmentServer:
    def __init__(self):
        self.sim_env = None
        self.real_env = None
        self.real_agent = None
        self.real_robot = None
        self.camera_offset = torch.zeros(3, dtype=torch.float32)
        self.fov_offset = 0.0
        self.is_initialized = False
        self.running = False
        self.config_path = "camera_alignment_config.json"
        self.output_dir = "camera_alignment_output"
        self.frame_count = 0
        
        # Display settings
        self.show_camera = True
        self.show_sim = True
        self.overlay_opacity = 0.5
        
        # Movement settings
        self.MOVEMENT_STEP = 0.01
        self.FOV_STEP = 0.01
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def initialize(self, env_id: str = "SO101GraspCube-v1", env_kwargs_json_path: Optional[str] = None):
        """Initialize the simulation and real robot environments"""
        if self.is_initialized:
            return
        
        # Load saved camera configuration if it exists
        self.load_camera_config()
        
        # Initialize real robot
        self.real_robot = create_real_robot(uid="so101")
        self.real_robot.connect()
        self.real_agent = LeRobotRealAgent(self.real_robot)
        
        # Environment configuration
        env_kwargs = dict(
            obs_mode="rgb+segmentation",
            render_mode="sensors",
            reward_mode="none",
            sensor_configs=dict(width=512, height=512)
        )
        
        if env_kwargs_json_path and os.path.exists(env_kwargs_json_path):
            with open(env_kwargs_json_path, "r") as f:
                env_kwargs.update(json.load(f))
        
        # Create simulation environment
        self.sim_env = gym.make(env_id, **env_kwargs)
        self.sim_env = FlattenRGBDObservationWrapper(self.sim_env)
        
        # Create real environment
        self.real_env = Sim2RealEnv(sim_env=self.sim_env, agent=self.real_agent)
        
        # Setup safety exit
        setup_safe_exit(self.sim_env, self.real_env, self.real_agent)
        
        # Reset environment
        self.real_env.reset()
        
        self.is_initialized = True
        self.running = True
    
    def overlay_envs(self):
        """Overlay simulation observations onto real observations based on display settings"""
        if not self.show_camera and not self.show_sim:
            # Return a blank image if both are disabled
            blank_img = torch.zeros((512, 512, 3), dtype=torch.float32)
            return blank_img
        
        real_obs = self.real_env.get_obs()["sensor_data"]
        sim_obs = self.sim_env.get_obs()["sensor_data"]
        
        assert sorted(real_obs.keys()) == sorted(sim_obs.keys()), \
            f"real camera names {real_obs.keys()} and sim camera names {sim_obs.keys()} differ"
        
        overlaid_dict = self.sim_env.get_obs()["sensor_data"]
        overlaid_imgs = []
        
        for name in overlaid_dict:
            real_imgs = real_obs[name]["rgb"][0] / 255
            sim_imgs = overlaid_dict[name]["rgb"][0].cpu() / 255
            
            if self.show_camera and self.show_sim:
                # Blend both based on opacity
                overlaid_img = (1 - self.overlay_opacity) * real_imgs + self.overlay_opacity * sim_imgs
            elif self.show_camera:
                # Show only camera
                overlaid_img = real_imgs
            elif self.show_sim:
                # Show only simulation
                overlaid_img = sim_imgs
            
            overlaid_imgs.append(overlaid_img)
        
        return tile_images(overlaid_imgs)
    
    def update_camera(self, controls: Dict[str, bool]):
        """Update camera position based on control inputs"""
        if controls.get("reset", False):
            self.camera_offset = torch.zeros(3, dtype=torch.float32)
            self.fov_offset = 0.0
            return
        
        # Update camera offset based on controls
        if controls.get("forward", False):
            self.camera_offset[0] -= self.MOVEMENT_STEP
        if controls.get("backward", False):
            self.camera_offset[0] += self.MOVEMENT_STEP
        if controls.get("right", False):
            self.camera_offset[1] += self.MOVEMENT_STEP
        if controls.get("left", False):
            self.camera_offset[1] -= self.MOVEMENT_STEP
        if controls.get("up", False):
            self.camera_offset[2] += self.MOVEMENT_STEP
        if controls.get("down", False):
            self.camera_offset[2] -= self.MOVEMENT_STEP
        
        # FOV control
        if controls.get("fov_decrease", False):
            self.fov_offset -= self.FOV_STEP
        if controls.get("fov_increase", False):
            self.fov_offset += self.FOV_STEP
        
        # Apply camera updates
        pos = self.sim_env.unwrapped.base_camera_settings["pos"] + self.camera_offset
        pose = sapien_utils.look_at(pos, self.sim_env.unwrapped.base_camera_settings["target"])
        self.sim_env.unwrapped.camera_mount.set_pose(pose)
        self.sim_env.unwrapped._sensors["base_camera"].camera.set_fovy(
            self.sim_env.unwrapped.base_camera_settings["fov"] + self.fov_offset
        )
    
    def update_display_settings(self, settings: Dict):
        """Update display settings for camera/sim toggles and opacity"""
        self.show_camera = settings.get("showCamera", True)
        self.show_sim = settings.get("showSim", True)
        self.overlay_opacity = settings.get("opacity", 0.5)
    
    def save_camera_config(self):
        """Save camera configuration to JSON file"""
        config = CameraConfig(
            camera_offset=self.camera_offset.tolist(),
            fov_offset=float(self.fov_offset),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        with open(self.config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        return config
    
    def load_camera_config(self):
        """Load camera configuration from JSON file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            self.camera_offset = torch.tensor(config_data["camera_offset"], dtype=torch.float32)
            self.fov_offset = config_data["fov_offset"]
            return CameraConfig(**config_data)
        return None
    
    def get_current_frames(self) -> Dict[str, str]:
        """Get all three frames: overlay, sim-only, and camera-only as base64 encoded JPEGs"""
        if not self.is_initialized:
            return {"overlayFrame": None, "simFrame": None, "cameraFrame": None}
        
        # Get raw observations
        real_obs = self.real_env.get_obs()["sensor_data"]
        sim_obs = self.sim_env.get_obs()["sensor_data"]
        
        frames = {}
        
        # Process each camera view
        for name in real_obs:
            real_imgs = real_obs[name]["rgb"][0] / 255
            sim_imgs = sim_obs[name]["rgb"][0].cpu() / 255
            
            # Create overlay frame
            overlaid_img = (1 - self.overlay_opacity) * real_imgs + self.overlay_opacity * sim_imgs
            
            # Convert and encode overlay frame
            overlaid_tensor = tile_images([overlaid_img])
            if isinstance(overlaid_tensor, torch.Tensor):
                overlaid_np = overlaid_tensor.cpu().numpy()
            else:
                overlaid_np = overlaid_tensor
            if overlaid_np.dtype != np.uint8:
                overlaid_np = (overlaid_np * 255).astype(np.uint8)
            overlaid_bgr = cv2.cvtColor(overlaid_np, cv2.COLOR_RGB2BGR)
            _, overlay_buffer = cv2.imencode('.jpg', overlaid_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frames["overlayFrame"] = base64.b64encode(overlay_buffer).decode('utf-8')
            
            # Convert and encode sim-only frame
            sim_tensor = tile_images([sim_imgs])
            if isinstance(sim_tensor, torch.Tensor):
                sim_np = sim_tensor.cpu().numpy()
            else:
                sim_np = sim_tensor
            if sim_np.dtype != np.uint8:
                sim_np = (sim_np * 255).astype(np.uint8)
            sim_bgr = cv2.cvtColor(sim_np, cv2.COLOR_RGB2BGR)
            _, sim_buffer = cv2.imencode('.jpg', sim_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frames["simFrame"] = base64.b64encode(sim_buffer).decode('utf-8')
            
            # Convert and encode camera-only frame
            camera_tensor = tile_images([real_imgs])
            if isinstance(camera_tensor, torch.Tensor):
                camera_np = camera_tensor.cpu().numpy()
            else:
                camera_np = camera_tensor
            if camera_np.dtype != np.uint8:
                camera_np = (camera_np * 255).astype(np.uint8)
            camera_bgr = cv2.cvtColor(camera_np, cv2.COLOR_RGB2BGR)
            _, camera_buffer = cv2.imencode('.jpg', camera_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frames["cameraFrame"] = base64.b64encode(camera_buffer).decode('utf-8')
            
            # Save overlay to disk for reference
            output_path = os.path.join(self.output_dir, "camera_alignment.jpg")
            cv2.imwrite(output_path, overlaid_bgr)
            
            break  # Process only the first camera for now
        
        self.frame_count += 1
        return frames
    
    def get_state(self) -> SimulationState:
        """Get current simulation state"""
        if not self.is_initialized:
            return SimulationState(
                camera_position=[0, 0, 0],
                fov=0,
                frame_count=0,
                is_running=False
            )
        
        return SimulationState(
            camera_position=self.camera_offset.tolist(),
            fov=float(self.sim_env.unwrapped.base_camera_settings["fov"] + self.fov_offset),
            frame_count=self.frame_count,
            is_running=self.running
        )
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.real_robot:
            self.real_robot.disconnect()
        if self.sim_env:
            self.sim_env.close()
        if self.real_env:
            self.real_env.close()

# Create global server instance
server = CameraAlignmentServer()

@app.on_event("startup")
async def startup():
    """Initialize server on startup"""
    # Check if env_config.json exists in the project root
    import os
    from pathlib import Path
    env_config_path = Path(__file__).parent.parent.parent.parent.parent / "env_config.json"
    if env_config_path.exists():
        print(f"Loading environment config from {env_config_path}")
        server.initialize(env_kwargs_json_path=str(env_config_path))
    else:
        server.initialize()

@app.on_event("shutdown")
async def shutdown():
    """Clean up on shutdown"""
    server.cleanup()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    
    try:
        while server.running:
            # Receive control inputs from client
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.05)
                
                if data["type"] == "controls":
                    server.update_camera(data["controls"])
                elif data["type"] == "display_settings":
                    server.update_display_settings(data["settings"])
                elif data["type"] == "save_config":
                    config = server.save_camera_config()
                    await websocket.send_json({
                        "type": "config_saved",
                        "config": asdict(config)
                    })
                elif data["type"] == "get_config":
                    config = server.load_camera_config()
                    if config:
                        await websocket.send_json({
                            "type": "config",
                            "config": asdict(config)
                        })
            except asyncio.TimeoutError:
                pass
            
            # Send current frames and state
            frames = server.get_current_frames()
            state = server.get_state()
            
            if frames["overlayFrame"]:
                await websocket.send_json({
                    "type": "frames",
                    **frames,
                    "state": asdict(state)
                })
            
            # Small delay to control frame rate
            await asyncio.sleep(0.05)  # ~20 FPS
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.get("/api/status")
async def get_status():
    """Get server status"""
    return {
        "initialized": server.is_initialized,
        "running": server.running,
        "state": asdict(server.get_state())
    }

@app.get("/api/config")
async def get_config():
    """Get current camera configuration"""
    config = server.load_camera_config()
    if config:
        return asdict(config)
    return {"error": "No configuration found"}

@app.post("/api/config/save")
async def save_config():
    """Save current camera configuration"""
    config = server.save_camera_config()
    return asdict(config)

# Serve static files (frontend)
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)