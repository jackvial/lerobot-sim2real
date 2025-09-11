import json
from typing import Optional
import gymnasium as gym
import torch
from lerobot_sim2real.utils.safety import setup_safe_exit
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from lerobot_sim2real.config.real_robot import create_real_robot
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
import cv2
import numpy as np
import tyro
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils import sapien_utils
from dataclasses import dataclass
import os
import threading
import sys
import select
import termios
import tty
import time

@dataclass
class Args:
    env_id: str = "SO101GraspCube-v1"
    """The environment id to train on"""
    env_kwargs_json_path: Optional[str] = None
    """Path to a json file containing additional environment kwargs to use."""
    camera_config_path: Optional[str] = "camera_alignment_config.json"
    """Path to save/load camera alignment configuration"""

def overlay_envs(sim_env, real_env):
    """
    Overlays sim_env observtions onto real_env observations
    Requires matching ids between the two environments' sensors
    e.g. id=phone_camera sensor in real_env / real_robot config, must have identical id in sim_env
    """
    real_obs = real_env.get_obs()["sensor_data"]
    sim_obs = sim_env.get_obs()["sensor_data"]
    assert sorted(real_obs.keys()) == sorted(
        sim_obs.keys()
    ), f"real camera names {real_obs.keys()} and sim camera names {sim_obs.keys()} differ"

    overlaid_dict = sim_env.get_obs()["sensor_data"]
    overlaid_imgs = []
    for name in overlaid_dict:
        real_imgs = real_obs[name]["rgb"][0] / 255
        sim_imgs = overlaid_dict[name]["rgb"][0].cpu() / 255
        overlaid_imgs.append(0.5 * real_imgs + 0.5 * sim_imgs)

    return tile_images(overlaid_imgs)


def update_camera(sim_env):
    global camera_offset, fov_offset
    
    # Fixed step size for single key presses
    MOVEMENT_STEP = 0.01  # units per key press
    FOV_STEP = 0.01  # radians per key press

    # Reset camera position and FOV on backspace
    if "backspace" in active_keys:
        camera_offset = torch.zeros(3, dtype=torch.float32)
        fov_offset = 0.0
        print("Camera reset to initial position")

    # Camera movement mapping based on active keys
    if "w" in active_keys:
        camera_offset[0] -= MOVEMENT_STEP  # Move forward
    if "s" in active_keys:
        camera_offset[0] += MOVEMENT_STEP  # Move back
    if "d" in active_keys:
        camera_offset[1] += MOVEMENT_STEP  # Move right
    if "a" in active_keys:
        camera_offset[1] -= MOVEMENT_STEP  # Move left
    if "up" in active_keys:
        camera_offset[2] += MOVEMENT_STEP  # Move up
    if "down" in active_keys:
        camera_offset[2] -= MOVEMENT_STEP  # Move down

    # FOV control
    if "left" in active_keys:
        fov_offset -= FOV_STEP
    if "right" in active_keys:
        fov_offset += FOV_STEP

    # update camera position and fov
    pos = sim_env.unwrapped.base_camera_settings["pos"] + camera_offset
    pose = sapien_utils.look_at(pos, sim_env.unwrapped.base_camera_settings["target"])
    sim_env.unwrapped.camera_mount.set_pose(pose)
    sim_env.unwrapped._sensors["base_camera"].camera.set_fovy(
        sim_env.unwrapped.base_camera_settings["fov"] + fov_offset
    )

    if len(active_keys) > 0 and "backspace" not in active_keys:
        print(f"Camera position: {pose.p}, FOV: {sim_env.unwrapped.base_camera_settings['fov'] + fov_offset:.3f}")

camera_offset = torch.zeros(3, dtype=torch.float32)
fov_offset = 0.0
active_keys = set()
running = True

def save_camera_config(config_path, camera_offset, fov_offset):
    """Save camera configuration to JSON file"""
    config = {
        "camera_offset": camera_offset.tolist(),
        "fov_offset": float(fov_offset),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nCamera configuration saved to: {config_path}")
    print(f"  Position offset: {camera_offset.tolist()}")
    print(f"  FOV offset: {fov_offset:.3f}")

def load_camera_config(config_path):
    """Load camera configuration from JSON file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        camera_offset = torch.tensor(config["camera_offset"], dtype=torch.float32)
        fov_offset = config["fov_offset"]
        print(f"\nLoaded camera configuration from: {config_path}")
        print(f"  Position offset: {camera_offset.tolist()}")
        print(f"  FOV offset: {fov_offset:.3f}")
        print(f"  Saved at: {config.get('timestamp', 'unknown')}")
        return camera_offset, fov_offset
    else:
        print(f"\nNo saved configuration found at: {config_path}")
        print("Starting with default camera position")
        return torch.zeros(3, dtype=torch.float32), 0.0

def get_single_char():
    """Get a single character from terminal input"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def keyboard_listener(config_path=None):
    """Background thread to listen for keyboard input"""
    global running, active_keys, camera_offset, fov_offset
    while running:
        if select.select([sys.stdin], [], [], 0.1)[0]:
            ch = get_single_char()
            active_keys.clear()
            
            if ch == '\x1b':  # ESC key
                running = False
                print("\nExiting camera alignment...")
            elif ch == 'w':
                active_keys.add('w')
            elif ch == 's':
                active_keys.add('s')
            elif ch == 'a':
                active_keys.add('a')
            elif ch == 'd':
                active_keys.add('d')
            elif ch == 'u':  # 'u' for up
                active_keys.add('up')
            elif ch == 'j':  # 'j' for down (like vim)
                active_keys.add('down')
            elif ch == ',':
                active_keys.add('left')
            elif ch == '.':
                active_keys.add('right')
            elif ch == '\x7f' or ch == '\x08':  # Backspace
                active_keys.add('backspace')
            elif ch == 'p' and config_path:  # 'p' to save (persist)
                save_camera_config(config_path, camera_offset, fov_offset)
            elif ch == 'q':  # Alternative quit key
                running = False
                print("\nExiting camera alignment...")

def main(args: Args):
    global running, camera_offset, fov_offset
    
    # Load saved camera configuration if it exists
    camera_offset, fov_offset = load_camera_config(args.camera_config_path)
    
    real_robot = create_real_robot(uid="so101")
    real_robot.connect()
    real_agent = LeRobotRealAgent(real_robot)

    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        render_mode="sensors",
        reward_mode="none",
        # use larger camera resolution to make it easier to align. In training we won't use this however
        sensor_configs=dict(width=512, height=512)
    )
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs.update(json.load(f))
    sim_env = gym.make(
        args.env_id,
        **env_kwargs,
    )
    sim_env = FlattenRGBDObservationWrapper(sim_env)
    real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent)
    # safety setup, now ctrl+c will first reset the robot to a resting position and then close environments and turn of torque
    setup_safe_exit(sim_env, real_env, real_agent)

    real_env.reset()

    # Create output directory for alignment images
    output_dir = "camera_alignment_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("CAMERA ALIGNMENT TOOL")
    print("="*60)
    print("\nSaving alignment images to:", os.path.abspath(output_dir))
    print("Camera config file:", os.path.abspath(args.camera_config_path))
    print("\n=== Keyboard Controls (Terminal Mode) ===")
    print("  Movement:")
    print("    w/s     - Move forward/backward (X axis)")
    print("    a/d     - Move left/right (Y axis)")
    print("    u/j     - Move up/down (Z axis)")
    print("  Camera:")
    print("    ,/.     - Decrease/Increase FOV")
    print("  Other:")
    print("    p         - Save (persist) current camera position")
    print("    backspace - Reset camera to initial position")
    print("    q or ESC  - Exit")
    print("\nImages are continuously saved to disk.")
    print("Open another terminal and use an image viewer to monitor:")
    print(f"  watch -n 0.5 'ls -la {output_dir}/'")
    print(f"  Or: feh --reload 0.5 {output_dir}/camera_alignment.jpg")
    print("="*60 + "\n")
    
    # Start keyboard listener thread with config path
    keyboard_thread = threading.Thread(target=keyboard_listener, args=(args.camera_config_path,), daemon=True)
    keyboard_thread.start()
    
    frame_count = 0
    while running:
        overlaid_imgs = overlay_envs(sim_env, real_env)
        
        # Convert tensor to numpy array if needed
        if isinstance(overlaid_imgs, torch.Tensor):
            overlaid_imgs = overlaid_imgs.cpu().numpy()
        
        # Convert to BGR for OpenCV saving (assuming RGB input)
        if overlaid_imgs.dtype != np.uint8:
            overlaid_imgs = (overlaid_imgs * 255).astype(np.uint8)
        overlaid_imgs_bgr = cv2.cvtColor(overlaid_imgs, cv2.COLOR_RGB2BGR)
        
        # Save the current frame
        output_path = os.path.join(output_dir, "camera_alignment.jpg")
        cv2.imwrite(output_path, overlaid_imgs_bgr)
        
        # Also save numbered frames periodically (every 10th frame)
        if frame_count % 10 == 0:
            numbered_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(numbered_path, overlaid_imgs_bgr)
        
        # Update camera position based on active keys
        update_camera(sim_env)
        
        frame_count += 1
        
        # Small delay to prevent excessive CPU usage
        time.sleep(0.05)  # 20 FPS
    
    # Save final camera configuration
    if args.camera_config_path:
        save_camera_config(args.camera_config_path, camera_offset, fov_offset)
    
    print("\nCamera alignment completed.")
    print(f"Final images saved in: {os.path.abspath(output_dir)}")
    print(f"Camera configuration saved to: {os.path.abspath(args.camera_config_path)}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)