"""
Script to evaluate a trained PPO policy in the ManiSkill simulator.
"""

from dataclasses import dataclass
import json
import random
from typing import Optional
import gymnasium as gym
import numpy as np
import torch
import tyro
from tqdm import tqdm

from lerobot_sim2real.rl.ppo_rgb import Agent
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode

@dataclass
class Args:
    checkpoint: str
    """Path to the trained checkpoint file to load agent weights from for evaluation."""
    
    env_id: str = "SO100GraspCube-v1"
    """The environment id to use for evaluation. Should match the training environment."""
    
    env_kwargs_json_path: Optional[str] = None
    """Path to a json file containing additional environment kwargs."""
    
    num_episodes: int = 10
    """Number of episodes to evaluate."""
    
    max_episode_steps: int = 100
    """Maximum number of steps per episode."""
    
    seed: int = 1
    """Random seed for reproducibility."""
    
    record_dir: Optional[str] = None
    """Directory to save recorded videos. If None, no recordings are saved."""
    
    render: bool = False
    """Whether to render the environment visually during evaluation."""
    
    include_state: bool = True
    """Whether to include state information in observations (should match training)."""

def main(args: Args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup environment kwargs
    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        render_mode="human" if args.render else "sensors",
        max_episode_steps=args.max_episode_steps,
        reward_mode="normalized_dense"
    )
    
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs.update(json.load(f))
    
    # Create environment
    env = gym.make(args.env_id, **env_kwargs)
    
    # Apply wrappers
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=args.include_state)
    
    if args.record_dir is not None:
        env = RecordEpisode(
            env, 
            output_dir=args.record_dir, 
            save_trajectory=True,
            trajectory_name="trajectory",
            video_fps=env.unwrapped.control_freq,
            info_on_video=True
        )
    
    # Get initial observation for agent initialization
    obs, _ = env.reset()
    
    # The FlattenRGBDObservationWrapper already returns torch tensors with batch dimension
    sample_obs = obs
    
    # Load agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env, sample_obs=sample_obs)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.to(device)
    agent.eval()  # Set to evaluation mode
    
    print(f"Loaded agent from {args.checkpoint}")
    print(f"Evaluating on {args.env_id} for {args.num_episodes} episodes")
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    # Main evaluation loop
    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        print(f"\nEpisode {episode + 1}/{args.num_episodes}")
        
        for _ in tqdm(range(args.max_episode_steps), desc="Steps"):
            # Observations from FlattenRGBDObservationWrapper are already torch tensors with batch dimension
            # Just move them to the appropriate device
            obs_dict = {k: v.to(device) for k, v in obs.items()}
            
            # Get action from agent
            with torch.no_grad():
                action = agent.get_action(obs_dict)
            
            # Step environment
            # The action is already a tensor, convert to numpy
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            
            # Convert outputs to scalars if they're tensors
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            elif isinstance(reward, np.ndarray):
                reward = float(reward)
            
            if isinstance(terminated, (torch.Tensor, np.ndarray)):
                terminated = bool(terminated)
            if isinstance(truncated, (torch.Tensor, np.ndarray)):
                truncated = bool(truncated)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                # Check for success - handle both tensor and scalar
                success = info.get("success", False)
                if isinstance(success, (torch.Tensor, np.ndarray)):
                    success = bool(success)
                successes.append(success)
                
                print(f"  Episode finished: reward={episode_reward:.2f}, length={episode_length}, success={success}")
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # If episode didn't terminate, it was truncated at max steps
        if not (terminated or truncated):
            successes.append(False)
    
    # Print evaluation results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Episodes evaluated: {args.num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success rate: {np.mean(successes)*100:.1f}%")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    
    if args.record_dir:
        print(f"\nVideos saved to: {args.record_dir}")
    
    env.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)