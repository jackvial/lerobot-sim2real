# Evaluating Trained Policies in ManiSkill Simulator

This guide explains how to evaluate PPO policies trained in ManiSkill environments using the lerobot-sim2real framework.

## Overview

After training a policy using `train_ppo_rgb.py`, you can evaluate its performance in simulation to:
- Measure success rates and average rewards
- Generate evaluation videos
- Test on different environment configurations
- Debug policy behavior before real-world deployment

## Quick Start

### Basic Evaluation

Evaluate a trained checkpoint with default settings:

```bash
python lerobot_sim2real/scripts/eval_ppo_sim.py \
    --checkpoint runs/your_run_name/final_ckpt.pt \
    --env-id SO100GraspCube-v1 \
    --num-episodes 10
```

Example output:
```
Loaded agent from runs/SO100GraspCube__ppo_rgb__1__1757276141/ckpt_1.pt
Evaluating on SO100GraspCube-v1 for 3 episodes

Episode 1/3
  Episode finished: reward=6.96, length=100, success=False

Episode 2/3
  Episode finished: reward=19.00, length=100, success=False

Episode 3/3
  Episode finished: reward=18.89, length=100, success=False

==================================================
EVALUATION RESULTS
==================================================
Episodes evaluated: 3
Average reward: 14.95 ± 5.65
Average episode length: 100.0 ± 0.0
Success rate: 0.0%
Min reward: 6.96
Max reward: 19.00
```

### Evaluation with Video Recording

Record videos of the evaluation episodes:

```bash
python lerobot_sim2real/scripts/eval_ppo_sim.py \
    --checkpoint runs/your_run_name/final_ckpt.pt \
    --env-id SO100GraspCube-v1 \
    --num-episodes 10 \
    --record-dir evaluation_videos/
```

### Evaluation with Custom Environment Configuration

Use a JSON file to specify custom environment parameters:

```bash
python lerobot_sim2real/scripts/eval_ppo_sim.py \
    --checkpoint runs/your_run_name/final_ckpt.pt \
    --env-id SO100GraspCube-v1 \
    --env-kwargs-json-path configs/eval_env_config.json \
    --num-episodes 20
```

Example `eval_env_config.json`:
```json
{
    "domain_randomization": true,
    "robot_uid": "so100",
    "camera_cfgs": {
        "phone_camera": {
            "width": 640,
            "height": 480
        }
    }
}
```

## Using the Built-in PPO Evaluation Mode

The original `ppo_rgb.py` training script also supports evaluation mode:

```bash
python lerobot_sim2real/rl/ppo_rgb.py \
    --evaluate \
    --checkpoint runs/your_run_name/final_ckpt.pt \
    --env-id SO100GraspCube-v1 \
    --num-eval-envs 8 \
    --capture-video
```

This will:
- Load the checkpoint
- Run evaluation on parallel environments
- Save videos to `runs/your_run_name/test_videos/`
- Save trajectory data if specified

## Command-Line Arguments

### eval_ppo_sim.py Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | Required | Path to trained checkpoint file |
| `--env-id` | str | SO100GraspCube-v1 | ManiSkill environment ID |
| `--env-kwargs-json-path` | str | None | Path to JSON file with environment kwargs |
| `--num-episodes` | int | 10 | Number of episodes to evaluate |
| `--max-episode-steps` | int | 100 | Maximum steps per episode |
| `--seed` | int | 1 | Random seed for reproducibility |
| `--record-dir` | str | None | Directory to save evaluation videos |
| `--render` | bool | False | Enable visual rendering during evaluation |
| `--include-state` | bool | False | Include state in observations (must match training) |

### PPO Built-in Evaluation Arguments

When using `--evaluate` flag with the training script:

| Argument | Type | Description |
|----------|------|-------------|
| `--evaluate` | flag | Enable evaluation mode |
| `--checkpoint` | str | Path to checkpoint to evaluate |
| `--num-eval-envs` | int | Number of parallel evaluation environments |
| `--num-eval-steps` | int | Steps per evaluation episode |
| `--capture-video` | flag | Save evaluation videos |
| `--eval-partial-reset` | flag | Reset on termination vs truncation |

## Understanding Evaluation Metrics

The evaluation script reports several key metrics:

- **Success Rate**: Percentage of episodes where the task was successfully completed
- **Average Reward**: Mean cumulative reward across all episodes
- **Episode Length**: Average number of steps taken per episode
- **Min/Max Rewards**: Range of performance across episodes

## Advanced Usage

### Batch Evaluation on Multiple Checkpoints

Evaluate multiple checkpoints from training:

```bash
for ckpt in runs/your_run_name/ckpt_*.pt; do
    echo "Evaluating $ckpt"
    python lerobot_sim2real/scripts/eval_ppo_sim.py \
        --checkpoint $ckpt \
        --env-id SO100GraspCube-v1 \
        --num-episodes 20 \
        --record-dir videos/$(basename $ckpt .pt)/
done
```

### Evaluation with Different Random Seeds

Test robustness across different initializations:

```bash
for seed in 1 2 3 4 5; do
    python lerobot_sim2real/scripts/eval_ppo_sim.py \
        --checkpoint runs/your_run_name/final_ckpt.pt \
        --env-id SO100GraspCube-v1 \
        --seed $seed \
        --num-episodes 10
done
```

### Custom Evaluation Loop

For more control, you can write a custom evaluation script:

```python
import torch
import gymnasium as gym
from lerobot_sim2real.rl.ppo_rgb import Agent
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

# Setup environment
env = gym.make("SO100GraspCube-v1", obs_mode="rgb+segmentation")
env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False)

# Load agent
obs, _ = env.reset()
agent = Agent(env, sample_obs=obs)
agent.load_state_dict(torch.load("path/to/checkpoint.pt"))
agent.eval()

# Custom evaluation logic
for episode in range(10):
    obs, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Add custom logging or analysis here
```

## Integration with Real Robot Evaluation

The evaluation scripts are designed to be compatible with real robot deployment:

1. **Consistent Observation Space**: Use the same `obs_mode` and wrappers during training and evaluation
2. **Camera Configuration**: Match camera parameters between sim and real using `env_kwargs_json_path`
3. **Action Space**: Ensure control modes are consistent

To transition from sim to real evaluation:

```bash
# First validate in simulation
python lerobot_sim2real/scripts/eval_ppo_sim.py \
    --checkpoint runs/your_run_name/final_ckpt.pt \
    --env-id SO100GraspCube-v1

# Then evaluate on real robot
python lerobot_sim2real/scripts/eval_ppo_rgb.py \
    --checkpoint runs/your_run_name/final_ckpt.pt \
    --env-id SO100GraspCube-v1 \
    --debug  # Enable sim-real visualization
```

## Important Notes

### Matching Training Configuration

The evaluation script must use the same observation configuration as training:
- **include_state**: Defaults to `True` (most PPO policies are trained with state information)
- **obs_mode**: Should match training (typically `"rgb+segmentation"`)
- **Environment ID**: Must match the training environment

If you get errors about unexpected keys in state_dict or shape mismatches, check:
1. Whether state was included during training (check training logs or config)
2. The observation mode used during training
3. The exact environment ID used

## Troubleshooting

### Common Issues

1. **State dict mismatch errors**:
   ```
   RuntimeError: Error(s) in loading state_dict for Agent:
   Unexpected key(s) in state_dict: "feature_net.extractors.state.weight"
   ```
   Solution: Set `--include-state False` if the model was trained without state information

2. **Shape mismatch errors**:
   ```
   size mismatch for critic.0.weight: copying a param with shape torch.Size([512, 512])
   ```
   Solution: Ensure `--include-state` matches the training configuration

3. **Environment not found**: Install required ManiSkill environments
4. **CUDA out of memory**: Reduce batch size or use CPU with smaller models
5. **Video recording fails**: Check write permissions for output directory

### Performance Tips

- Use `--num-eval-envs` > 1 for faster parallel evaluation
- Disable rendering (`--render False`) for faster evaluation
- Use GPU for model inference when available
- Set appropriate `--max-episode-steps` based on task complexity

## References

- [ManiSkill Documentation](https://maniskill.readthedocs.io/)
- [PPO Training Guide](./training_ppo.md)
- [Real Robot Evaluation](./real_robot_evaluation.md)