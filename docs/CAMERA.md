# Camera Settings Documentation

## How Camera Settings Work in Training

### Configuration Loading
The camera settings from `env_config.json` **ARE used during training** for the actual observations that the PPO policy learns from.

#### Configuration Flow:
1. `train_ppo_rgb.py` loads the JSON config via `--env_kwargs_json_path`
2. Settings are passed as `env_kwargs` to the environment (lerobot_sim2real/rl/ppo_rgb.py:318)
3. The environment receives `base_camera_settings` from the kwargs

### Camera Settings Usage in Environment

The `SO100GraspCube` environment (and similar environments) use `base_camera_settings` to configure the observation camera:

- **Camera FOV**: Used directly from `base_camera_settings["fov"]` (grasp_cube.py:148)
- **Camera Position**: Set from `base_camera_settings["pos"]` (grasp_cube.py:343)
- **Camera Target**: Set from `base_camera_settings["target"]` (grasp_cube.py:344)

These settings configure the `base_camera` sensor that provides RGB observations for training.

### Default vs Custom Settings

Without `env_config.json`, the environment uses defaults:
```python
base_camera_settings=dict(
    fov=52 * np.pi / 180,
    pos=[0.5, 0.3, 0.35],
    target=[0.3, 0.0, 0.1],
)
```

With `env_config.json`, your custom settings override these defaults:
```json
{
    "base_camera_settings": {
        "pos": [0.05, 0.57, 0.38],
        "fov": 0.4256,
        "target": [0.185, -0.15, 0.0]
    }
}
```

## Important Distinction: Training vs Evaluation Videos

### Training Observations ✓
- Use your custom camera settings from `env_config.json`
- These are the actual RGB images the policy learns from
- Correctly positioned according to your configuration

### Evaluation Videos (RecordEpisode) ✗
- Use the environment's default `render()` method
- Do NOT inherit the custom camera settings
- This is why eval videos may appear with different camera angles
- The `RecordEpisode` wrapper (from ManiSkill) doesn't have a mechanism to pass custom camera parameters

### Key Takeaway
**Your custom camera settings ARE being used for training the policy**, which is what matters for learning. The evaluation videos showing a different angle is only a visualization issue and doesn't affect the training process or the learned policy's performance.