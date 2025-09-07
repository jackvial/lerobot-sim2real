# LeRobot Sim2Real

LeRobot Sim2real provides code to train with Reinforcement Learning in fast GPU parallelized simulation and rendering via [ManiSkill](https://github.com/haosulab/ManiSkill) and deploy to the real-world. The codebase is designed for use with the [🤗 LeRobot](https://github.com/huggingface/lerobot) library, which handles all of the hardware interfacing code. Once you clone and follow the installation instructions you can try out the [zero-shot RGB sim2real tutorial](./docs/zero_shot_rgb_sim2real.md) to train in pure simulation something that can pick up cubes in the real world like below:

https://github.com/user-attachments/assets/ca20d10e-d722-48fe-94af-f57e0b2b2fcd

Note that this project is still in a very early stage. There are many ways the sim2real can be improved (like more system ID tools, better reward functions etc.), but we plan to keep this repo extremely simple for readability and hackability.

If you find this project useful, give this repo and [ManiSkill](https://github.com/haosulab/ManiSkill) a star! If you are using [SO100](https://github.com/TheRobotStudio/SO-ARM100/)/[LeRobot](https://github.com/huggingface/lerobot), make sure to also give them a star. If you use ManiSkill / this sim2real codebase in your research, please cite our [research paper](https://arxiv.org/abs/2410.00425):

```
@article{taomaniskill3,
  title={ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI},
  author={Stone Tao and Fanbo Xiang and Arth Shukla and Yuzhe Qin and Xander Hinrichsen and Xiaodi Yuan and Chen Bao and Xinsong Lin and Yulin Liu and Tse-kai Chan and Yuan Gao and Xuanlin Li and Tongzhou Mu and Nan Xiao and Arnav Gurha and Viswesh Nagaswamy Rajesh and Yong Woo Choi and Yen-Ru Chen and Zhiao Huang and Roberto Calandra and Rui Chen and Shan Luo and Hao Su},
  journal = {Robotics: Science and Systems},
  year={2025},
}
```

## Getting Started

### Prerequisites

- Python 3.10+
- UV package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Installation

```bash
# Clone the repository
git clone https://github.com/StoneT2000/lerobot-sim2real.git
cd lerobot-sim2real

# Create a virtual environment with UV
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install the package in editable mode with UV
uv pip install -e .

# The dependencies including torch will be installed automatically via UV
```

### Quick Start

Once installed, you can start training a PPO agent for the SO100 robot grasping task:

TODO - update to outline instructions in docs/zero_shot_rgb_sim2real.md
TODO - Add eval in sim instructions so we can sanity check our model before running on the real robot

```bash
uv run python lerobot_sim2real/scripts/train_ppo_rgb.py --env-id SO100GraspCube
```

The ManiSkill/SAPIEN simulator code is dependent on working NVIDIA drivers and vulkan packages. After running pip install above, if something is wrong with drivers/vulkan, please follow the troubleshooting guide here: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#troubleshooting

To double check if the simulator is installed correctly, you can run

```bash
python -m mani_skill.examples.demo_random_action
```

## Sim2Real Tutorial

We currently provide a tutorial on how to train a RGB based model controlling an SO100 robot arm in simulation and deploying that zero-shot in the real world to grasp cubes. Follow the tutorial [here](./docs/zero_shot_rgb_sim2real.md). Note while SO101 looks similar to SO100, we have found that there are some key differences that make sim2real fail for SO101, we will updaye this repository once SO101 is modelled correctly.

We are also working on a tutorial showing you how to make your own environments ready for sim2real, stay tuned!
