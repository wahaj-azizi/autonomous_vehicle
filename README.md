# Autonomous Vehicle with Deep Reinforcement Learning (DDQN)

This repository contains a ROS 2-based autonomous vehicle project using **Double Dueling Deep Q-Networks (DDQN)** for self-driving capabilities in simulation. The vehicle perceives the environment via a camera sensor and makes real-time driving decisions based on reinforcement learning.

## Features
- **ROS 2 Humble** implementation
- **Gazebo simulation** with a realistic driving environment
- **Dueling DDQN** for decision-making
- **Real-time control** of steering, throttle, and braking
- **Camera-based perception** with CNN feature extraction
- **Replay buffer & training pipeline**

## Installation & Setup
### 1. Clone the Repository
```bash
git clone https://github.com/wahaj-azizi/autonomous_vehicle.git
cd autonomous_vehicle
```

### 2. Install Dependencies
Ensure you have **ROS 2 Humble** installed. Then, install the required Python packages:
```bash
sudo apt update && sudo apt install -y ros-humble-cv-bridge ros-humble-sensor-msgs ros-humble-nav-msgs
pip install torch torchvision numpy opencv-python
```

### 3. Build the Workspace
```bash
cd ~/autonomous_vehicle
colcon build --symlink-install
source install/setup.bash
```

## Running the Simulation
### 1. Start Gazebo Simulation
```bash
ros2 launch aws_robomaker_racetrack_world racetrack.launch.py
```

### 2. Run the DDQN Agent
```bash
ros2 run ddqn_audibot ddqn_agent
```

## Training the DDQN Model
To train the DDQN agent and save the model weights:
```bash
ros2 launch ddqn_audibot train_ddqn_launch.py
```
The model weights will be saved in:
```bash
~/ddqn_weights
```

## Debugging & Logging
The agent logs CNN outputs, selected actions, rewards, velocities, and Q-values in real-time. Use the following to view logs:
```bash
ros2 param set /ddqn_node logger_level debug
ros2 param get /ddqn_node logger_level
```

## Contributing
Feel free to open **issues** or **pull requests** if you find bugs or want to improve the project!

## References
This project integrates multiple open-source repositories and dependencies, including:
- **[Audibot](https://github.com/OrebroUniversity/audibot)**: The base vehicle model for ROS 2 simulation.
- **[AWS Robomaker Racetrack World](https://github.com/aws-robotics/aws-robomaker-racetrack-world)**: The simulated track environment for reinforcement learning.
- **[ROS 2 Humble](https://docs.ros.org/en/humble/index.html)**: The middleware framework enabling real-time robotic control.

## License
This project is open-source under the **MIT License**. Additionally, users must comply with the licensing terms of any integrated repositories, including Audibot and AWS Robomaker Racetrack World, as detailed in their respective repositories.

### Additional Licensing Information
- The **Audibot** model follows the licensing terms provided by [OrebroUniversity/audibot](https://github.com/OrebroUniversity/audibot).
- The **AWS Robomaker Racetrack World** follows the licensing terms of [aws-robotics/aws-robomaker-racetrack-world](https://github.com/aws-robotics/aws-robomaker-racetrack-world).
- Ensure compliance with the licensing terms of these repositories when using or modifying their components.

