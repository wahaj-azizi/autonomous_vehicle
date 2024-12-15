import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, UInt8
from cv_bridge import CvBridge
import cv2

class DuelingDDQN(nn.Module):
    def __init__(self, input_shape, action_size):
        super(DuelingDDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_input_size = self._get_conv_output(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU()
        )
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, action_size)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 255.0  # Normalize input to range [0, 1]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean()

class DDQNNode(Node):
    def __init__(self):
        super().__init__('ddqn_agent')
        self.bridge = CvBridge()
        self.state_size = (3, 128, 128)  # Input shape (channels, height, width)
        self.action_size = self.declare_parameter('action_size', 6).value  # Added a new action
        self.gamma = self.declare_parameter('gamma', 0.99).value
        self.epsilon = self.declare_parameter('epsilon', 1.0).value
        self.epsilon_decay = self.declare_parameter('epsilon_decay', 0.995).value
        self.min_epsilon = self.declare_parameter('min_epsilon', 0.01).value
        self.learning_rate = self.declare_parameter('learning_rate', 0.001).value
        self.batch_size = self.declare_parameter('batch_size', 32).value

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingDDQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DuelingDDQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.memory = deque(maxlen=200)
        self.steps_done = 0

        # Publishers for control commands
        self.steering_pub = self.create_publisher(Float64, '/audibot/steering_cmd', 10)
        self.throttle_pub = self.create_publisher(Float64, '/audibot/throttle_cmd', 10)
        self.brake_pub = self.create_publisher(Float64, '/audibot/brake_cmd', 10)
        self.gear_pub = self.create_publisher(UInt8, '/audibot/gear_cmd', 10)

        # Subscribers for the camera feed and odometry data
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/audibot/odom', self.odom_callback, 10)

        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        self.get_logger().info("DDQN Node initialized.")

        # Vehicle state variables
        self.steering_angle = 0.0  # Radians
        self.throttle = 0.0  # 0 to 1
        self.brake = 0.0  # Nm
        self.gear = 0  # Drive = 0, Reverse = 1
        self.car_velocity = 0.0  # Linear velocity in m/s
        self.lateral_distance = 0.0  # Distance from lane center

    def odom_callback(self, msg):
        self.car_velocity = msg.twist.twist.linear.x  # Update velocity from odometry
        self.lateral_distance = msg.pose.pose.position.y  # Assume y-axis represents lateral distance

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            cv_image = cv2.resize(cv_image, (128, 128))
            state = np.transpose(cv_image, (2, 0, 1))  # Convert to CHW format
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Log CNN's output
            cnn_output = self.policy_net(state)
            self.get_logger().debug(f"CNN Output: {cnn_output.cpu().detach().numpy()}")

            # Select and log the action
            action = self.select_action(state)
            self.get_logger().debug(f"Selected Action: {action}")

            # Execute the action
            self.execute_action(action)

            # Simulate car position and track state for reward
            is_off_track = abs(self.lateral_distance) > 2.0  # Example threshold for being off-track

            # Ensure `is_off_track` is treated as a float
            is_off_track = float(is_off_track)

            # Calculate and log the reward
            reward = self.calculate_reward(action, self.lateral_distance, self.car_velocity, is_off_track)
            self.get_logger().debug(f"Reward: {reward}")

            # Simulate the next state (use actual next frame in practice)
            next_state = state
            self.store_transition(state, action, reward, next_state, done=False)

            # Train the model if memory has enough samples
            if len(self.memory) > self.batch_size:
                self.replay()
        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        with torch.no_grad():
            return torch.argmax(self.policy_net(state)).item()


    def execute_action(self, action):
        if action == 0:  # Move forward (increase throttle)
            self.gear = 0  # Ensure forward gear
            self.throttle = min(self.throttle + 0.03, 0.2)
            self.brake = 0.0

        elif action == 1:  # Turn left
            self.gear = 0  # Ensure forward gear
            self.steering_angle = max(self.steering_angle + 0.5, 0.0)

        elif action == 2:  # Turn right
            self.gear = 0  # Ensure forward gear
            self.steering_angle = min(self.steering_angle - 0.5, 0.0)

        elif action == 3:  # Brake
            self.gear = 0  # Ensure forward gear
            self.brake = 300.0
            self.throttle = 0.0

        elif action == 4:  # Reset steering
            self.gear = 0  # Ensure forward gear
            self.steering_angle = 0.0

        elif action == 5:  # Reverse gear
            self.gear = 1
            self.throttle = min(self.throttle + 0.01, 0.5)  # Reverse with limited speed

        # Publish control commands
        self.steering_pub.publish(Float64(data=self.steering_angle))
        self.throttle_pub.publish(Float64(data=self.throttle))
        self.brake_pub.publish(Float64(data=self.brake))
        self.gear_pub.publish(UInt8(data=self.gear))



    def calculate_reward(self, action, lateral_distance, car_velocity, is_off_track):
        if is_off_track > 0.0:  # Use float value instead of boolean
            return -10.0  # Penalty for going off the track

        reward = 0.0

        # Reward for higher velocity (with a cap to avoid excessive speed)
        reward += min(car_velocity, 5.0) * 0.1

        # Reward based on lateral distance from lane center
        if abs(lateral_distance) < 0.5:
            reward += 2.0  # Center of lane
        elif abs(lateral_distance) < 1.0:
            reward += 1.0  # Near center
        else:
            reward -= 2.0  # Far from center

        if action in [1, 2]:  # Turning left or right
            reward -= 0.2  # Reduced penalty for steering

        if action == 5:  # Reverse gear
            reward -= 2.0  # Increased penalty for using reverse gear

        return reward



    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        transitions = [self.memory[idx] for idx in batch]
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1

        # Update target network periodically
        if self.steps_done % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def run(self):
        rclpy.spin(self)


def main(args=None):
    rclpy.init(args=args)
    ddqn_node = DDQNNode()
    ddqn_node.run()
    ddqn_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
