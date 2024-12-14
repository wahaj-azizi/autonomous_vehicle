# import rclpy
# from rclpy.node import Node
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from collections import deque
# from sensor_msgs.msg import Image
# from std_msgs.msg import Float64, UInt8
# from cv_bridge import CvBridge
# import cv2

# class DuelingDDQN(nn.Module):
#     def __init__(self, input_shape, action_size):
#         super(DuelingDDQN, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )
#         self.fc_input_size = self._get_conv_output(input_shape)
#         self.fc = nn.Sequential(
#             nn.Linear(self.fc_input_size, 512),
#             nn.ReLU()
#         )
#         self.value = nn.Linear(512, 1)
#         self.advantage = nn.Linear(512, action_size)

#     def _get_conv_output(self, shape):
#         with torch.no_grad():
#             o = self.conv(torch.zeros(1, *shape))
#         return int(np.prod(o.size()))

#     def forward(self, x):
#         x = x.float() / 255.0  # Ensure input is float32
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         value = self.value(x)
#         advantage = self.advantage(x)
#         return value + advantage - advantage.mean()

# class DDQNNode(Node):
#     def __init__(self):
#         super().__init__('ddqn_agent')
#         self.bridge = CvBridge()
#         self.state_size = (3, 128, 128)  # Input shape (channels, height, width)
#         self.action_size = self.declare_parameter('action_size', 5).value
#         self.gamma = self.declare_parameter('gamma', 0.99).value
#         self.epsilon = self.declare_parameter('epsilon', 1.0).value
#         self.epsilon_decay = self.declare_parameter('epsilon_decay', 0.995).value
#         self.min_epsilon = self.declare_parameter('min_epsilon', 0.01).value
#         self.learning_rate = self.declare_parameter('learning_rate', 0.001).value
#         self.batch_size = self.declare_parameter('batch_size', 32).value

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.policy_net = DuelingDDQN(self.state_size, self.action_size).to(self.device)
#         self.target_net = DuelingDDQN(self.state_size, self.action_size).to(self.device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval()
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

#         self.memory = deque(maxlen=20000)
#         self.steps_done = 0

#         # Publishers for control commands
#         self.steering_pub = self.create_publisher(Float64, '/audibot/steering_cmd', 10)
#         self.throttle_pub = self.create_publisher(Float64, '/audibot/throttle_cmd', 10)
#         self.brake_pub = self.create_publisher(Float64, '/audibot/brake_cmd', 10)
#         self.gear_pub = self.create_publisher(UInt8, '/audibot/gear_cmd', 10)

#         # Subscriber for the camera feed
#         self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
#         self.get_logger().info("DDQN Node initialized.")

#         # Vehicle state variables
#         self.steering_angle = 0.0  # Radians
#         self.throttle = 0.0  # 0 to 1
#         self.brake = 0.0  # Nm
#         self.gear = 0  # Drive = 0, Reverse = 1

#     def image_callback(self, msg):
#         try:
#             cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
#             cv_image = cv2.resize(cv_image, (128, 128))
#             state = np.transpose(cv_image, (2, 0, 1))  # Convert to CHW format
#             state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

#             action = self.select_action(state)
#             self.execute_action(action)

#             reward = self.calculate_reward(action)  # Replace with reward function
#             next_state = state  # In practice, this should come from the next frame
#             self.store_transition(state, action, reward, next_state, done=False)

#             if len(self.memory) > self.batch_size:
#                 self.replay()
#         except Exception as e:
#             self.get_logger().error(f"Error in image_callback: {e}")

#     def select_action(self, state):
#         if np.random.rand() <= self.epsilon:
#             return np.random.randint(0, self.action_size)
#         with torch.no_grad():
#             return torch.argmax(self.policy_net(state)).item()

#     def execute_action(self, action):
#         if action == 0:  # Move forward (increase throttle)
#             self.throttle = min(self.throttle + 0.05, 1.0)
#             self.brake = 0.0

#         elif action == 1:  # Turn left
#             self.steering_angle = max(self.steering_angle + 0.2, 0.0)

#         elif action == 2:  # Turn right
#             self.steering_angle = min(self.steering_angle - 0.2, 0.0)

#         elif action == 3:  # Brake
#             self.brake = 300.0
#             self.throttle = 0.0

#         elif action == 4:  # Reset steering
#             self.steering_angle = 0.0

#         # Publish control commands
#         self.steering_pub.publish(Float64(data=self.steering_angle))
#         self.throttle_pub.publish(Float64(data=self.throttle))
#         self.brake_pub.publish(Float64(data=self.brake))
#         self.gear_pub.publish(UInt8(data=self.gear))

#     def calculate_reward(self, action):
#         # Placeholder reward function, replace with a more sophisticated one
#         return 1.0 if action == 0 else -1.0

#     def store_transition(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def replay(self):
#         batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
#         states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])

#         states = torch.cat(states).to(self.device)
#         actions = torch.tensor(actions).unsqueeze(1).to(self.device)
#         rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
#         next_states = torch.cat(next_states).to(self.device)
#         dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

#         q_values = self.policy_net(states).gather(1, actions)
#         next_q_values = self.target_net(next_states).max(1)[0].detach()
#         target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

#         loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         self.steps_done += 1
#         if self.steps_done % 100 == 0:
#             self.target_net.load_state_dict(self.policy_net.state_dict())

#         if self.epsilon > self.min_epsilon:
#             self.epsilon *= self.epsilon_decay


# def main(args=None):
#     rclpy.init(args=args)
#     node = DDQNNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         if not rclpy.is_shutdown():
#             rclpy.shutdown()


# if __name__ == '__main__':
#     main()
# import rclpy
# from rclpy.node import Node
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from collections import deque
# from sensor_msgs.msg import Image
# from std_msgs.msg import Float64, UInt8
# from cv_bridge import CvBridge
# import cv2

# class DuelingDDQN(nn.Module):
#     def __init__(self, input_shape, action_size):
#         super(DuelingDDQN, self).__init__()
#         # Convolutional layers to process input image
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )
#         # Compute the size of the fully connected layer input
#         self.fc_input_size = self._get_conv_output(input_shape)
#         # Fully connected layers to process features from the convolutional layers
#         self.fc = nn.Sequential(
#             nn.Linear(self.fc_input_size, 512),
#             nn.ReLU()
#         )
#         # Value and advantage streams for Dueling architecture
#         self.value = nn.Linear(512, 1)
#         self.advantage = nn.Linear(512, action_size)

#     def _get_conv_output(self, shape):
#         # Helper method to determine the size of the output from the conv layers
#         with torch.no_grad():
#             o = self.conv(torch.zeros(1, *shape))
#         return int(np.prod(o.size()))

#     def forward(self, x):
#         # Normalize input and pass through the network
#         x = x.float() / 255.0  # Scale pixel values to [0, 1]
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)  # Flatten the tensor
#         x = self.fc(x)
#         # Compute value and advantage streams
#         value = self.value(x)
#         advantage = self.advantage(x)
#         return value + advantage - advantage.mean()  # Combine streams for final Q-value

# class DDQNNode(Node):
#     def __init__(self):
#         super().__init__('ddqn_agent')
#         # Initialize variables and parameters
#         self.bridge = CvBridge()
#         self.state_size = (3, 128, 128)  # Input shape (channels, height, width)
#         self.action_size = self.declare_parameter('action_size', 5).value
#         self.gamma = self.declare_parameter('gamma', 0.99).value  # Discount factor
#         self.epsilon = self.declare_parameter('epsilon', 1.0).value  # Exploration rate
#         self.epsilon_decay = self.declare_parameter('epsilon_decay', 0.995).value
#         self.min_epsilon = self.declare_parameter('min_epsilon', 0.01).value
#         self.learning_rate = self.declare_parameter('learning_rate', 0.001).value
#         self.batch_size = self.declare_parameter('batch_size', 32).value

#         # Set device for computations (use GPU if available)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # Initialize policy and target networks
#         self.policy_net = DuelingDDQN(self.state_size, self.action_size).to(self.device)
#         self.target_net = DuelingDDQN(self.state_size, self.action_size).to(self.device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval()  # Target network is used only for inference
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

#         # Replay memory to store experiences
#         self.memory = deque(maxlen=20000)
#         self.steps_done = 0

#         # Publishers for control commands
#         self.steering_pub = self.create_publisher(Float64, '/audibot/steering_cmd', 10)
#         self.throttle_pub = self.create_publisher(Float64, '/audibot/throttle_cmd', 10)
#         self.brake_pub = self.create_publisher(Float64, '/audibot/brake_cmd', 10)
#         self.gear_pub = self.create_publisher(UInt8, '/audibot/gear_cmd', 10)

#         # Subscriber for the camera feed
#         self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
#         self.get_logger().info("DDQN Node initialized.")

#         # Vehicle state variables
#         self.steering_angle = 0.0  # Radians
#         self.throttle = 0.0  # 0 to 1
#         self.brake = 0.0  # Nm
#         self.gear = 0  # Drive = 0, Reverse = 1

#     def image_callback(self, msg):
#         self.get_logger().info("Image callback triggered.")
#         try:
#             # Convert ROS image to OpenCV format
#             cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
#             self.get_logger().info("Image converted to OpenCV format.")
#             cv_image = cv2.resize(cv_image, (128, 128))  # Resize to match network input
#             self.get_logger().info("Image resized to 128x128.")
#             # Preprocess image for the model
#             state = np.transpose(cv_image, (2, 0, 1))  # Convert to CHW format
#             state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

#             # Select and execute an action based on the current state
#             action = self.select_action(state)
#             self.get_logger().info(f"Selected action: {action}")
#             self.execute_action(action)

#             # Calculate reward and store transition
#             reward = self.calculate_reward(action)  # Replace with a custom reward function
#             self.get_logger().info(f"Calculated reward: {reward}")
#             next_state = state  # In practice, this should come from the next frame
#             self.store_transition(state, action, reward, next_state, done=False)
#             self.get_logger().info("Transition stored in memory.")

#             # Perform experience replay if memory is sufficient
#             if len(self.memory) > self.batch_size:
#                 self.get_logger().info("Starting experience replay.")
#                 self.replay()
#         except Exception as e:
#             self.get_logger().error(f"Error in image_callback: {e}")

#     def select_action(self, state):
#         # Epsilon-greedy policy for exploration vs. exploitation
#         self.get_logger().info(f"Epsilon value: {self.epsilon}")
#         if np.random.rand() <= self.epsilon:
#             self.get_logger().info("Choosing random action (exploration).")
#             return np.random.randint(0, self.action_size)  # Explore
#         with torch.no_grad():
#             self.get_logger().info("Choosing action based on policy network (exploitation).")
#             return torch.argmax(self.policy_net(state)).item()  # Exploit

#     def execute_action(self, action):
#         self.get_logger().info(f"Executing action: {action}")
#         # Define behavior for each action
#         if action == 0:  # Move forward (increase throttle)
#             self.throttle = min(self.throttle + 0.01, 1.0)
#             self.brake = 0.0

#         elif action == 1:  # Turn left
#             self.steering_angle = max(self.steering_angle + 0.2, 0.0)

#         elif action == 2:  # Turn right
#             self.steering_angle = min(self.steering_angle - 0.2, 0.0)

#         elif action == 3:  # Brake
#             self.brake = 300.0
#             self.throttle = 0.0

#         elif action == 4:  # Reset steering
#             self.steering_angle = 0.0

#         # Publish control commands to the vehicle
#         self.steering_pub.publish(Float64(data=self.steering_angle))
#         self.get_logger().info(f"Steering angle published: {self.steering_angle}")
#         self.throttle_pub.publish(Float64(data=self.throttle))
#         self.get_logger().info(f"Throttle published: {self.throttle}")
#         self.brake_pub.publish(Float64(data=self.brake))
#         self.get_logger().info(f"Brake published: {self.brake}")
#         self.gear_pub.publish(UInt8(data=self.gear))
#         self.get_logger().info(f"Gear published: {self.gear}")

#     def calculate_reward(self, action):
#         # Placeholder reward function, replace with a more sophisticated one
#         reward = 1.0 if action == 0 else -1.0
#         self.get_logger().info(f"Reward calculated for action {action}: {reward}")
#         return reward

#     def store_transition(self, state, action, reward, next_state, done):
#         # Store experience in the replay memory
#         self.memory.append((state, action, reward, next_state, done))
#         self.get_logger().info("Stored transition in replay memory.")

#     def replay(self):
#         # Sample a batch of experiences from memory
#         self.get_logger().info("Sampling a batch of experiences for replay.")
#         batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
#         states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])

#         # Convert batch data to tensors
#         states = torch.cat(states).to(self.device)
#         actions = torch.tensor(actions).unsqueeze(1).to(self.device)
#         rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
#         next_states = torch.cat(next_states).to(self.device)
#         dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

#         # Compute current and target Q-values
#         q_values = self.policy_net(states).gather(1, actions)
#         next_q_values = self.target_net(next_states).max(1)[0].detach()
#         target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

#         # Compute loss and update the policy network
#         loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))
#         self.get_logger().info(f"Loss calculated: {loss.item()}")
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         # Update target network periodically
#         self.steps_done += 1
#         if self.steps_done % 100 == 0:
#             self.get_logger().info("Updating target network.")
#             self.target_net.load_state_dict(self.policy_net.state_dict())

#         # Decay epsilon for exploration
#         if self.epsilon > self.min_epsilon:
#             self.epsilon *= self.epsilon_decay
#             self.get_logger().info(f"Epsilon decayed to: {self.epsilon}")

# def main(args=None):
#     rclpy.init(args=args)  # Initialize the ROS 2 Python client library
#     node = DDQNNode()
#     try:
#         rclpy.spin(node)  # Keep the node running
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()  # Destroy the node when exiting
#         if not rclpy.is_shutdown():
#             rclpy.shutdown()  # Shutdown ROS 2 if not already done

# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from sensor_msgs.msg import Image
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

    # def forward(self, x):
    #     x = x.float() / 255.0  # Normalize input to range [0, 1]
    #     x = self.conv(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)
    #     value = self.value(x)
    #     advantage = self.advantage(x)
    #     return value + advantage - advantage.mean(dtype=torch.float32)
    def forward(self, x):
        x = x.float() / 255.0  # Normalize input to range [0, 1]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        value = self.value(x)
        advantage = self.advantage(x)

        # Debugging prints
        print(f"Value dtype: {value.dtype}, Shape: {value.shape}")
        print(f"Advantage dtype: {advantage.dtype}, Shape: {advantage.shape}")
        print(f"Advantage mean dtype: {advantage.mean().dtype}")

        return value + advantage - advantage.mean()



class DDQNNode(Node):
    def __init__(self):
        super().__init__('ddqn_agent')
        self.bridge = CvBridge()
        self.state_size = (3, 128, 128)  # Input shape (channels, height, width)
        self.action_size = self.declare_parameter('action_size', 5).value
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

        self.memory = deque(maxlen=20000)
        self.steps_done = 0

        # Publishers for control commands
        self.steering_pub = self.create_publisher(Float64, '/audibot/steering_cmd', 10)
        self.throttle_pub = self.create_publisher(Float64, '/audibot/throttle_cmd', 10)
        self.brake_pub = self.create_publisher(Float64, '/audibot/brake_cmd', 10)
        self.gear_pub = self.create_publisher(UInt8, '/audibot/gear_cmd', 10)

        # Subscriber for the camera feed
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.get_logger().info("DDQN Node initialized.")

        # Vehicle state variables
        self.steering_angle = 0.0  # Radians
        self.throttle = 0.0  # 0 to 1
        self.brake = 0.0  # Nm
        self.gear = 0  # Drive = 0, Reverse = 1

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            cv_image = cv2.resize(cv_image, (128, 128))
            state = np.transpose(cv_image, (2, 0, 1))  # Convert to CHW format
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Log CNN's output
            cnn_output = self.policy_net(state)
            self.get_logger().info(f"CNN Output: {cnn_output}")

            # Select and log the action
            action = self.select_action(state)
            self.get_logger().info(f"Selected Action: {action}")

            # Execute the action
            self.execute_action(action)

            # Calculate and log the reward
            reward = self.calculate_reward(action)  # Replace with reward function
            self.get_logger().info(f"Reward: {reward}")

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
            self.throttle = min(self.throttle + 0.01, 1.0)
            self.brake = 0.0

        elif action == 1:  # Turn left
            self.steering_angle = max(self.steering_angle - 0.2, 0.0)

        elif action == 2:  # Turn right
            self.steering_angle = min(self.steering_angle + 0.2, 0.0)

        elif action == 3:  # Brake
            self.brake = 300.0
            self.throttle = 0.0

        elif action == 4:  # Reset steering
            self.steering_angle = 0.0

        # Publish control commands
        self.steering_pub.publish(Float64(data=self.steering_angle))
        self.throttle_pub.publish(Float64(data=self.throttle))
        self.brake_pub.publish(Float64(data=self.brake))
        self.gear_pub.publish(UInt8(data=self.gear))

    def calculate_reward(self, action):
        # Placeholder reward function, replace with a more sophisticated one
        return 1.0 if action == 0 else -1.0

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])

        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        # Log loss for debugging
        self.get_logger().info(f"Training Loss: {loss.item()}")

def main(args=None):
    rclpy.init(args=args)
    node = DDQNNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if not rclpy.is_shutdown():
          rclpy.shutdown()

if __name__ == '__main__':
    main()
