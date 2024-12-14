import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from ddqn_audibot.ddqn_agent import DuelingDDQN  # Import your model class

# Hyperparameters
STATE_SIZE = (3, 128, 128)  # Input dimensions (channels, height, width)
ACTION_SIZE = 5  # Number of discrete actions
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 1000
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# Replay memory
memory = deque(maxlen=MEMORY_SIZE)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize networks
policy_net = DuelingDDQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net = DuelingDDQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

# Epsilon-greedy strategy
epsilon = EPSILON_START

# Functions
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(0, ACTION_SIZE)
    with torch.no_grad():
        return torch.argmax(policy_net(state)).item()

def store_transition(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def replay():
    if len(memory) < BATCH_SIZE:
        return

    batch = np.random.choice(len(memory), BATCH_SIZE, replace=False)
    states, actions, rewards, next_states, dones = zip(*[memory[i] for i in batch])

    states = torch.cat(states).to(device)
    actions = torch.tensor(actions).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards).to(device)
    next_states = torch.cat(next_states).to(device)
    dones = torch.tensor(dones).to(device)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].detach()
    target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

    loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def update_target_network():
    target_net.load_state_dict(policy_net.state_dict())

# Training loop (mockup example)
def train():
    global epsilon

    for episode in range(1000):  # Example: 1000 episodes
        # Initialize environment state (replace with actual environment interaction)
        state = torch.zeros(1, *STATE_SIZE).to(device)

        done = False
        while not done:
            action = select_action(state, epsilon)
            # Mockup next state, reward, and done signal (replace with actual environment interaction)
            next_state = torch.zeros(1, *STATE_SIZE).to(device)
            reward = 1.0
            done = np.random.rand() < 0.1  # Example termination condition

            store_transition(state, action, reward, next_state, done)
            state = next_state

            replay()

        # Update target network periodically
        if episode % TARGET_UPDATE_FREQ == 0:
            update_target_network()

        # Decay epsilon
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        print(f"Episode {episode + 1} completed. Epsilon: {epsilon:.4f}")

if __name__ == "__main__":
    train()
