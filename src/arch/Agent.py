import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Agent(nn.Module):
    def __init__(self, STATE_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)
        self.optimizer = optim.Adam(self.parameters(), lr=0.1)

    def forward(self, x):
        """Forward pass for Q-value estimation."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # no softmax; raw Q-values

    def act(self, state, epsilon=0.05):
        """Choose an action using epsilon-greedy strategy."""
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, 4, (1,)).item()
    
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.forward(state)
        return torch.argmax(q_values).item()

    def train_step(self, state, action, reward, next_state, done):
        """
        Perform one training step using Q-learning.
        state, next_state: list or array of floats
        action: int (0 to ACTION_SIZE-1)
        reward: float
        done: bool — whether episode ended
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor([[action]])
        reward = torch.tensor([[reward]])
        done = torch.tensor([[done]], dtype=torch.bool)

        # 1. Get predicted Q-values for current state
        q_values = self(state)

        # 2. Select the Q-value for the action taken
        current_q = q_values.gather(1, action)

        # 3. Compute the target Q-value
        with torch.no_grad():
            next_q = self(next_state)
            max_next_q = next_q.max(1, keepdim=True)[0]
            target_q = reward + (0.99 * max_next_q * (~done))  # no future reward if done

        # 4. Compute loss and backpropagate
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.state_dict(), path)