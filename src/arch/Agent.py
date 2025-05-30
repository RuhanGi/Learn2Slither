import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
PURPLE  = "\033[35m"
CYAN    = "\033[36m"
GRAY    = "\033[90m"
BLACK   = "\033[30m"
WHITE   = "\033[37m"
RESET   = "\033[0m"

class QNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 4):
        super().__init__()
        self.l1 = nn.Linear(in_features=in_dim, out_features=100)
        self.l2 = nn.Linear(in_features=100, out_features=16)
        self.l3 = nn.Linear(in_features=16, out_features=out_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.l1, self.l2, self.l3]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x


class Agent:
    def __init__(self, sf):
        self.gamma = 0.9
        self.lr = 0.01

        self.epsilon = 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.action_size = 4
        self.state_features = sf

        self.qnet = QNet(in_dim=self.state_features, out_dim=self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        # self.criterion = nn.SmoothL1Loss()

    def act(self, state) -> int:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, 4, (1,)).item()
        else:
            with torch.no_grad():
                qs = self.qnet(torch.tensor(state, dtype=torch.float32))
                return torch.argmax(qs).item()

    def train_step(
            self,
            state,
            action: int,
            reward: float,
            next_state,
            done: bool
    ):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        if done:
            next_q = torch.tensor(0.0)
        else:
            with torch.no_grad():
                next_q = self.qnet(next_state).max()

        target = reward + self.gamma * next_q
        qs = self.qnet(state)  # (1, action_size)
        q = qs.squeeze(0)[action]  # (1, 4) -> 1次元(4,)

        loss = self.criterion(q, target)

        # loss = torch.clamp(loss, min=-1.0, max=1.0)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=1.0)

        self.optimizer.step()
        return loss.item()

# class Agent(nn.Module):
#     def __init__(self, STATE_SIZE):
#         super().__init__()
#         self.fc1 = nn.Linear(STATE_SIZE, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 4)
#         self.optimizer = optim.Adam(self.parameters(), lr=0.01)

#     def forward(self, x):
#         """Forward pass for Q-value estimation."""
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)  # no softmax; raw Q-values

#     def act(self, state, epsilon=0.05):
#         """Choose an action using epsilon-greedy strategy."""
#         if torch.rand(1).item() < epsilon:
#             return torch.randint(0, 4, (1,)).item()
    
#         state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#         q_values = self.forward(state)
#         return torch.argmax(q_values).item()

#     def train_step(self, state, action, reward, next_state, done):
#         """
#         Perform one training step using Q-learning.
#         state, next_state: list or array of floats
#         action: int (0 to ACTION_SIZE-1)
#         reward: float
#         done: bool — whether episode ended
#         """
#         state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#         next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
#         action = torch.tensor([[action]])
#         reward = torch.tensor([[reward]])
#         done = torch.tensor([[done]], dtype=torch.bool)

#         # 1. Get predicted Q-values for current state
#         q_values = self(state)

#         # 2. Select the Q-value for the action taken
#         current_q = q_values.gather(1, action)

#         # 3. Compute the target Q-value
#         with torch.no_grad():
#             next_q = self(next_state)
#             max_next_q = next_q.max(1, keepdim=True)[0]
#             target_q = reward + (0.99 * max_next_q * (~done))  # no future reward if done

#         # 4. Compute loss and backpropagate
#         loss = F.mse_loss(current_q, target_q)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

    def load(self, path):
        if path is not None:
            self.qnet.load_state_dict(torch.load(path))

    def save(self, path):
        if path is not None:
            torch.save(self.qnet.state_dict(), path)
            print(GREEN + "Model Saved!", RESET)
    