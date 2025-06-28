import torch
import torch.nn as nn
import torch.optim as optim


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

    def act(self, state, args) -> int:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        with torch.no_grad():
            qs = self.qnet(torch.tensor(state, dtype=torch.float32))
            if not args.nolearn and torch.rand(1).item() < self.epsilon:
                sorted_indices = torch.argsort(qs, descending=True).tolist()
                return sorted_indices[torch.randint(1, 3, (1,)).item()]
            return torch.argmax(qs).item()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        if done:
            next_q = torch.tensor(0.0)
        else:
            with torch.no_grad():
                next_q = self.qnet(next_state).max()

        target = reward + self.gamma * next_q
        qs = self.qnet(state)
        q = qs.squeeze(0)[action]
        loss = self.criterion(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def load(self, path):
        if path is not None:
            self.qnet.load_state_dict(torch.load(path))

    def save(self, path):
        if path is not None:
            torch.save(self.qnet.state_dict(), path)
            print("\033[32m" + "Model Saved!", "\033[0m")
