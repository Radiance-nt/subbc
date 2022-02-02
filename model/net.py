import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, state_space_size, action_space_size):
        super(Net, self).__init__()
        self.mlps = nn.Sequential(
            nn.Linear(state_space_size, 40),
            nn.ReLU(),
            # nn.BatchNorm1d(20),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, action_space_size),
            nn.Softmax()
        )
        # self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)

    def forward(self, x):
        output = self.mlps(x)
        return output
