import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        out, h_n = self.rnn(x)

        # Use the output from the last time step
        out = out[:, -1, :]  # Get last time step output
        out = self.fc(out)

        return out