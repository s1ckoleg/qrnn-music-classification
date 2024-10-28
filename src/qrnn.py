import torch
import torch.nn as nn


class QRNN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3):
        super(QRNN, self).__init__()
        self.conv1d = nn.Conv1d(input_size, hidden_size * 3, kernel_size=kernel_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x should be of shape (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)  # Convert to (batch_size, input_size, seq_len) for Conv1d
        conv_out = self.conv1d(x)
        # Split the convolution output into gates
        Z, F, O = torch.chunk(conv_out, 3, dim=1)
        Z = torch.tanh(Z)
        F = torch.sigmoid(F)
        O = torch.sigmoid(O)

        # Initialize hidden state
        h_t = torch.zeros(x.size(0), self.hidden_size, x.size(2)).to(x.device)
        # Sequentially update h_t
        for t in range(x.size(2)):
            h_t = F[:, :, t] * h_t + (1 - F[:, :, t]) * Z[:, :, t]
            h_t = O[:, :, t] * h_t

        return h_t[:, :, -1]  # Output the last time step


class QRNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(QRNNClassifier, self).__init__()
        self.qrnn = QRNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        x, _ = self.qrnn(x)  # Consider second output if it's there

        # Output the last time step
        x = x[:, -1, :]  # Take last hidden state
        x = self.fc(x)

        return x


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        out, h_n = self.rnn(x)

        # Use the output from the last time step
        out = out[:, -1, :]  # Get last time step output
        out = self.fc(out)

        return out

