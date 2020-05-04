import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, n_input, hidden_size, n_classes, n_layers):
        super().__init__()
        self.n_input = n_input
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size=n_input, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_out = h_n.view(-1, self.hidden_size)
        out = self.fc(h_out)

        return out
