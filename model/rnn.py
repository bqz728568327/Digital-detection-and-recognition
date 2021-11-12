import torch

class RNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fcl = torch.nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, h_n = self.rnn(x, None)
        out = self.fcl(out[:, -1 ,:])
        return out