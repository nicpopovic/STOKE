import torch


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, cuda=False):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)  # Input layer to hidden layer
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)  # Hidden layer to output layer
        if cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.to(self.device)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
        return x
    

class MLPProbe(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, cuda=False):
        super(MLPProbe, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)  # Input layer to hidden layer
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)  # Hidden layer to output layer
        if cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.to(self.device)

    def forward(self, x):
        
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
                    
        return x
