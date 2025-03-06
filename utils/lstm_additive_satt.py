import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(True),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        outputs=(encoder_outputs * weights.unsqueeze(-1))
        outputs = outputs.sum(dim=1)
        return outputs

class LSTMWithAdditiveSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, bidirectional=True, num_layers=2):
        super(LSTMWithAdditiveSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5, bidirectional=bidirectional)
        self.bidirectional=2 if bidirectional else 1
        self.attention = SelfAttention(hidden_dim*self.bidirectional)
        self.fc = nn.Linear(hidden_dim*self.bidirectional, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*self.bidirectional, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers*self.bidirectional, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  
        out = self.attention(out)
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

if __name__=="__main__":
    input_dim = 100
    hidden_dim = 64
    num_classes = 2 
    num_layers = 2
    bidirectional=True
    model = LSTMWithAdditiveSelfAttention(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, bidirectional=bidirectional, num_layers=num_layers)

    batch_size = 32
    seq_length = 20
    input_array = torch.randn(batch_size, seq_length, input_dim)
    output = model(input_array)
    print("Output shape:", output.shape)