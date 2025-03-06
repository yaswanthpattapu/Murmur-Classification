import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)        
        output = torch.matmul(attention_weights, value)
        return output

class LSTMWithScaledDotProductAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, bidirectional, num_layers=2):
        super(LSTMWithScaledDotProductAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5, bidirectional=bidirectional)
        self.bidirectional=2 if bidirectional else 1
        self.attention = ScaledDotProductAttention(hidden_dim*self.bidirectional)
        self.fc = nn.Linear(hidden_dim*self.bidirectional, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*self.bidirectional, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers*self.bidirectional, x.size(0), self.hidden_dim).to(x.device)        
        out, _ = self.lstm(x, (h0, c0))
        output = self.attention(out, out, out)
        output = self.fc(output[:, -1, :])
        return F.log_softmax(output, dim=-1)


if __name__=="__main__":
    input_dim = 100
    hidden_dim = 64
    num_classes = 2
    num_layers = 3
    bidirectional=True
    model = LSTMWithScaledDotProductAttention(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, bidirectional=bidirectional, num_layers=num_layers)

    batch_size = 32
    seq_length = 20
    input_array = torch.randn(batch_size, seq_length, input_dim)

    output = model(input_array)
    print("Output shape:", output.shape)