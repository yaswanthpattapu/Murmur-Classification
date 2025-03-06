#create a model that take a model as input and add a fully connected layer on top of it and return the prob distribution for each class using softmax
import torch.nn as nn
import torch.nn.functional as F
class Wav2Vec2LL(nn.Module):
    def __init__(self, model, num_classes):
        super(Wav2Vec2LL, self).__init__()
        self.model = model
        self.fc1 = nn.Linear(768*124, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        out = self.model(x)
        out = out.last_hidden_state
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        # out = F.log_softmax(out, dim=-1)
        return out
