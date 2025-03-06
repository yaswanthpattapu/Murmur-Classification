#create a model that take a model as input and add a fully connected layer on top of it and return the prob distribution for each class using softmax
import torch.nn as nn
import torch
import torch.nn.functional as F


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=1)



class Wav2Vec2LL(nn.Module):
    def __init__(self, model, num_classes):
        super(Wav2Vec2LL, self).__init__()
        self.model = model
        
        self.mean_pool = MeanPooling()
        self.fc1 = nn.Linear(768, 128)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        features = self.model(x)
        # print(features.shape)
        # logits = self.fc(features)
        # pooled_logits = self.mean_pool(logits)
        # print(features.last_hidden_state.shape)
        print(features.last_hidden_state.shape)
        pooled = self.mean_pool(features.last_hidden_state)

        # print(pooled.shape)
        pooled = self.tanh(pooled)
        pooled_logits = self.fc1(pooled)
        pooled_logits = self.tanh(pooled_logits)
        pooled_logits = self.fc2(pooled_logits)
        # print(pooled_logits.shape)
        # return F.log_softmax(pooled_logits, dim=-1)
        return pooled_logits
        # return pooled_logits
