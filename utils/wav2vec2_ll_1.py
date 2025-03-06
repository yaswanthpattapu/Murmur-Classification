#create a model that take a model as input and add a fully connected layer on top of it and return the prob distribution for each class using softmax
import torch.nn as nn
import torch.nn.functional as F
class Wav2Vec2LL(nn.Module):
    def __init__(self, model, num_classes):
        super(Wav2Vec2LL, self).__init__()
        self.model = model
        print(f"hidden size - {self.model.config.hidden_size}")
        # self.fc = nn.Linear(model.config.hidden_size, num_classes)
        self.fc = nn.Linear(32, num_classes)
        
    def forward(self, x):
        out = self.model(x)
        #dimensions of out - (batch_size, seq_len, hidden_size)
        # print(out.last_hidden_state.shape)
        print("in model")
        out = out.logits
        print(out.shape)
        out = self.fc(out)
        print(out.shape)

        # return out.transpose(0,1)
        return out