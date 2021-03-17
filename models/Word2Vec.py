import torch as torch
from torch import nn

class Word2Vec(nn.Module):
    def __init__(self, features, embedding_size):
        super().__init__()
        initrange = 0.5 / embedding_size
        self.fc1 = nn.Linear(features, embedding_size)
        self.fc2 = nn.Linear(embedding_size, features)


    def forward(self, one_hot):
        x = self.fc1(one_hot.float())
        x = self.fc2(x)
        log_softmax = torch.nn.functional.log_softmax(x, dim=1)
        return log_softmax
      