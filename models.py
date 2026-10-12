import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.conv = nn.Conv1d(64, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(64, 3)

    def forward(self, words, lengths):
        vectors = self.embedding(words)
        vectors = vectors.transpose(1, 2)
        features = torch.relu(self.conv(vectors))
        positions = torch.arange(features.size(2))
        valid_windows = (lengths - 2).clamp(min=1)
        mask = positions.unsqueeze(0) < valid_windows.unsqueeze(1)
        features = features.masked_fill(~mask.unsqueeze(1), -1e9)

        pooled = features.max(dim=2).values
        return self.output(self.dropout(pooled))


class LSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(64, 3)

    def forward(self, words, lengths):
        vectors = self.embedding(words)
        hidden, last_state = self.lstm(vectors)
        positions = torch.arange(words.size(1))
        mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        hidden = hidden * mask.unsqueeze(2)
        pooled = hidden.sum(dim=1) / lengths.unsqueeze(1)
        return self.output(self.dropout(pooled))
