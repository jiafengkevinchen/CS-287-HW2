from namedtensor import NamedTensor
import torch

class LSTM_prior:
    def __init__(self, lstm, unigram):
        self.lstm = lstm.cuda()
        self.unigram = NamedTensor(torch.log(unigram + 1e-6).cuda(), names='classes')

    def __call__(self, batch_text):
        return self.lstm(batch_text) + self.unigram
