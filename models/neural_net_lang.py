import torch
from torch import nn 

# embedding_dim = 20
# hidden = 100
# seqlen = 32
# pad_i = TEXT.vocab.stoi['<pad>']

class Pad(nn.Module):
    def __init__(self, seqlen, pad_i):
        super().__init__()
        self.seqlen = seqlen
        self.pad_i = pad_i
    def forward(self, x):
        init = torch.ones(self.seqlen, x.shape[1]) * pad_i
        init[-x.shape[0]:, :] = x
        return init.long()

    
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.permute(1,0,2).flatten(start_dim=1,end_dim=2)

def get_nn_lang_model(embedding_dim, hidden, TEXT, n_hidden_layers=1, seqlen=32):
    pad_i = TEXT.vocab.stoi['<pad>']
    if n_hidden_layers == 1:
        return nn.Sequential(
                Pad(seqlen, pad_i),
                nn.Embedding(len(TEXT.vocab), embedding_dim),
                Flatten(),
                nn.Linear(seqlen * embedding_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, len(TEXT.vocab)),
                nn.Softmax(dim=0)
            )
    else:
        layers = []
        for k in range(n_hidden_layers-1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        return nn.Sequential(
                Pad(seqlen, pad_i),
                nn.Embedding(len(TEXT.vocab), embedding_dim),
                Flatten(),
                nn.Linear(seqlen * embedding_dim, hidden),
                nn.Tanh(),
                *layers,
                nn.Linear(hidden, len(TEXT.vocab)),
                nn.Softmax(dim=0)
            )
    