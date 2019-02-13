import torch
from torch import nn 
import namedtensor
from namedtensor.nn import nn as namednn


# embedding_dim = 20
# hidden = 100
# seqlen = 32
# pad_i = TEXT.vocab.stoi['<pad>']

# class Pad(nn.Module):
#     def __init__(self, seqlen, pad_i):
#         super().__init__()
#         self.seqlen = seqlen
#         self.pad_i = pad_i
#     def forward(self, x):
#         init = torch.ones(self.seqlen, x.shape[1]) * pad_i
#         init[-x.shape[0]:, :] = x
#         return init.long()

    
# class Flatten(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         return x.permute(1,0,2).flatten(start_dim=1,end_dim=2)

# def get_nn_lang_model(embedding_dim, hidden, TEXT, n_hidden_layers=1, seqlen=32):
#     pad_i = TEXT.vocab.stoi['<pad>']
#     if n_hidden_layers == 1:
#         return nn.Sequential(
#                 Pad(seqlen, pad_i),
#                 nn.Embedding(len(TEXT.vocab), embedding_dim),
#                 Flatten(),
#                 nn.Linear(seqlen * embedding_dim, hidden),
#                 nn.Tanh(),
#                 nn.Linear(hidden, len(TEXT.vocab)),
#                 nn.Softmax(dim=0)
#             )
#     else:
#         layers = []
#         for k in range(n_hidden_layers-1):
#             layers.append(nn.Linear(hidden, hidden))
#             layers.append(nn.Tanh())
#         return nn.Sequential(
#                 Pad(seqlen, pad_i),
#                 nn.Embedding(len(TEXT.vocab), embedding_dim),
#                 Flatten(),
#                 nn.Linear(seqlen * embedding_dim, hidden),
#                 nn.Tanh(),
#                 *layers,
#                 nn.Linear(hidden, len(TEXT.vocab)),
#                 nn.Softmax(dim=0)
#             )


class NNLangModel(namednn.Module):
    def __init__(self, TEXT, embedding_dim, kernel_size, hidden, dropout=.5):
        super().__init__()
        V = len(TEXT.vocab)
        pad_idx = TEXT.vocab.stoi['<pad>']

        self.embed = namednn.Embedding(num_embeddings=V,
                                       embedding_dim=embedding_dim,
                                       padding_idx=pad_idx)
        self.conv = namednn.Conv1d(embedding_dim, embedding_dim, 
                                   kernel_size=kernel_size).spec('embedding', 'seqlen')

        self.w1 = namednn.Linear(embedding_dim, hidden).spec('embedding', 'hidden')
        self.w2 = namednn.Linear(hidden, hidden).spec('hidden', 'hidden2')
        self.w3 = namednn.Linear(hidden, V).spec('hidden2', 'classes')
        self.dropout = namednn.Dropout(dropout)
    
    def forward(self, batch_text):
        embedded = self.embed(batch_text)
        conved = self.conv(embedded)
        h1 = self.w1(conved).tanh()
        h2 = self.w2(self.dropout(h1)).tanh()
        out = self.w3(self.dropout(h2))
        return out

nn_lang_loss = namednn.CrossEntropyLoss().spec('seqlen')
def nn_lang_loss_fn(model, batch):
    output = model(batch.text)
    size = output.shape['seqlen']
    target_size = batch.target.size('seqlen')
    target = (batch.target[{'seqlen' : slice(target_size-size, target_size)}]
                   .transpose('batch', 'seqlen'))
    return nn_lang_loss(output, target)