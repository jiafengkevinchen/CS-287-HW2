import torch
from torch import nn
import namedtensor
from namedtensor.nn import nn as namednn


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

nn_lang_loss = namednn.CrossEntropyLoss().spec('classes')
def nn_lang_loss_fn(model, batch):
    output = model(batch.text)
    size = output.shape['seqlen']
    target_size = batch.target.size('seqlen')
    target = (batch.target[{'seqlen' : slice(target_size-size, target_size)}])
    return nn_lang_loss(output, target)
