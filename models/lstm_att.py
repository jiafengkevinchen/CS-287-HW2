from namedtensor.nn import nn as nnn
from namedtensor import ntorch
import torch
from namedtensor import NamedTensor
from numpy import inf


class MaskedAttention(nnn.Module):
    def __init__(self, cuda=True):
        super().__init__()
        self.cuda_enabled = cuda

    def forward(self, hidden):
        dotted = (hidden * hidden.rename("seqlen", "seqlen2")).sum("embedding")
        mask = torch.arange(hidden.size('seqlen'))
        mask = (NamedTensor(mask, names='seqlen') < NamedTensor(mask, names='seqlen2')).float()
        mask[mask.byte()] = -inf
        if self.cuda_enabled:
            attn = ((dotted + mask.cuda()) / (hidden.size("embedding") ** .5)).softmax('seqlen2')
        else:
            attn = ((dotted + mask) / (hidden.size("embedding") ** .5)).softmax('seqlen2')
        return (attn * hidden.rename('seqlen', 'seqlen2')).sum('seqlen2')

class LSTM_att(nnn.Module):
    """
    LSTM implementation for sentence completion.
    """
    def __init__(self, TEXT,
                 embedding_dim=100,
                 hidden_dim=150,
                 num_layers=1,
                 dropout=0,
                 nn_dropout=.5,
                 **kwargs):
        super().__init__()

        pad_idx = TEXT.vocab.stoi['<pad>']

        self.embed = nnn.Embedding(num_embeddings=len(TEXT.vocab),
                                   embedding_dim=embedding_dim,
                                   padding_idx=pad_idx)

        self.lstm = nnn.LSTM(input_size=embedding_dim,
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             dropout=dropout) \
                        .spec("embedding", "seqlen")


        self.w1 = (nnn.Linear(in_features=hidden_dim, out_features=hidden_dim)
                   .spec("embedding", "embedding"))
        self.w2 = (nnn.Linear(in_features=hidden_dim, out_features=hidden_dim)
                   .spec("embedding", "embedding"))
        self.w3 = (nnn.Linear(in_features=hidden_dim, out_features=hidden_dim)
                   .spec("embedding", "embedding"))


        self.lins = [self.w1, self.w2, self.w3]
        self.attn = MaskedAttention(**kwargs)

        h_len = len(self.lins) + 2
        self.w = nnn.Linear(in_features=hidden_dim * h_len,
                            out_features=len(TEXT.vocab)) \
                        .spec("embedding", "classes")
        self.dropout = nnn.Dropout(nn_dropout)


    def forward(self, batch_text):
        embedded = self.embed(batch_text)
        H, _ = self.lstm(embedded)
        joint = ntorch.cat([H, self.attn(H)] + [self.attn(l(H)) for l in self.lins], "embedding")
        log_probs = self.w(self.dropout(joint))
        return log_probs


ce_loss = nnn.CrossEntropyLoss().spec('classes')

def lstm_loss(model, batch):
    """
    Calculate loss of the model on a batch.
    """
    return ce_loss(model(batch.text), batch.target)
