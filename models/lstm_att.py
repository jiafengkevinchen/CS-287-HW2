from namedtensor.nn import nn as nnn
from namedtensor import ntorch

class LSTM_att(nnn.Module):
    """
    LSTM implementation for sentence completion.
    """
    def __init__(self, TEXT,
                 embedding_dim=100,
                 hidden_dim=150,
                 num_layers=1,
                 dropout=0):
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

        self.w = nnn.Linear(in_features=hidden_dim * 2,
                            out_features=len(TEXT.vocab)) \
                        .spec("embedding", "classes")



    def forward(self, batch_text):
        embedded = self.embed(batch_text)
        H, _ = self.lstm(embedded)

        sentence_len = H.size('seqlen')
        contexts = ntorch.zeros(*H.shape.values(), names=list(H.shape.keys())).cuda()
        for t in range(sentence_len):
            ht = H[{'seqlen':t}]
            context_mat = H[{'seqlen':slice(0,t)}]
            a_weights = (ht * context_mat).sum('embedding').softmax("seqlen")
            contexts[{'seqlen' : t}] = (a_weights * context_mat).sum('seqlen')

        joint = ntorch.cat([contexts, H], dim='embedding')
        log_probs = self.w(joint)
        return log_probs


ce_loss = nnn.CrossEntropyLoss().spec('batch')

def lstm_loss(model, batch):
    """
    Calculate loss of the model on a batch.
    """
    return ce_loss(model(batch.text), batch.target)
