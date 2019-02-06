import torch
from torch import Tensor
from torch import sparse as sp
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import numpy as np
from torch import nn

def sparse_select(dims, indices, t):
    """
    Select sparse tensor t on on dimensions dims and indices chosen by indices.
    Equivalent to t[i_0, i_1, ...] where i_d = : if d is not in dims and
    i_(dims[j]) = indices[j] for all j
    """
    if type(dims) is not list:
        dims = [dims]
    if type(indices) is not list:
        indices = [indices]

    t_indices = t._indices()
    t_values = t._values()
    selector = torch.ones(t_indices.shape[-1]).byte()
    for dim, index in zip(dims, indices):
        selector = selector & (t._indices()[dim, :] == index)
    remaining_dimensions = list(filter(lambda x: x not in dims,
                                  range(t_indices.shape[0])))

    indices_selected = t_indices[:, selector][remaining_dimensions, :]
    values_selected = t_values[selector]
    new_shape = torch.Size(t.shape[d] for d in remaining_dimensions)

    out = sp.FloatTensor(indices_selected, values_selected, new_shape)
    return out



class Trigram:
    def __init__(self, TEXT):
        self.log_weights = torch.zeros(3, requires_grad=True)
        self.TEXT = TEXT
        self.V =  len(TEXT.vocab)

    def get_probabilities(self, train_iter):
        TEXT = self.TEXT
        V = self.V
        unigram = sp.FloatTensor(V)
        bigram = sp.FloatTensor(V,V)
        trigram = sp.FloatTensor(V,V,V)

        for batch in tqdm(train_iter):
            i = batch.text.flatten().unsqueeze(0)
            unigram_counts = sp.FloatTensor(
                 i, torch.ones(i.shape[1]), torch.Size([V])
            )

            unigram += unigram_counts

            ii = torch.stack([batch.text[:-1,:], batch.text[1:, :]]).view(2, -1)
            bigram_counts = sp.FloatTensor(
                 ii, torch.ones(ii.shape[-1]), torch.Size([V, V])
            )
            bigram += bigram_counts

            iii = torch.stack([batch.text[:-2,:], batch.text[1:-1, :], batch.text[2:, :]]).view(3, -1)
            trigram_counts = sp.FloatTensor(
                 iii, torch.ones(iii.shape[-1]), torch.Size([V, V, V])
            )
            trigram += trigram_counts

        unigram = unigram.coalesce()
        unigram = unigram / sp.sum(unigram)

        bigram = bigram.coalesce()
        trigram = trigram.coalesce()

        bigram_df = pd.DataFrame(np.hstack([bigram.indices().numpy().T,
                                bigram.values().numpy()[:, np.newaxis]]),
                     dtype=int, columns=['word1', 'word2', 'counts'])

        trigram_df = pd.DataFrame(np.hstack([trigram.indices().numpy().T,
                                trigram.values().numpy()[:, np.newaxis]]),
                     dtype=int, columns=['word1', 'word2', 'word3', 'counts'])

        bigram_df['prob'] = ((bigram_df['counts'] / bigram_df.groupby(['word1'])
                              .transform('sum')['counts']))

        bigram_ind = torch.from_numpy(bigram_df[['word1', 'word2']].values.T)
        bigram_val = torch.from_numpy(bigram_df['prob'].values)
        bigram = torch.sparse.FloatTensor(bigram_ind, bigram_val, bigram.shape)

        trigram_df['prob'] = (trigram_df['counts'] / trigram_df.groupby(['word1', 'word2'])
                              .transform('sum')['counts'])

        trigram_ind = torch.from_numpy(trigram_df[['word1', 'word2', 'word3']].values.T)
        trigram_val = torch.from_numpy(trigram_df['prob'].values)
        trigram = torch.sparse.FloatTensor(trigram_ind, trigram_val, trigram.shape)

        self.unigram = unigram.float()
        self.bigram = bigram.float()
        self.trigram = trigram.float()


    def predict(self, past_two_words):
        weights = torch.softmax(self.log_weights, dim=0)
        output_batch = torch.zeros(len(past_two_words), self.V)
        for i, pair in enumerate(past_two_words):
            bi = sparse_select(0, pair[-1], self.bigram)
            tri = sparse_select([0,1], pair.tolist(), self.trigram)
            uni = self.unigram
            output_batch[i, :] = (
                tri.to_dense() * weights[0]
                + bi.to_dense() * weights[1]
                + uni.to_dense() * weights[2])
        return output_batch

    def __call__(self, batch_text):
        packaged = torch.stack([batch_text[:-1,:], batch_text[1:, :]]).view(2, -1).t()
        return self.predict(packaged)

cross_entropy_loss = nn.CrossEntropyLoss()
def trigram_loss_fn(model, batch):
    pred = model(batch.text)
    labels = batch.target[1:,:].flatten()
    loss = cross_entropy_loss(pred, labels)
    return loss

