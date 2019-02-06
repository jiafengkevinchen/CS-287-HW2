import torch
from torch import Tensor
from collections import Counter
from torch import sparse as sp
from tqdm import tqdm_notebook as tqdm

class Trigram:
    def __init__(self, TEXT):
        self.weights = torch.zeros(3, requires_grad=True)
        self.TEXT = TEXT

    def get_probabilities(self, train_iter):
        TEXT = self.TEXT
        V = len(TEXT.vocab)
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
        bigram = bigram / sp.sum(bigram, dim=-1)

        trigram = trigram.coalesce()
        trigram = trigram / sp.sum(bigram, dim=-1)

        self.unigram = unigram
        self.bigram = bigram
        self.trigram = trigram

    def __call__(batch_text):

