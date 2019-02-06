from tqdm import tqdm_notebook as tqdm

def train_model(
    model, loss_fn=None, optimizer=None, train_iter=None,
    val_iter=None, num_epochs=5, writer=None, callback=None, inner_callback=None):
    """
    TODO
    """
    if hasattr(model, 'train'):
        model.train()
    elif loss_fn is None or optimizer is None:
        raise ValueError
    else:
        for epoch in range(num_epochs):
            train_loss = 0
            total = 0
            for batch in train_iter:
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
                for batch in train_iter:
                    loss = loss_fn(model, batch)
                    val_loss += loss.item()
                if writer is not None:
                    writer.add_scalar('validation_loss', val_loss / total, epoch)
            if callback is not None:
                callback(**locals())



def tensor_to_text(t, TEXT):
    return ' '.join([TEXT.vocab.itos[i] for i in t])





