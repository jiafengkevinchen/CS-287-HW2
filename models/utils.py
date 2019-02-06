def train_model(
    model, criterion=None, optimizer=None, ):
    if hasattr(model, 'train'):
        model.train()
    elif criterion is None or optimizer is None:
        raise ValueError

