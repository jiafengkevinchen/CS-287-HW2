from namedtensor import ntorch

class Ensemble:
    def __init__(self, *models):
        self.models = models

    def __call__(self, batch_text):
        return ntorch.stack(
            [model(batch_text) for model in self.models], "model").mean("model")

