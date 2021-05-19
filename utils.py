import torch
from torchvision import models
from torch import nn

MODELS = {
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
}


def build_model(arch, hidden_units, dropout=0.2):
    model = MODELS[arch](pretrained=True)
    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # change image classifier
    first_layer = [
        nn.Linear(25088, hidden_units[0]),
        nn.ReLU(),
        nn.Dropout(dropout)
    ]
    last_layer = [
        nn.Linear(hidden_units[-1], 102),
        nn.LogSoftmax(dim=1)
    ]
    hidden_layers = []
    inner_units = zip(hidden_units[:-1], hidden_units[1:])
    for h1, h2 in inner_units:
        hidden_layers.extend([
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])

    model.classifier = nn.Sequential(
        *first_layer, *hidden_layers, *last_layer
    )
    return model


def determine_device(gpu):
    if gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise Exception("GPU not available on this machine")
    else:
        return torch.device("cpu")
