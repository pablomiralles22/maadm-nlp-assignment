from models.pretrained_transformer_model import PretrainedTransformerModel
from models.ensemble_model import EnsembleModel

def freeze_layers_transformer(transformer_model, num_unfrozen_layers):
    for param in transformer_model.parameters():
        param.requires_grad = False

    layers = transformer_model.encoder.layer
    frozen_layers = len(layers) - num_unfrozen_layers
    for layer in layers[frozen_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

def freeze_layers(model, num_unfrozen_layers):
    if isinstance(model, PretrainedTransformerModel) is True:
        freeze_layers_transformer(model.transformer_model, num_unfrozen_layers)
    elif isinstance(model, EnsembleModel) is True:
        for submodel in model.models:
            freeze_layers(submodel, num_unfrozen_layers)
