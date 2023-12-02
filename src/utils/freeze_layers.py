from models.pretrained_transformer_model import PretrainedTransformerModel
from models.ensemble_model import EnsembleModel

def freeze_layers_transformer(transformer_model, num_unfrozen_layers):
    """
    Freezes the specified number of layers in a transformer model.

    Args:
        transformer_model (nn.Module): The transformer model to freeze layers in.
        num_unfrozen_layers (int): The number of layers to keep unfrozen.

    Returns:
        None
    """
    # freeze every parameter first
    for param in transformer_model.parameters():
        param.requires_grad = False

    layers = transformer_model.encoder.layer
    frozen_layers = len(layers) - num_unfrozen_layers
    # for each layer in the last num_unfrozen_layers, unfreeze the parameters
    for layer in layers[frozen_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

def freeze_layers(model, num_unfrozen_layers):
    """
    Freezes a specified number of layers in every transformer submodel of the
    given model.

    Args:
        model (object): The model to freeze layers in.
        num_unfrozen_layers (int): The number of layers to leave unfrozen.

    Returns:
        None
    """
    if isinstance(model, PretrainedTransformerModel) is True:
        freeze_layers_transformer(model.transformer_model, num_unfrozen_layers)
    elif isinstance(model, EnsembleModel) is True:
        for submodel in model.models:
            freeze_layers(submodel, num_unfrozen_layers)
