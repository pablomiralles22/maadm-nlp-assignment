def freeze_layers(transformer_model, num_unfrozen_layers):
    for param in transformer_model.parameters():
        param.requires_grad = False

    layers = transformer_model.encoder.layer
    frozen_layers = len(layers) - num_unfrozen_layers
    for layer in layers[frozen_layers:]:
        for param in layer.parameters():
            param.requires_grad = True