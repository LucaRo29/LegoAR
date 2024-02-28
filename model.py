import torchvision.models as models
import torch.nn as nn


def build_model(num_classes, pretrained=True, fine_tune=True):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        model = models.efficientnet_v2_m(weights='DEFAULT') 
    else:
        print('[INFO]: Not loading pre-trained weights')
        model = models.efficientnet_b5()

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # Change the final classification head.

    

    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes) #b5 2048 b3 1536
    #print(model)
    return model


