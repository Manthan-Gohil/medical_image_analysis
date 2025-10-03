import torch.nn as nn
import torchvision.models as models

def build_model(pretrained=True):
    """
    Builds a Vision Transformer (ViT-B/16) model and prepares it for fine-tuning.
    This version uses the legacy 'pretrained=True' flag for older torchvision versions.
    """
    print('[INFO] Building Vision Transformer (ViT) model for full fine-tuning...')

    # --- KEY CHANGE FOR COMPATIBILITY ---
    # We are changing from the new 'weights' enum to the older 'pretrained' boolean flag.
    # This ensures the code works on older versions of torchvision.
    model = models.vit_b_16(pretrained=pretrained)

    # Unfreeze all layers of the model to allow them to be trained
    for params in model.parameters():
        params.requires_grad = True

    # Replace the final classification "head"
    num_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)  # 2 output classes: Normal vs. Fracture
    )
    return model