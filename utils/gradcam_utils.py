# gradcam_utils
import numpy as np
import torch.nn.functional as F

def normalize_activation_map(activation_map):
    activation_map -= activation_map.min()
    activation_map /= activation_map.max() + 1e-8  # Avoid division by zero
    return activation_map

def get_gradcam(model, input_tensor, predicted_class_idx, image_size):
    from torchcam.methods import GradCAM
    import torch.nn.functional as F

    cam_extractor = GradCAM(model, target_layer='layer4')
    outputs = model(input_tensor)

    # Get activation map list and pick first tensor
    activation_map = cam_extractor(predicted_class_idx, outputs)[0]  # shape: [1, H, W]

    # Check shape
    if activation_map.dim() == 3:  # shape: [N, H, W]
        activation_map = activation_map.unsqueeze(1)  # add channel dim: [N, 1, H, W]

    # Resize to match image size
    activation_map_resized = F.interpolate(
        activation_map,
        size=(image_size[0], image_size[1]),  # (height, width)
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()

    # Normalize activation map to [0, 1]
    activation_map_resized -= activation_map_resized.min()
    activation_map_resized /= activation_map_resized.max() + 1e-8

    return activation_map_resized, outputs
