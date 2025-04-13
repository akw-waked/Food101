import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from utils.gradcam_utils import get_gradcam

# Config
NUM_CLASSES = 101
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build Model
def build_model(num_classes=NUM_CLASSES, pretrained=False, freeze=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if pretrained and freeze:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Load Model with checkpoint
def load_model(model_path):
    # Initialize model architecture
    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    # Load state dict directly
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Load class names
def load_classes(filepath='./classes.txt'):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Predict and get GradCAM
def predict_and_gradcam(model, input_tensor, image_size, classes):
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    predicted_class_idx = probabilities.argmax().item()

    activation_map_resized, _ = get_gradcam(
        model,
        input_tensor,
        predicted_class_idx,
        image_size=image_size
    )

    top_probs, top_idxs = torch.topk(probabilities, 3)
    predictions = [
        (classes[class_idx], prob.item())
        for prob, class_idx in zip(top_probs, top_idxs)
        if 0 <= class_idx < len(classes)
    ]

    return activation_map_resized, predictions
