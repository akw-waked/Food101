from PIL import ImageEnhance, Image
import torchvision.transforms as transforms
import torch

# Image transformations and preprocessing

def apply_transformations(image, resize_width, resize_height, rotation_angle, brightness, contrast):
    image = image.resize((resize_width, resize_height))
    image = image.rotate(rotation_angle)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return image

def preprocess_image(image, device):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0).to(device)
