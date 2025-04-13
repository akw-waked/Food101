import torch
from torchvision import datasets, transforms

# Define transforms
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define dataset and loader
test_data_dir = './data/test' 

test_dataset = datasets.ImageFolder(root=test_data_dir, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Get class names
classes = test_dataset.classes
