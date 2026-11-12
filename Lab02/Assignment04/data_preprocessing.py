"""Fixed 80/20 split; ImageNet-normalized ResNet-50 input at 224x224."""
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Subset
from transformers import AutoImageProcessor
import torch


def processing(train_dir, test_dir, batch_size=32, seed=42, img_size=224):
    processor = AutoImageProcessor.from_pretrained('microsoft/resnet-50')
    mean, std = processor.image_mean, processor.image_std
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    base_train = datasets.ImageFolder(train_dir, transform=transform)
    test = datasets.ImageFolder(test_dir, transform=transform)

    n_train = int(0.8 * len(base_train))
    n_dev = len(base_train) - n_train

    train, dev = random_split(base_train, [n_train, n_dev], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(dev, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test, shuffle=False, batch_size=batch_size)
    print(f'Preprocessing: DONE | train={len(train)}, val={len(dev)}, test={len(test)}')
    return train_loader, valid_loader, test_loader
