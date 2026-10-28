from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
import torch

def processing(train_dir: str = './dataset/train', test_dir: str='./dataset/test'):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    train_set = datasets.ImageFolder(root=train_dir, transform=transform)
    test = datasets.ImageFolder(root=test_dir, transform=transform)

    n_train = int(0.8 * len(train_set))
    n_dev = len(train_set) - n_train

    train, dev = random_split(
        train_set,
        [n_train, n_dev],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dev, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=True)

    print('Preprocessing: DONE')
    return train_loader, valid_loader, test_loader
