from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Subset
import torch

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def processing(train_dir='./dataset/train', test_dir='./dataset/test', seed=42,
               batch_size=32, img_size=96, augment=False, num_workers=2):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Tách cùng một tập index cho cả hai thí nghiệm.
    base_train = datasets.ImageFolder(root=train_dir)
    n_train = int(0.8 * len(base_train))
    n_dev = len(base_train) - n_train
    if n_train == 0 or n_dev == 0:
        raise ValueError('Train dataset must contain at least two images')
    tr_split, dev_split = random_split(
        base_train, [n_train, n_dev],
        generator=torch.Generator().manual_seed(seed)
    )
    train_set = datasets.ImageFolder(root=train_dir, transform=train_tf if augment else eval_tf)
    dev_set = datasets.ImageFolder(root=train_dir, transform=eval_tf)
    test_set = datasets.ImageFolder(root=test_dir, transform=eval_tf)
    if train_set.class_to_idx != test_set.class_to_idx:
        raise ValueError('Train/test class_to_idx differ: check class folders')

    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(Subset(train_set, tr_split.indices), shuffle=True, **kwargs)
    valid_loader = DataLoader(Subset(dev_set, dev_split.indices), shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **kwargs)
    print(f'Preprocessing: DONE | augmentation={augment} | '
          f'train={n_train}, val={n_dev}, test={len(test_set)}')
    return train_loader, valid_loader, test_loader
