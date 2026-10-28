import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from LeNet import LeNet5
from engine import train_epochs, evaluation
from data_preprocessing import processing

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-dir', default='dataset/train')
    parser.add_argument('--test-dir', default='dataset/test')
    parser.add_argument('--epochs', default=3)

    args = parser.parse_args()

    train_loader, dev_loader, test_loader = processing(args.train_dir, args.test_dir)
    model = LeNet5().to(device)
    optimize = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    train_epochs(model, loader=train_loader, dev_loader=dev_loader, optimize=optimize, criterion=criterion, epochs=args.epochs, device=device)
    f1 = evaluation(model, test_loader, device)
    print(f1)

if __name__ == '__main__':
    main()