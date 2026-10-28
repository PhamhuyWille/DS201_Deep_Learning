import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from data_preprocessing import processing
from engine import train_epochs, evaluation
from GoogleNet import GoogleNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-dir', default='../dataset/train')
    parser.add_argument('--test-dir', default='../dataset/test')
    parser.add_argument('--checkpoint-dir', default='checkpoint')
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    train_loader, valid_loader, test_loader = processing(args.train_dir, args.test_dir)
    model = GoogleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    checkpoint_path = Path(args.checkpoint_dir) / "best_model.pth"
    train_epochs(
        model=model,
        loader=train_loader,
        dev_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        checkpoint_path=checkpoint_path
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    f1, precision, recall, accuracy = evaluation(model, test_loader, device)
    print('\nTest Score:')
    print(f'F1 Score: {f1}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    main()
