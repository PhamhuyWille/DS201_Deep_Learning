import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_preprocessing import processing
from engine import train_epochs, evaluation
from GoogleNet import GoogleNet, layer_table
from Assignment01.reporting import save_results, write_csv
from comparison import save_comparison

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser()

    lab_dir = Path(__file__).resolve().parent.parent
    parser.add_argument('--train-dir', default=str(lab_dir / 'dataset/train'))
    parser.add_argument('--test-dir', default=str(lab_dir / 'dataset/test'))
    parser.add_argument('--checkpoint-dir', default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'results'))
    parser.add_argument('--lenet-results-dir', default=str(lab_dir / 'Assignment01/results'))

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs, batch-size and lr must be positive')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f'Device: {device}')
    train_loader, valid_loader, test_loader = processing(
        args.train_dir, args.test_dir, seed=args.seed, batch_size=args.batch_size)
    classes = train_loader.dataset.dataset.classes
    if len(classes) != 21 or classes != test_loader.dataset.classes:
        raise ValueError('Train/test must contain the same 21 class folders')
    model = GoogleNet(num_classes=len(classes)).to(device)
    layers, channels, total = layer_table(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint_dir or output_dir) / "best_model.pth"
    history = train_epochs(
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

    scores = evaluation(model, test_loader, device, criterion=criterion, detailed=True)
    summary = {'model': 'GoogLeNet (3 Inception blocks)', 'parameters': total,
               'best_epoch': checkpoint['epoch'], 'best_val_f1': checkpoint['best_f1'],
               'seconds_per_epoch': sum(h['seconds'] for h in history) / len(history),
               **{k: scores[k] for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']}}
    save_results(output_dir, classes, scores, history, layers, summary, vars(args),
                 title='GoogLeNet (3 Inception blocks)')
    write_csv(output_dir / 'inception_channels.csv', channels, list(channels[0]))
    save_comparison(output_dir, Path(args.lenet_results_dir), summary, vars(args))
    print(f'\nTest (best epoch {checkpoint["epoch"]}): {summary}')
    print(f'Results: {output_dir.resolve()}')

if __name__ == '__main__':
    main()
