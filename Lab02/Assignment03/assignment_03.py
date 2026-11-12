import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ResNet import ResNet18
from engine import train_epochs, evaluation
from preprocessing import processing
from reporting import save_results, save_comparison


def n_params(module):
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def layer_table(model, img=96):
    model.eval()
    parameter = next(model.parameters())
    x = torch.zeros(1, 3, img, img, device=parameter.device)
    rows = []
    for name in ('stem', 'layer1', 'layer2', 'layer3', 'layer4'):
        layer = getattr(model, name)
        x = layer(x)
        rows.append((name, 'x'.join(map(str, x.shape[1:])), n_params(layer)))
    x = model.pool(x).flatten(1)
    rows.append(('avgpool', str(x.shape[1]), n_params(model.pool)))
    x = model.fc(x)
    rows.append(('fc', str(x.shape[1]), n_params(model.fc)))
    total = sum(r[2] for r in rows)
    print(f'\nSum of Parameters: {total:,}')
    print(f'{"Layer":<12}{"Output":<18}{"Parameters":>12}')
    for name, shape, p in rows:
        print(f'{name:<12}{shape:<18}{p:>12,}')
    return rows, total


def divergence_epoch(history):
    for prev, curr in zip(history, history[1:]):
        if curr['val_loss'] > prev['val_loss'] and curr['train_loss'] < prev['train_loss']:
            return curr['epoch']
    return None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    lab_dir = Path(__file__).resolve().parent.parent
    parser.add_argument('--train-dir', default=str(lab_dir / 'dataset/train'))
    parser.add_argument('--test-dir', default=str(lab_dir / 'dataset/test'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--img-size', type=int, choices=[96, 112], default=96)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'results'))
    parser.add_argument('--mode', choices=['no_aug', 'aug'], default='aug')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.num_workers < 0:
        parser.error('epochs, batch-size, lr must be positive; num-workers nonnegative')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    summaries = []
    
    set_seed(args.seed)
    augment = args.mode == 'aug'
    print(f'\nResNet-18 | {args.mode} | input {args.img_size}x{args.img_size}')
    train_loader, dev_loader, test_loader = processing(
        args.train_dir, args.test_dir, seed=args.seed,
        batch_size=args.batch_size, img_size=args.img_size,
        augment=augment, num_workers=args.num_workers)
    classes = train_loader.dataset.dataset.classes
    model = ResNet18(num_classes=len(classes)).to(device)
    rows, total = layer_table(model, args.img_size)
    if len(classes) == 21:
        assert total == 11_187_285, f'Expected 11,187,285 parameters, got {total}'
    run_dir = Path(args.output_dir) / args.mode
    checkpoint_path = run_dir / 'best_model.pth'
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    history = train_epochs(model=model, loader=train_loader,
                            dev_loader=dev_loader, optimize=optimizer,
                            criterion=criterion, device=device, epochs=args.epochs,
                            checkpoint_path=checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    scores = evaluation(model, test_loader, device, criterion=criterion, detailed=True)
    summary = {
        'augmentation': args.mode,
        'parameters': total,
        'best_epoch': checkpoint['epoch'],
        'best_val_f1': checkpoint['best_f1'],
        'seconds_per_epoch': sum(h['seconds'] for h in history) / len(history),
        'loss_divergence_epoch': divergence_epoch(history),
        **{k: scores[k] for k in ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro')},
    }
    save_results(run_dir, classes, scores, history, rows, summary,
                    vars(args), title=f'ResNet-18 {args.mode}')
    summaries.append(summary)
    print('Test:', summary)
    save_comparison(Path(args.output_dir), summaries)
    print(f'\nResults: {Path(args.output_dir).resolve()}')


if __name__ == '__main__':
    main()
