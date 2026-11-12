import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from LeNet import LeNet5
from engine import train_epochs, evaluation
from data_preprocessing import processing
from reporting import save_results, save_comparison

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def n_params(module, trainable_only=False):
    ps = module.parameters()
    if trainable_only:
        ps = (p for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in ps)


@torch.no_grad()
def layer_table(model, img, in_ch=3):
    parameter = next(model.parameters())
    x = torch.zeros(1, in_ch, img, img, device=parameter.device, dtype=parameter.dtype)
    rows, total = [], 0
    for name, layer in [("C1", model.c1), ("S2", model.s2),
                        ("C3", model.c3), ("S4", model.s4)]:
        x = layer(x)
        p = n_params(layer)
        total += p
        rows.append((name, "x".join(map(str, tuple(x.shape)[1:])), p))
    x = x.flatten(1)
    rows.append(("flatten", str(x.shape[1]), 0))
    for name, layer in [("F5", model.f5), ("F6", model.f6), ("OUT", model.out)]:
        x = layer(x)
        p = n_params(layer)
        total += p
        rows.append((name, str(x.shape[1]), p))

    print(f'Sum of Parameters: {total}\n')
    print('Table of Parameters')
    print(f"{'Layer':<10}{'Output':<16}{'Parameters':>12}")
    for name, shape, p in rows:
        print(f"{name:<10}{shape:<16}{p:>12,}")
    return rows, total

def main():
    parser = argparse.ArgumentParser()

    lab_dir = Path(__file__).resolve().parent.parent
    parser.add_argument('--train-dir', default=str(lab_dir / 'dataset/train'))
    parser.add_argument('--test-dir', default=str(lab_dir / 'dataset/test'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pool', choices=['avg', 'max'], default='avg')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'results'))

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs, batch-size and lr must be positive')
        
    print(f'Device: {device}')
    summaries = []

    train_loader, dev_loader, test_loader = processing(args.train_dir, args.test_dir, batch_size=args.batch_size, seed=args.seed)
    classes = train_loader.dataset.dataset.classes

    model = LeNet5(pool=args.pool, num_classes=len(classes)).to(device)
    print(f'\nLeNet-5 / {args.pool.upper()} pooling')
    rows, total = layer_table(model, 64)

    run_dir = Path(args.output_dir) / args.pool
    checkpoint_path = run_dir / 'best_model.pth'
    optimize = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    history = train_epochs(model=model, 
                           loader=train_loader, 
                           dev_loader=dev_loader, 
                           optimize=optimize, 
                           criterion=criterion, 
                           device=device,
                           epochs=args.epochs, 
                           checkpoint_path=checkpoint_path)

    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    scores = evaluation(model, test_loader, device, criterion=criterion, detailed=True)
    summary = {'pool': args.pool, 'parameters': total, 'best_epoch': checkpoint['epoch'],
                'best_val_f1': checkpoint['best_f1'],
                'seconds_per_epoch': sum(h['seconds'] for h in history) / len(history),
                **{k: scores[k] for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']}}
    save_results(run_dir, classes, scores, history, rows, summary, vars(args))
    summaries.append(summary)
    print(f'Test (best epoch {checkpoint["epoch"]}): {summary}')
    save_comparison(Path(args.output_dir), summaries)
    print(f'\nResults: {Path(args.output_dir).resolve()}')

if __name__ == '__main__':
    main()
