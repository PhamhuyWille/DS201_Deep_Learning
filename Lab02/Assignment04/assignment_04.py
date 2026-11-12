"""DS201 Lab 2 - Bài 4: pretrained Hugging Face ResNet-50 on VinaFood21."""
import argparse
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pretrained_resnet import PretrainedResnet
from engine import train_epochs, evaluation
from data_preprocessing import processing
from reporting import save_results, save_comparison, save_four_model_table


def n_params(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if not trainable_only or p.requires_grad)


def layer_table(model):
    rows = [('Backbone (ResNet-50)', '2048x1x1', n_params(model.resnet)),
            ('Flatten', '2048', 0),
            ('Classifier', str(model.classifier.out_features), n_params(model.classifier))]
    print(f"{'Layer':<24}{'Output':<16}{'Parameters':>12}")
    for name, shape, params in rows:
        print(f'{name:<24}{shape:<16}{params:>12,}')
    total = sum(r[2] for r in rows)
    print(f'Sum of Parameters: {total:,} | Trainable: {n_params(model, True):,}')
    return rows, total


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    lab_dir = Path(__file__).resolve().parent.parent
    parser.add_argument('--train-dir', default=str(lab_dir/'dataset/train'))
    parser.add_argument('--test-dir', default=str(lab_dir/'dataset/test'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=['both', 'feature_extraction', 'fine_tuning'], default='both')
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent/'results'))
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error('epochs/batch-size must be positive and num-workers nonnegative')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    seed_everything(args.seed)
    train_loader, dev_loader, test_loader = processing(
        args.train_dir, args.test_dir, batch_size=args.batch_size,
        seed=args.seed, img_size=224)
    
    classes = train_loader.dataset.dataset.classes
    modes = ['feature_extraction', 'fine_tuning'] if args.mode == 'both' else [args.mode]
    summaries = []

    for mode in modes:
        seed_everything(args.seed)
        freeze = mode == 'feature_extraction'
        lr = 1e-3 if freeze else 1e-4

        model = PretrainedResnet(num_classes=len(classes), freeze_backbone=freeze).to(device)
        print(f'\nResNet-50 / {mode} / lr={lr}')
        layers, total = layer_table(model)
        run_dir = Path(args.output_dir) / mode
        checkpoint_path = run_dir/'best_model.pth'
        optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
        criterion = nn.CrossEntropyLoss()
        history = train_epochs(model, train_loader, dev_loader, optimizer, criterion,
                               device, epochs=args.epochs, checkpoint_path=checkpoint_path)
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        scores = evaluation(model, test_loader, device, criterion=criterion, detailed=True)

        summary = {'mode': mode, 'parameters': total,
                   'trainable_parameters': n_params(model, True), 'lr': lr,
                   'best_epoch': checkpoint['epoch'], 'best_val_f1': checkpoint['best_f1'],
                   'seconds_per_epoch': sum(h['seconds'] for h in history)/len(history),
                   **{k: scores[k] for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']}}
        save_results(run_dir, classes, scores, history, layers, summary, vars(args))
        summaries.append(summary)
        print(f'Test (best epoch {checkpoint["epoch"]}): {summary}')
        del model, optimizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    save_comparison(Path(args.output_dir), summaries)
    save_four_model_table(Path(args.output_dir), lab_dir, summaries)
    print(f'\nResults: {Path(args.output_dir).resolve()}')


if __name__ == '__main__':
    main()
