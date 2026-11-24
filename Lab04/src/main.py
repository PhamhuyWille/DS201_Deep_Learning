import torch
import numpy as np
import pandas as pd
import random
import argparse
from pathlib import Path

from preprocessing import preprocessing, make_vocab, choose, loader
from engine import fit, evaluate
from model import Seq2Seq
from plotting import plot_learning_curves

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    lab_dir = Path(__file__).resolve().parent.parent
    parser.add_argument('--train-dir', default=str(lab_dir / 'data/small-train.json'))
    parser.add_argument('--dev-dir', default=str(lab_dir / 'data/small-dev.json'))
    parser.add_argument('--test-dir', default=str(lab_dir / 'data/small-test.json'))
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-words', type=int, default=50)
    parser.add_argument('--max-vocab', type=int, default=10000)
    parser.add_argument('--min-freq', type=int, default=2)

    parser.add_argument('--num-layer', type=int, default=3)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--embedding-size', type=int, default=256)

    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'results'))
    parser.add_argument('--checkpoint', default=str(Path(__file__).resolve().parent / 'checkpoint'))
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('PyTorch:', torch.__version__, '| device:', device)

    _, filtered = preprocessing(train_dir=args.train_dir,
                                dev_dir=args.dev_dir,
                                test_dir=args.test_dir,
                                max_word=args.max_words)

    src_vocab = make_vocab((r['src'] for r in filtered['train']),
                           min_freq=args.min_freq, max_vocab=args.max_vocab)
    
    tgt_vocab = make_vocab((r['tgt'] for r in filtered['train']),
                           min_freq=args.min_freq, max_vocab=args.max_vocab)

    train_rows = choose(filtered['train'])
    dev_rows = choose(filtered['dev'])
    test_rows = choose(filtered['test'])
    histories = {}
    
    for mode in ['none', 'luong', 'bahdanau']:
        seed_all(args.seed)

        train_loader = loader(train_rows, src_vocab, tgt_vocab, seed=args.seed, shuffle=True, batch_size=args.batch_size)
        dev_loader = loader(dev_rows, src_vocab, tgt_vocab, batch_size=args.batch_size)
        test_loader = loader(test_rows, src_vocab, tgt_vocab, batch_size=args.batch_size)
    
        model = Seq2Seq(mode=mode, 
                        src_size=len(src_vocab),
                        tgt_size=len(tgt_vocab),
                        embedding_dim=args.embedding_size,
                        hidden_size=args.hidden_size,
                        layers=args.num_layer,
                        dropout=args.dropout,
                        ).to(device)
            
        history, best_epoch = fit(model, 
            train_loader=train_loader,
            dev_loader=dev_loader,
            tgt_vocab=tgt_vocab,
            name=mode,
            max_words=args.max_words,
            lr=args.lr,
            epoch=args.epochs, 
            device=device,
            checkpoint=args.checkpoint)

        histories[mode] = history
        plot_path = plot_learning_curves(histories, args.output_dir)
        print(f'Learning curves saved to: {plot_path}')
        print(f'{mode} best epoch: {best_epoch}')

        checkpoint_path = Path(args.checkpoint) / f'{mode}.pt'
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()

        metrics = evaluate(model, 
                           loader=test_loader, 
                           tgt_vocab=tgt_vocab, 
                           device=device, 
                           max_words=args.max_words)

        print('*' * 50)
        print(f'Mode: {mode}')
        print(f'Rougel_f1: {metrics['rougeL_f1']}')
        print(f'RougeL_precision: {metrics['rougeL_precision']}')
        print(f'RougeL_recall: {metrics['rougeL_recall']}')
        print(f'Bleu: {metrics['bleu']}')
        print(f'Bleu_signature: {metrics['bleu_signature']}\n')
        
if __name__ == '__main__':
    main()
