import unicodedata, json
import torch
import numpy as np
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from pathlib import Path
from collections import Counter
from __init__ import SPECIAL, UNK, EOS, BOS, PAD

def tokens(text):
    text = unicodedata.normalize('NFC', text.replace('﻿', ''))
    return text.lower().split()

def preprocessing(train_dir: str = 'data/small-train.json', 
                  dev_dir: str = 'data/small-dev.json', 
                  test_dir: str = 'data/small-test.json',
                  max_word: int = 50):
    raw, filtered = {}, {}
    path = {
        'train': Path(train_dir),
        'dev': Path(dev_dir),
        'test': Path(test_dir)
    }

    for split in ['train', 'dev', 'test']:
        raw[split] = json.loads(path[split].read_text(encoding='utf-8-sig'))
        
        filtered[split] = []
        for i, row in enumerate(raw[split]):
            source, target = tokens(row['english']), tokens(row['vietnamese'])
            if 1 <= len(source) <= max_word and 1 <= len(target) <= max_word:
                filtered[split].append(dict(id=i, src=source, tgt=target))

    return raw, filtered

def make_vocab(sequences, min_freq: int=2, max_vocab: int=3000):
    freq = Counter(t for seq in sequences for t in seq)
    ranked = sorted(freq, key=lambda t: (-freq[t], t))
    keep = [t for t in ranked if freq[t] >= min_freq
            and t not in SPECIAL][:max_vocab-4]
    return {token: i for i, token in enumerate(SPECIAL + keep)}

def encode(seq, vocab):
    return [vocab.get(t, UNK) for t in seq]

def collate(rows, src_vocab, tgt_vocab):
    src = [torch.tensor(encode(r['src'], src_vocab)+[EOS]) for r in rows]
    tgt = [torch.tensor([BOS]+encode(r['tgt'], tgt_vocab)+[EOS])
           for r in rows]
    return dict(
        src=pad_sequence(src, batch_first=True, padding_value=PAD),
        lengths=torch.tensor([len(x) for x in src]),
        tgt=pad_sequence(tgt, batch_first=True, padding_value=PAD),
        refs=[' '.join(r['tgt']) for r in rows],
        ids=[r['id'] for r in rows],
    )

def choose(rows, limit: int = None):
    if limit is None or len(rows) <= limit:
        return rows
    ids = np.random.default_rng(42).permutation(len(rows))[:limit]
    return [rows[int(i)] for i in ids]

def loader(rows, src_vocab, tgt_vocab, seed: int=42, shuffle=False, batch_size: int=32):
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=partial(collate,
                                        src_vocab=src_vocab,
                                        tgt_vocab=tgt_vocab),
                      generator=torch.Generator().manual_seed(seed))
