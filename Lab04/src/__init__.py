from pathlib import Path

base = Path.cwd()
LAB = base/'Lab4' if (base/'Lab4').is_dir() else base
DATA = LAB/'data'
ART = LAB/'artifacts'
PAD, UNK, BOS, EOS = 0, 1, 2, 3

CONFIG = dict(
    emb=64, hidden=64, layers=1,  # Phần C: 256, 256, 3
    dropout=0.3, batch=32, lr=1e-3, teacher_ratio=0.5,
    epochs=3,                     # Phần C: 10
    max_words=25,                 # Phần C: 50
    vocab_limit=3000,             # Phần C: 10000
    min_freq=2,
    train_limit=512,              # Phần C: None (toàn bộ train)
    dev_limit=64,                 # Phần C: None (toàn bộ dev)
)

SPECIAL = ['<pad>', '<unk>', '<bos>', '<eos>']
