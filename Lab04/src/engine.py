import numpy as np
import torch 
import torch.nn as nn
import time
from pathlib import Path
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from __init__ import PAD, BOS, EOS
from preprocessing import tokens

# Huấn luyện một epoch, trả loss trung bình trên token thật
def train_epoch(model, train_loader, tgt_vocab, optimizer, device, criterion, teacher_ratio):
    model.train()
    loss_total, tokens_total = 0.0, 0
    for b in train_loader:
        src, tgt = b['src'].to(device), b['tgt'].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(src, b['lengths'], tgt, teacher_ratio=teacher_ratio)
        gold = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, len(tgt_vocab)),
                         gold.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n = int((gold != PAD).sum())
        loss_total += loss.item()*n
        tokens_total += n
    return loss_total/tokens_total

def fit(model, train_loader, dev_loader, tgt_vocab, name, device, max_words, lr=1e-5, teacher_ratio: int=0.5, epoch: int=10, checkpoint: str=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    best_key, best_state, best_epoch = (-1., float('-inf')), None, 0
    history = []
    for epoch in range(1, epoch + 1):
        start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, tgt_vocab, optimizer, device, criterion, teacher_ratio)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        measured = evaluate(model, dev_loader, tgt_vocab, device, max_words)
        key = (measured['rougeL_f1'], -measured['dev_nll'])
        if key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        history.append(dict(epoch=epoch, train_loss=train_loss,
                            seconds=seconds, **measured))
        print(f"{name} epoch {epoch}: loss={train_loss:.4f} "
              f"dev_ROUGE-L={measured['rougeL_f1']:.4f} "
              f"BLEU={measured['bleu']:.2f}")
    checkpoint = Path(checkpoint)
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint/f'{name}.pt')
    return history, best_epoch

@torch.no_grad()
def generate(model, src, lengths, max_steps):
    model.eval()
    memory, state = model.encoder(src, lengths)
    mask = src != PAD
    token = torch.full((src.size(0),), BOS, device=src.device)
    feeding = memory.new_zeros(src.size(0), model.hidden)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    outputs, all_weights = [], []
    for _ in range(max_steps):
        logits, state, feeding, weights = model.decoder.step(
            token, state, memory, mask, feeding)
        logits[:, [PAD, BOS]] = float('-inf')
        token = logits.argmax(-1)
        token = torch.where(finished, torch.full_like(token, EOS), token)
        outputs.append(token)
        if weights is not None:
            all_weights.append(weights)
        finished = finished | (token == EOS)
        if bool(finished.all()):
            break
    ids = torch.stack(outputs, dim=1)
    weights = torch.stack(all_weights, dim=1) if all_weights else None
    return ids, weights

def decode(ids, tgt_itos):
    words = []
    for value in ids:
        i = int(value)
        if i == EOS:
            break
        if i not in {PAD, BOS}:
            words.append(tgt_itos[i])
    return ' '.join(words)

class UnicodeTokens:
    def tokenize(self, text):
        return tokens(text)

def translation_metrics(hypotheses, references):
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False,
                                 tokenizer=UnicodeTokens())
    scores = [rouge.score(ref, hyp)['rougeL']
              for hyp, ref in zip(hypotheses, references)]
    
    bleu = BLEU(tokenize='none', lowercase=False)
    value = bleu.corpus_score(hypotheses, [references])
    return dict(
        rougeL_f1=float(np.mean([s.fmeasure for s in scores])),
        rougeL_precision=float(np.mean([s.precision for s in scores])),
        rougeL_recall=float(np.mean([s.recall for s in scores])),
        bleu=value.score, bleu_signature=str(bleu.get_signature()))

@torch.no_grad()
def evaluate(model, loader, tgt_vocab, device, max_words: int=50):
    model.eval()
    hypotheses, references = [], []
    nll, count = 0.0, 0
    tgt_itos = {i: t for t, i in tgt_vocab.items()}
    for b in loader:
        src, tgt = b['src'].to(device), b['tgt'].to(device)
        target = tgt[:, 1:]
        logits = model(src, b['lengths'], tgt, teacher_ratio=1.0)
        nll += nn.functional.cross_entropy(
            logits.reshape(-1, len(tgt_vocab)), target.reshape(-1),
            ignore_index=PAD, reduction='sum').item()
        count += int((target != PAD).sum())
        generated, _ = generate(model, src, b['lengths'], max_words + 1)
        hypotheses.extend(decode(row, tgt_itos) for row in generated.cpu())
        references.extend(b['refs'])

    metrics = translation_metrics(hypotheses, references)
    metrics['dev_nll'] = nll/count
    return metrics
