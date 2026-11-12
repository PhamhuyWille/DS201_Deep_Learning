from pathlib import Path
from time import perf_counter
import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def train_epochs(model, loader, dev_loader, optimize, criterion, device, epochs=10, checkpoint_path=None):
    checkpoint_path = Path(checkpoint_path or Path(__file__).parent / 'checkpoints/best_model.pth')
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1, history = -1., []
    for epoch in range(epochs):
        start = perf_counter()
        print(f'\nEpoch: {epoch + 1}/{epochs}', flush=True)
        model.train()
        running_loss, total = 0., 0

        for xb, yb in tqdm(loader, desc=f'Train {epoch + 1}', leave=False):
            xb = xb.to(device)
            yb = yb.to(device)

            optimize.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimize.step()
            running_loss += loss.item() * yb.size(0)
            total += yb.size(0)

        scores = evaluation(model, dev_loader, device, criterion=criterion, detailed=True)
        row = {'epoch': epoch + 1, 'train_loss': running_loss / total,
               'val_loss': scores['loss'], 'val_f1_macro': scores['f1_macro'],
               'val_accuracy': scores['accuracy'], 'seconds': perf_counter() - start}
        history.append(row)
        print(f"Train loss: {row['train_loss']:.4f} | Val loss: {row['val_loss']:.4f} | "
              f"F1: {scores['f1_macro']:.4f} | Precision: {scores['precision_macro']:.4f} | "
              f"Recall: {scores['recall_macro']:.4f} | Accuracy: {scores['accuracy']:.4f}")
        if scores['f1_macro'] > best_f1:
            best_f1 = scores['f1_macro']
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimize.state_dict(), 'best_f1': best_f1}, checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')
    return history


@torch.no_grad()
def evaluation(model, loader, device, criterion=None, detailed=False):
    model.eval()
    preds, truth, running_loss = [], [], 0.
    for xb, yb in tqdm(loader, desc='Eval', leave=False):
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        if criterion is not None:
            running_loss += criterion(logits, yb.to(device, non_blocking=True)).item() * yb.size(0)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        truth.extend(yb.tolist())
    if not truth: raise ValueError('Cannot evaluate empty dataset')
    labels = list(range(model.classifier.out_features))
    f1 = f1_score(truth, preds, labels=labels, average="macro", zero_division=0)
    precision = precision_score(truth, preds, labels=labels, average="macro", zero_division=0)
    recall = recall_score(truth, preds, labels=labels, average="macro", zero_division=0)
    accuracy = accuracy_score(truth, preds)
    if detailed:
        return {'f1_macro': f1, 'precision_macro': precision, 'recall_macro': recall,
                'accuracy': accuracy, 'loss': running_loss / len(truth) if criterion is not None else None,
                'y_true': truth, 'y_pred': preds}
    return f1, precision, recall, accuracy
