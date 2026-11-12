import torch
from pathlib import Path
from time import perf_counter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def train_epochs(model, loader, dev_loader, criterion, optimizer, device, epochs: int=10, checkpoint_path="checkpoints/best_model.pth"):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []
    
    for epoch in range(epochs):
        start = perf_counter()
        print(f'Epoch {epoch + 1}:')
        model.train()
        running_loss, total = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            total += yb.size(0)

        scores = evaluation(model, dev_loader, device, criterion=criterion, detailed=True)
        f1, precision, recall, accuracy = (scores[k] for k in
            ['f1_macro', 'precision_macro', 'recall_macro', 'accuracy'])
        history.append({'epoch': epoch + 1, 'train_loss': running_loss / total,
                        'val_loss': scores['loss'], 'val_f1_macro': f1,
                        'val_accuracy': accuracy, 'seconds': perf_counter() - start})
        print(f'f1: {f1} | precision: {precision} | recall: {recall} | accuracy: {accuracy}')

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, checkpoint_path)

            print(f"Saved checkpoint: {checkpoint_path}")
    return history

@torch.no_grad()
def evaluation(model, test, device, criterion=None, detailed=False):
    model.eval()

    truth, preds = [], []
    running_loss = 0.0
    for xb, yb in test:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if criterion is not None:
            running_loss += criterion(logits, yb).item() * yb.size(0)
        y_pred = torch.argmax(logits, dim=1)

        preds += y_pred.cpu().tolist()
        truth += yb.cpu().tolist()

    if not truth:
        raise ValueError('Cannot evaluate an empty dataset')
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
