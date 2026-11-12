import torch
from pathlib import Path
from time import perf_counter
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def train_epochs(model, loader, dev_loader, optimize, criterion, device, epochs: int=10, checkpoint_path=None):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).resolve().parent / "checkpoints/best_model.pth"

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []

    for epoch in range(epochs):
        start = perf_counter()
        print(f'\nEpoch: {epoch + 1}')
        model.train()
        running_loss, total = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimize.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimize.step()
            running_loss += loss.item() * yb.size(0)
            total += yb.size(0)

        scores = evaluation(model, dev_loader, device, criterion=criterion, detailed=True)
        dev_f1 = scores['f1_macro']
        dev_accuracy = scores['accuracy']
        dev_precision = scores['precision_macro']
        dev_recall = scores['recall_macro']
        history.append({
            'epoch': epoch + 1, 'train_loss': running_loss / total,
            'val_loss': scores['loss'], 'val_f1_macro': dev_f1,
            'val_accuracy': dev_accuracy, 'seconds': perf_counter() - start,
        })
        print(f'f1: {dev_f1} | precision: {dev_precision} | recall: {dev_recall} | accuracy: {dev_accuracy}')

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimize.state_dict(),
                'best_f1': best_f1,
            }, checkpoint_path)

            print(f"Saved checkpoint: {checkpoint_path}")
    return history


@torch.no_grad()
def evaluation(model, loader, device, criterion=None, detailed=False):
    model.eval()
    preds, truth = [], []
    running_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        if criterion is not None:
            running_loss += criterion(logits, yb.to(device)).item() * yb.size(0)
        y_preds = torch.argmax(logits, dim=1)

        preds += y_preds.cpu().tolist()
        truth += yb.cpu().tolist()

    if not truth:
        raise ValueError("Cannot evaluate an empty dataset")
    labels = list(range(model.out.out_features))
    f1 = f1_score(truth, preds, labels=labels, average='macro', zero_division=0)
    accuracy = accuracy_score(truth, preds)
    precision = precision_score(truth, preds, labels=labels, average="macro", zero_division=0)
    recall = recall_score(truth, preds, labels=labels, average="macro", zero_division=0)
    if detailed:
        return {'f1_macro': f1, 'accuracy': accuracy, 'precision_macro': precision,
                'recall_macro': recall, 'loss': running_loss / len(truth) if criterion is not None else None,
                'y_true': truth, 'y_pred': preds}
    return f1, accuracy, precision, recall
