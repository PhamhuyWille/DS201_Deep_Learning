import torch
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def train_epochs(model, loader, dev_loader, optimize, criterion, device, epochs: int=10, checkpoint_path="checkpoints/best_model.pth"):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimize.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimize.step()

        dev_f1, dev_accuracy, dev_precision, dev_recall = evaluation(model, dev_loader, device)
        print(f'f1: {dev_f1} | precision: {dev_precision} | recall: {dev_recall} | accuracy: {dev_accuracy}')

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimize.state_dict(),
                'best_f1': best_f1,
            }, checkpoint_path)

            print(f"Đã lưu checkpoint: {checkpoint_path}")



@torch.no_grad()
def evaluation(model, loader, device):
    model.eval()
    preds, truth = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        y_preds = torch.argmax(logits, dim=1)

        preds += y_preds.cpu().tolist()
        truth += yb.tolist()

    f1 = f1_score(truth, preds, average='macro')
    accuracy = accuracy_score(truth, preds)
    precision = precision_score(truth, preds, average="macro")
    recall = recall_score(truth, preds, average="macro")
    return f1, accuracy, precision, recall