import torch
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def train_epochs(model, loader, dev_loader, criterion, optimizer, device, epochs: int=10, checkpoint_path="checkpoints/best_model.pth"):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}:')
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()

        f1, precision, recall, accuracy = evaluation(model, dev_loader, device)
        print(f'f1: {f1} | precision: {precision} | recall: {recall} | accuracy: {accuracy}')

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, checkpoint_path)

            print(f"Đã lưu checkpoint: {checkpoint_path}")

@torch.no_grad()
def evaluation(model, test, device):
    model.eval()

    truth, preds = [], []
    for xb, yb in test:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        y_pred = torch.argmax(logits, dim=1)

        preds += y_pred.cpu().tolist()
        truth += yb.cpu().tolist()

    f1 = f1_score(truth, preds, average="macro", zero_division=0)
    precision = precision_score(truth, preds, average="macro", zero_division=0)
    recall = recall_score(truth, preds, average="macro", zero_division=0)
    accuracy = accuracy_score(truth, preds)

    return f1, precision, recall, accuracy
