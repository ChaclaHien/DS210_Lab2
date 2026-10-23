import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "microsoft/resnet-50"


def build_resnet50(num_classes, device):
    """Tải ResNet50 pretrained từ HuggingFace và thay classifier cho num_classes."""
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    resnet50  = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True
    ).to(device)
    return processor, resnet50


def finetune_resnet50(resnet50, train_loader, test_loader, device, epochs=10, lr=1e-4):
    opt       = torch.optim.Adam(resnet50.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        resnet50.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(resnet50(X).logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        resnet50.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                preds = resnet50(X).logits.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        acc  = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        print(f"Epoch {epoch+1}/{epochs} | loss={total_loss/len(train_loader):.4f} | acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f}")

    print(f"\n[ResNet50 Fine-tune Test] acc={acc:.4f} | precision={prec:.4f} | recall={rec:.4f} | f1={f1:.4f}")
    return acc, prec, rec, f1
