
import torch
import torch.nn.functional as F
from tqdm import tqdm

def train(model, data_loader, optimizer, criterion, device, mem_weight=1.0):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, data.y)

        # Include MEM loss if applicable
        if hasattr(model, "mem_loss") and model.mem_loss is not None:
            loss += mem_weight * model.mem_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)

            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == data.y).sum().item()
            total += data.y.size(0)

    accuracy = correct / total if total > 0 else 0
    return total_loss / len(data_loader), accuracy
