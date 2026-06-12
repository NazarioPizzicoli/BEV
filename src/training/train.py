import torch


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0

    for feat, quest, ans in loader:
        # Sposta i dati sul device corretto (CPU o GPU)
        feat, quest, ans = feat.to(device), quest.to(device), ans.to(device)

        # Reset dei gradienti
        optimizer.zero_grad()

        # Forward pass
        out = model(feat, quest)
        loss = criterion(out, ans)

        # Backward pass e ottimizzazione
        loss.backward()
        optimizer.step()

        # Accumulo metriche
        total_loss += loss.item()
        correct += (out.argmax(1) == ans).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0

    # Disabilita il calcolo dei gradienti per risparmiare memoria e velocizzare
    with torch.no_grad():
        for feat, quest, ans in loader:
            feat, quest, ans = feat.to(device), quest.to(device), ans.to(device)

            # Forward pass
            out = model(feat, quest)
            loss = criterion(out, ans)

            # Accumulo metriche
            total_loss += loss.item()
            correct += (out.argmax(1) == ans).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)
