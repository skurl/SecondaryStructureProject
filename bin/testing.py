import torch
import torch.nn as nn
import torch.optim as optim

PAD_LABEL = -100
START_TOKEN = 0


def test_model(model, test_loader, device, tgt_vocab_size, criterion):
    
    model.eval()

    total_correct = 0
    total_positions = 0
    total_test_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            start_column = torch.zeros(
                (y_batch.size(0), 1),
                dtype=torch.long,
                device=device
            )

            tgt_input = torch.cat([start_column, y_batch[:, :-1]], dim=1)
            tgt_output = y_batch

            output = model(X_batch, tgt_input)
            predicted = output.argmax(dim=-1)
            mask = tgt_output != PAD_LABEL
            total_correct += ((predicted == tgt_output) & mask).sum().item()
            total_positions += mask.sum().item()

            loss = criterion(
                output.contiguous().view(-1, tgt_vocab_size),
                tgt_output.contiguous().view(-1)
            )

            total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_loader)
    
    test_accuracy = total_correct / total_positions * 100

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    return avg_test_loss, test_accuracy