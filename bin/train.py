import torch
import torch.nn as nn
import torch.optim as optim

PAD_LABEL = -100


def make_decoder_input(y_batch):
    start_column = torch.zeros(
        (y_batch.size(0), 1),
        dtype=torch.long,
        device=y_batch.device
    )

    tgt_input = torch.cat([start_column, y_batch[:, :-1]], dim=1) # essentially this is a workaround to shift the target sequence to the right by one position and add a start token at the beginning. This is necessary for teacher forcing during training, where the model is given the correct previous token as input to predict the next token. The tgt_output remains unchanged as it represents the actual target sequence that the model should learn to predict.
    tgt_output = y_batch 

    return tgt_input, tgt_output


def run_epoch(model, loader, criterion, optimizer, device, tgt_vocab_size, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        tgt_input, tgt_output = make_decoder_input(y_batch)

        if training:
            optimizer.zero_grad()

        output = model(X_batch, tgt_input)

        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_output.contiguous().view(-1)
        )

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    tgt_vocab_size,
    num_epochs=100,
    learning_rate=0.0001,
    save_path="./outputs/weights.pth"
):
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    model = model.to(device)

    for epoch in range(num_epochs):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            tgt_vocab_size=tgt_vocab_size,
            training=True
        )

        with torch.no_grad():
            val_loss = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                tgt_vocab_size=tgt_vocab_size,
                training=False
            )

        print(
            f"Epoch: {epoch + 1}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )

    torch.save(model.state_dict(), save_path)

    return model