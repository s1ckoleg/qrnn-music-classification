import torch
from src import settings
from src.dataset_loader import load_data


def train(model, criterion, optimizer):
    data_loader = load_data('train')
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}')

    torch.save(model.state_dict(), settings.MODEL_PATH)
    # print(f"Model saved to {settings.MODEL_PATH}")


