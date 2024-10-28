import torch.nn as nn
import torch.optim as optim
from src.qrnn import QRNNClassifier, SimpleRNN
from src import settings
from src.dataset_loader import load_data

model = SimpleRNN(input_size=13, hidden_size=1024, num_classes=len(set(settings.GENRES)), num_layers=1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

data_loader = load_data()

# Training loop
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

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}')
