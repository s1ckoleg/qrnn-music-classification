import torch.nn as nn
import torch.optim as optim
from src.qrnn import SimpleRNN
from src.train import train
from src.eval import eval
from src import settings


model = SimpleRNN(input_size=12, hidden_size=512, num_classes=len(set(settings.GENRES)), num_layers=1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == '__main__':
    accuracy = 0
    while accuracy <= 0.9:
        train(model, criterion, optimizer)
        accuracy = eval(model)
    print('Accuracy reached over 90%!!!')