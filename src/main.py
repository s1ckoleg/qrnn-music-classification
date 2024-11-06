import torch.nn as nn
import torch.optim as optim
from src.eval import test
from src.train import train
from src import settings
from src.rnn import RNN


model = RNN(input_size=64, hidden_size=512, num_classes=len(set(settings.GENRES)), num_layers=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


if __name__ == '__main__':
    # test(model)
    accuracy = 0
    counter = 1
    while accuracy <= 0.8:
        accuracy = 0
        print('Iteration:', counter)
        train(model, criterion, optimizer)
        for i in range(100):
            curr_accuracy = test(model)
            accuracy += curr_accuracy
            print(f'Accuracy {i + 1}: {curr_accuracy * 100:.2f}%')
        accuracy /= 100
        print(f'Total accuracy {accuracy * 100:.2f}%')
        print()
        counter += 1

    print('Accuracy reached over 80%!!!')
