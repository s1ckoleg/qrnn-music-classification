import torch
from src import settings
from src.dataset_loader import load_data


def test(model):
    test_loader = load_data('eval')
    model.load_state_dict(torch.load(settings.MODEL_PATH))

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    return accuracy