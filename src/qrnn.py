import os
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
from src import settings
from src.wav_to_mfcc import extract_mfcc


class MusicDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(np.asarray(self.data[idx])).float(), self.labels[idx]


def collate_fn(batch):
    data, labels = zip(*batch)
    padded_data = pad_sequence([torch.tensor(d) for d in data], batch_first=True)
    labels = torch.tensor(labels)
    return padded_data, labels


def load_data():
    paths = {
        'train': settings.BASE_PATH,
        'eval': settings.BASE_PATH_EVAL,
        'test': settings.BASE_PATH_EVAL
    }

    dataloaders = {}
    for stage, base_path in paths.items():
        data = []
        labels = []
        for genre in settings.GENRES:
            path = base_path + genre
            for filename in os.listdir(path):
                # print(f'Load {genre}: {filename}')
                data.append(extract_mfcc(path + '/' + filename))
                labels.append(settings.GENRES_MAP[genre])

        batch = []
        for i in range(len(data)):
            batch.append((data[i], labels[i]))

        data, labels = collate_fn(batch)

        music_dataset = MusicDataset(data, labels)
        dataloaders[stage] = DataLoader(music_dataset, batch_size=32, shuffle=True)
    return dataloaders


class QRNNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int, mode: str = "f", zoneout: float = 0.0):
        super(QRNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.mode = mode
        self.zoneout = zoneout

        self.zoneout_distribution = torch.distributions.Bernoulli(probs=self.zoneout)
        self.pad = nn.ConstantPad1d((self.kernel_size - 1, 0), value=0.0)
        self.z_conv = nn.Conv1d(input_size, hidden_size, kernel_size)
        self.f_conv = nn.Conv1d(input_size, hidden_size, kernel_size)

        if self.mode == "fo" or self.mode == "ifo":
            self.o_conv = nn.Conv1d(input_size, hidden_size, kernel_size)

        if self.mode == "ifo":
            self.i_conv = nn.Conv1d(input_size, hidden_size, kernel_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs = shape: [batch x timesteps x features]
        batch, timesteps, _ = inputs.shape

        # Apply convolutions
        inputs = inputs.transpose(1, 2)
        inputs = self.pad(inputs)
        raw_f = self.f_conv(inputs).transpose(1, 2)
        raw_z = self.z_conv(inputs).transpose(1, 2)

        if self.mode == "ifo":
            raw_i = self.i_conv(inputs).transpose(1, 2)
            log_one_minus_f = functional.logsigmoid(raw_i)
        else:
            log_one_minus_f = functional.logsigmoid(-raw_f)

        # Get log values of activations
        log_z = functional.logsigmoid(raw_z)  # Use sigmoid activation
        log_f = functional.logsigmoid(raw_f)

        # Optionally apply zoneout
        if self.zoneout > 0.0:
            zoneout_mask = self.zoneout_distribution.sample(sample_shape=log_f.shape).bool()
            zoneout_mask = zoneout_mask.to(log_f.device)
            log_f = torch.masked_fill(input=log_f, mask=zoneout_mask, value=0.0)
            log_one_minus_f = torch.masked_fill(input=log_one_minus_f, mask=zoneout_mask, value=-1e8)

        # Precalculate recurrent gate values by reverse cumsum
        recurrent_gates = log_f[:, 1:, :]
        recurrent_gates_cumsum = torch.cumsum(recurrent_gates, dim=1)
        recurrent_gates = recurrent_gates - recurrent_gates_cumsum + recurrent_gates_cumsum[:, -1:, :]

        # Pad last timestep
        padding = torch.zeros([batch, 1, self.hidden_size], device=recurrent_gates.device)
        recurrent_gates = torch.cat([recurrent_gates, padding], dim=1)

        # Calculate expanded recursion by cumsum (logcumsumexp in log space)
        log_hidden = torch.logcumsumexp(log_z + log_one_minus_f + recurrent_gates, dim=1)
        hidden = torch.exp(log_hidden - recurrent_gates)

        # Optionally multiply by output gate
        if self.mode == "fo" or self.mode == "ifo":
            o = torch.sigmoid(self.o_conv(inputs)).transpose(1, 2)
            hidden = hidden * o

        return hidden


class QRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_layers: int, hidden_size: int, kernel_size: int,
                 mode: str = "f", zoneout: float = 0.0, dropout: float = 0.0):
        super(QRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.mode = mode
        self.zoneout = zoneout
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.layers = []
        for layer in range(self.num_layers):
            input_size = self.embedding_dim if layer == 0 else self.hidden_size
            self.layers.append(
                QRNNLayer(
                    input_size=input_size, hidden_size=self.hidden_size, kernel_size=self.kernel_size, mode=self.mode,
                    zoneout=self.zoneout
                )
            )

            if layer + 1 < self.num_layers:
                self.layers.append(nn.Dropout(p=self.dropout))

        self.rnn = nn.Sequential(*self.layers)

        self.classifier = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        encoded = self.rnn(embedded)
        prediction_scores = self.classifier(encoded)
        return prediction_scores

size = 99999999


def moving_avg(old_val, new_val, gamma=0.95):
    if old_val is None:
        return new_val
    return gamma * old_val + (1 - gamma) * new_val


def evaluate_model(model: nn.Module, dataloaders, epochs: int):
    model = model.train()
    optimizer = AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_dataloader = dataloaders["train"]
    validation_dataloader = dataloaders["validation"]
    test_dataloader = dataloaders["test"]

    pbar = tqdm(total=epochs * len(train_dataloader), desc="Training Progress")
    running_loss = None

    validation_scores = []
    test_scores = []

    for epoch in range(epochs):
        model = model.train()

        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.long())
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            loss_item = loss.detach().cpu().item()
            running_loss = moving_avg(running_loss, loss_item)
            pbar.update(1)
            pbar.set_postfix_str(f"Avg. Loss: {running_loss:.5f}")

        model = model.eval()

        with torch.no_grad():
            # Get validation score
            for inputs, targets in validation_dataloader:
                outputs = model(inputs.long())
                loss = criterion(outputs, targets)
                validation_scores.append(torch.exp(loss).item())

            # Get test score
            for inputs, targets in test_dataloader:
                outputs = model(inputs.long())
                loss = criterion(outputs, targets)
                test_scores.append(torch.exp(loss).item())

    pbar.close()
    best_validation_epoch = np.argmin(validation_scores)
    return test_scores[best_validation_epoch]


if __name__ == '__main__':
    epochs = 1
    dataloads = load_data()
    qrnn_model = QRNN(vocab_size=size, embedding_dim=300, num_layers=2, hidden_size=64, kernel_size=1,
                      mode="fo", zoneout=0.1, dropout=0.4)
    test_ppl_qrnn = evaluate_model(model=qrnn_model, dataloaders=dataloads, epochs=epochs)
    print(f"Test PPL QRNN: {test_ppl_qrnn:.5f}")
