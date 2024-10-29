import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.wav_to_mfcc import extract_mfcc
from src import settings
import numpy as np


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


def load_data(stage):
    if stage == 'train':
        base_path = settings.BASE_PATH
    elif stage == 'eval':
        base_path = settings.BASE_PATH_EVAL
    else:
        raise ValueError('Should be train or eval')
    data = []
    labels = []
    for genre in settings.GENRES:
        path = base_path + genre
        for filename in os.listdir(path):
            print(f'Load {genre}: {filename}')
            data.append(extract_mfcc(path + '/' + filename))
            labels.append(settings.GENRES_MAP[genre])

    batch = []
    for i in range(len(data)):
        batch.append((data[i], labels[i]))

    data, labels = collate_fn(batch)

    music_dataset = MusicDataset(data, labels)
    return DataLoader(music_dataset, batch_size=32, shuffle=True)
