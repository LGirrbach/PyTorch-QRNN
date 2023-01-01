import torch
import numpy as np
import torch.nn as nn

from tqdm.auto import tqdm
from torch.optim import AdamW
from qrnn_layer import QRNNLayer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


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

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        encoded = self.rnn(embedded)
        prediction_scores = self.classifier(encoded)
        return prediction_scores


class LSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_layers: int, hidden_size: int, dropout: float = 0.0):
        super(LSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim, hidden_size=self.hidden_size, dropout=self.dropout, bidirectional=False,
            num_layers=self.num_layers
        )

        self.classifier = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)

        # Pack sequence
        lengths = torch.clamp(lengths, 1)  # Enforce all lengths are >= 1 (required by pytorch)
        embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        # Apply LSTM
        encoded, _ = self.rnn(embedded)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True)

        # Get prediction scores
        prediction_scores = self.classifier(encoded)
        return prediction_scores


def load_file(filename: str):
    sentences = []
    with open(filename) as df:
        for line in df:
            sentences.append(line.strip().split(" "))
    return sentences


class LMDataset(Dataset):
    def __init__(self, sentences):
        super(LMDataset, self).__init__()

        self.sentences = sentences

    def __getitem__(self, idx):
        return self.sentences[idx]

    def __len__(self):
        return len(self.sentences)


def get_dataloaders(batch_size: int):
    train_data = load_file("data/train.txt")
    validation_data = load_file("data/validation.txt")
    test_data = load_file("data/test.txt")

    vocabulary = ["<pad>", "<sos>", "<eos>"] + list(sorted(set.union(*(set(sentence) for sentence in train_data))))
    token2idx = {token: idx for idx, token in enumerate(vocabulary)}
    unk_token_idx = token2idx["<unk>"]

    train_dataset = LMDataset(train_data)
    validation_dataset = LMDataset(validation_data)
    test_dataset = LMDataset(test_data)

    def batch_collate(batch):
        sentences = [["<sos>"] + sentence + ["<eos>"] for sentence in batch]
        lengths = torch.tensor([len(sentence) for sentence in sentences]).long()
        indexed_sentences = [[token2idx.get(token, unk_token_idx) for token in sentence] for sentence in sentences]
        indexed_sentences = [torch.tensor(sentence).long() for sentence in indexed_sentences]
        indexed_sentences = pad_sequence(indexed_sentences, batch_first=True, padding_value=0)
        return indexed_sentences, lengths

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=batch_collate),
        "validation": DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_collate),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_collate),
        "vocabulary": vocabulary
    }


def moving_avg(old_val, new_val, gamma=0.95):
    if old_val is None:
        return new_val
    return gamma * old_val + (1 - gamma) * new_val


def evaluate_model(model: nn.Module, dataloaders, epochs: int):
    model = model.train().cuda()
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

        for sentences, lengths in train_dataloader:
            optimizer.zero_grad()

            inputs = sentences[:, :-1].cuda()
            labels = sentences[:, 1:].flatten().cuda()

            prediction_scores = model(inputs, lengths - 1)
            prediction_scores = prediction_scores.view(-1, prediction_scores.shape[-1])
            loss = criterion(prediction_scores, labels)
            loss.backward()
            optimizer.step()

            loss_item = loss.detach().cpu().item()
            running_loss = moving_avg(running_loss, loss_item)
            pbar.update(1)
            pbar.set_postfix_str(f"Avg. Loss: {running_loss:.5f}")

        model = model.eval()

        with torch.no_grad():
            # Get validation score
            for sentences, lengths in validation_dataloader:
                inputs = sentences[:, :-1].cuda()
                labels = sentences[:, 1:].flatten().cuda()

                prediction_scores = model(inputs, lengths - 1)
                prediction_scores = prediction_scores.view(-1, prediction_scores.shape[-1])
                loss = criterion(prediction_scores, labels)
                validation_scores.append(torch.exp(loss).item())

            # Get test score
            for sentences, lengths in test_dataloader:
                inputs = sentences[:, :-1].cuda()
                labels = sentences[:, 1:].flatten().cuda()

                prediction_scores = model(inputs, lengths - 1)
                prediction_scores = prediction_scores.view(-1, prediction_scores.shape[-1])
                loss = criterion(prediction_scores, labels)
                test_scores.append(torch.exp(loss).item())

    pbar.close()
    best_validation_epoch = np.argmin(validation_scores)
    return test_scores[best_validation_epoch]


if __name__ == '__main__':
    epochs = 40
    data = get_dataloaders(batch_size=32)

    qrnn_model = QRNN(
        vocab_size=len(data["vocabulary"]), embedding_dim=300, num_layers=2, hidden_size=640, kernel_size=5,
        mode="fo", zoneout=0.1, dropout=0.4
    )

    test_ppl_qrnn = evaluate_model(model=qrnn_model, dataloaders=data, epochs=epochs)
    print(f"Test PPL QRNN: {test_ppl_qrnn:.5f}")
    print()

    lstm_model = LSTM(
        vocab_size=len(data["vocabulary"]), embedding_dim=300, num_layers=2, hidden_size=640, dropout=0.4
    )
    test_ppl_lstm = evaluate_model(model=lstm_model, dataloaders=data, epochs=epochs)
    print(f"Test PPL LSTM: {test_ppl_lstm:.5f}")
