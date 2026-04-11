"""
Behavioral cloning avec LSTM — correction.

Lecture conseillee : apres avoir lu ce script, reponds a ces questions
AVANT de regarder les reponses dans `readme.md` section "Piege distribution
shift":
1. Pourquoi pack_padded_sequence ? (indice : padding + LSTM)
2. Pourquoi moyennee la loss au niveau token ? (pas au niveau sequence)
3. Qu'est-ce qui garantit le determinisme du training ?
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from generate_traces import ACTION_NAMES, STATE_DIM, generate_dataset

SEED = 42
N_ACTIONS = len(ACTION_NAMES)


class TraceDataset(Dataset):
    def __init__(self, traces: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.traces = traces

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s, a = self.traces[idx]
        return torch.from_numpy(s), torch.from_numpy(a)


def _collate(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """Pad sequences, renvoie (padded_states, padded_actions, lengths)."""
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    states = [b[0] for b in batch]
    actions = [b[1] for b in batch]
    lengths = torch.tensor([len(s) for s in states], dtype=torch.long)
    states_pad = pad_sequence(states, batch_first=True)
    actions_pad = pad_sequence(actions, batch_first=True, padding_value=-100)  # -100 ignore dans CE
    return states_pad, actions_pad, lengths


class BCModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_embed = nn.Linear(STATE_DIM, 32)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.head = nn.Linear(64, N_ACTIONS)

    def forward(self, states: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # states : (B, T, 7), lengths : (B,)
        x = self.state_embed(states)
        # pack pour eviter que le LSTM lise le padding -- ecrirait n'importe quoi
        # apres la fin de sequence et polluerait la loss.
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.lstm(packed)
        from torch.nn.utils.rnn import pad_packed_sequence
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return self.head(out)  # (B, T, N_ACTIONS)


def _split(data, ratios=(0.7, 0.15, 0.15)):
    n = len(data)
    a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    return data[:a], data[a:b], data[b:]


def train() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data = generate_dataset()
    train_data, val_data, test_data = _split(data)

    train_loader = DataLoader(TraceDataset(train_data), batch_size=16, shuffle=True, collate_fn=_collate)
    test_loader = DataLoader(TraceDataset(test_data), batch_size=16, shuffle=False, collate_fn=_collate)

    model = BCModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # ignore_index=-100 : les positions paddees n'entrent pas dans la loss.
    # Sans ca, on apprendrait a predire "0" pour le padding -> biais.
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(20):
        model.train()
        total, n_steps = 0.0, 0
        for states, actions, lengths in train_loader:
            opt.zero_grad()
            logits = model(states, lengths)  # (B, T, N)
            # CE attend (B*T, N) et (B*T,)
            loss = loss_fn(logits.reshape(-1, N_ACTIONS), actions.reshape(-1))
            loss.backward()
            opt.step()
            total += loss.item() * lengths.sum().item()
            n_steps += lengths.sum().item()
        print(f"Epoch {epoch} train loss {total/n_steps:.4f}")

    # Test accuracy per-step
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for states, actions, lengths in test_loader:
            logits = model(states, lengths)
            preds = logits.argmax(dim=-1)
            mask = actions != -100
            correct += ((preds == actions) & mask).sum().item()
            total += mask.sum().item()
    print(f"Test accuracy per-step : {correct/total:.3f}")


if __name__ == "__main__":
    train()
