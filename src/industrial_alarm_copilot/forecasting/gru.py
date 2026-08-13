'''Lightweight GRU forecasting model with train-only fitting.'''

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from industrial_alarm_copilot.forecasting.evaluation import ForecastScoreMatrix
from industrial_alarm_copilot.forecasting.sequences import EncodedForecastSequences
from industrial_alarm_copilot.forecasting.vocabulary import EncodedForecastLabels


class AlarmSequenceGRU(nn.Module):
    '''Encode alarm order and machine identity into future-alarm logits.'''

    def __init__(
        self,
        alarm_vocabulary_size: int,
        machine_vocabulary_size: int,
        output_size: int,
        embedding_dim: int,
        hidden_dim: int,
        machine_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.alarm_embedding = nn.Embedding(
            alarm_vocabulary_size + 2,
            embedding_dim,
            padding_idx=0,
        )
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.machine_embedding = nn.Embedding(
            machine_vocabulary_size + 1,
            machine_embedding_dim,
            padding_idx=0,
        )
        self.output = nn.Linear(hidden_dim + machine_embedding_dim, output_size)

    def forward(self, token_ids, sequence_lengths, machine_ids):
        embedded = self.alarm_embedding(token_ids)
        packed = pack_padded_sequence(
            embedded,
            sequence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        machine_state = self.machine_embedding(machine_ids)
        return self.output(torch.cat([hidden[-1], machine_state], dim=1))


@dataclass(frozen=True)
class FittedForecastGRU:
    '''Fitted network plus reproducible architecture and loss history.'''

    module: AlarmSequenceGRU
    alarm_codes: tuple[str, ...]
    sequence_input_version: str
    train_sample_count: int
    loss_history: tuple[float, ...]
    weight_mode: str
    model_version: str


def _validate_alignment(sequences, labels, encoded) -> pd.DataFrame:
    aligned_labels = labels.reset_index(drop=True)
    incident_ids = tuple(aligned_labels['incident_id'].astype(str))
    if incident_ids != sequences.incident_ids or incident_ids != encoded.incident_ids:
        raise ValueError('forecast sequences and labels must be identically aligned')
    return aligned_labels


def fit_forecast_gru(
    sequences: EncodedForecastSequences,
    labels: pd.DataFrame,
    encoded: EncodedForecastLabels,
    embedding_dim: int = 24,
    hidden_dim: int = 48,
    machine_embedding_dim: int = 8,
    batch_size: int = 256,
    epochs: int = 8,
    learning_rate: float = 0.001,
    weight_mode: str = 'none',
    positive_weight_cap: float = 20.0,
    random_seed: int = 0,
) -> FittedForecastGRU:
    '''Fit a deterministic CPU GRU on complete train outcomes only.'''
    if weight_mode not in ('none', 'balanced_capped'):
        raise ValueError('GRU weight_mode must be none or balanced_capped')
    if min(embedding_dim, hidden_dim, machine_embedding_dim, batch_size, epochs) < 1:
        raise ValueError('GRU dimensions, batch size, and epochs must be positive')
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('GRU learning rate must be finite and positive')
    if not np.isfinite(positive_weight_cap) or positive_weight_cap < 1:
        raise ValueError('GRU positive weight cap must be at least one')

    aligned_labels = _validate_alignment(sequences, labels, encoded)
    train_mask = (
        aligned_labels['split'].eq('train')
        & aligned_labels['outcome_is_complete'].astype(bool)
    ).to_numpy()
    train_rows = np.flatnonzero(train_mask)
    if not len(train_rows):
        raise ValueError('at least one complete train outcome is required')

    torch.manual_seed(random_seed)
    generator = torch.Generator().manual_seed(random_seed)
    module = AlarmSequenceGRU(
        alarm_vocabulary_size=len(sequences.encoder.alarm_tokens),
        machine_vocabulary_size=len(sequences.encoder.machine_labels),
        output_size=len(encoded.alarm_codes),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        machine_embedding_dim=machine_embedding_dim,
    )
    target = encoded.matrix[train_rows].toarray().astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(sequences.token_ids[train_rows]),
        torch.from_numpy(sequences.sequence_lengths[train_rows]),
        torch.from_numpy(sequences.machine_ids[train_rows]),
        torch.from_numpy(target),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    positive_weight = None
    if weight_mode == 'balanced_capped':
        positive_count = target.sum(axis=0)
        negative_count = len(target) - positive_count
        weights = np.divide(
            negative_count,
            positive_count,
            out=np.full_like(positive_count, positive_weight_cap),
            where=positive_count > 0,
        )
        positive_weight = torch.from_numpy(
            np.clip(weights, 1.0, positive_weight_cap).astype(np.float32)
        )
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    loss_history = []
    module.train()
    for _ in range(epochs):
        total_loss = 0.0
        for batch_tokens, batch_lengths, batch_machines, batch_target in loader:
            optimizer.zero_grad()
            logits = module(batch_tokens, batch_lengths, batch_machines)
            loss = loss_function(logits, batch_target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_tokens)
        loss_history.append(total_loss / len(dataset))
    return FittedForecastGRU(
        module=module,
        alarm_codes=encoded.alarm_codes,
        sequence_input_version=sequences.encoder.version,
        train_sample_count=len(train_rows),
        loss_history=tuple(loss_history),
        weight_mode=weight_mode,
        model_version=f'gru_{weight_mode}_v1',
    )


def score_forecast_gru(
    model: FittedForecastGRU,
    sequences: EncodedForecastSequences,
    batch_size: int = 512,
) -> ForecastScoreMatrix:
    '''Score all aligned episodes in bounded inference batches.'''
    if sequences.encoder.version != model.sequence_input_version:
        raise ValueError('GRU sequence input version does not match model')
    dataset = TensorDataset(
        torch.from_numpy(sequences.token_ids),
        torch.from_numpy(sequences.sequence_lengths),
        torch.from_numpy(sequences.machine_ids),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    score_batches = []
    model.module.eval()
    with torch.no_grad():
        for tokens, lengths, machines in loader:
            score_batches.append(
                torch.sigmoid(model.module(tokens, lengths, machines)).numpy()
            )
    scores = np.concatenate(score_batches, axis=0)
    return ForecastScoreMatrix(
        incident_ids=sequences.incident_ids,
        alarm_codes=model.alarm_codes,
        scores=scores,
        model_version=model.model_version,
    )
