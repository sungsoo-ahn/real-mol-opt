import os
import json

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RecurrentNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(RecurrentNetwork, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, vecs, lengths, hiddens):
        out = self.embedding(vecs)
        if lengths is not None:
            out = pack_padded_sequence(out, lengths, batch_first=True)

        out, hiddens = self.lstm(out, hiddens)

        if lengths is not None:
            out, _ = pad_packed_sequence(out, batch_first=True)

        out = self.linear(out)
        return out, lengths, hiddens

"""
class Generator:
    def __init__(self, vocab_size, hidden_size, num_layers, device):
        self.model = GeneratorNetwork(vocab_size, hidden_size, num_layers)
        self.model.to(device)

    def get_log_prob(self, vecs, lens, device):
        vecs =

    def get_action_log_prob(self, actions, seq_lengths, device):
        num_samples = actions.size(0)
        actions_seq_length = actions.size(1)
        log_probs = torch.FloatTensor(num_samples, actions_seq_length).to(device)

        number_batches = (num_samples + self.max_sampling_batch_size - 1) // self.max_sampling_batch_size
        remaining_samples = num_samples
        batch_start = 0
        for i in range(number_batches):
            batch_size = min(self.max_sampling_batch_size, remaining_samples)
            batch_end = batch_start + batch_size
            log_probs[batch_start:batch_end, :] = self._get_action_log_prob_batch(
                actions[batch_start:batch_end, :], seq_lengths[batch_start:batch_end], device
            )
            batch_start += batch_size
            remaining_samples -= batch_size

        return log_probs

    def save(self, save_dir):
        self.model.save(save_dir)

    def _get_action_log_prob_batch(self, actions, seq_lengths, device):
        batch_size = actions.size(0)
        actions_seq_length = actions.size(1)

        start_token_vector = self._get_start_token_vector(batch_size, device)
        input_actions = torch.cat([start_token_vector, actions[:, :-1]], dim=1)
        target_actions = actions

        input_actions = input_actions.to(device)
        target_actions = target_actions.to(device)

        output, _ = self.model(input_actions, hidden=None)
        output = output.view(batch_size * actions_seq_length, -1)
        log_probs = torch.log_softmax(output, dim=1)
        log_target_probs = log_probs.gather(dim=1, index=target_actions.reshape(-1, 1)).squeeze(dim=1)
        log_target_probs = log_target_probs.view(batch_size, self.max_seq_length)

        mask = torch.arange(actions_seq_length).expand(len(seq_lengths), actions_seq_length) > (
            seq_lengths - 1
        ).unsqueeze(1)
        log_target_probs[mask] = 0.0

        return log_target_probs

    def _sample_action_batch(self, batch_size, device):
        hidden = None
        inp = self._get_start_token_vector(batch_size, device)

        action = torch.zeros((batch_size, self.max_seq_length), dtype=torch.long).to(device)
        log_prob = torch.zeros((batch_size, self.max_seq_length), dtype=torch.float).to(device)
        seq_length = torch.zeros(batch_size, dtype=torch.long).to(device)

        ended = torch.zeros(batch_size, dtype=torch.bool).to(device)

        for t in range(self.max_seq_length):
            output, hidden = self.model(inp, hidden)

            prob = torch.softmax(output, dim=2)
            distribution = Categorical(probs=prob)
            action_t = distribution.sample()
            log_prob_t = distribution.log_prob(action_t)
            inp = action_t

            action[~ended, t] = action_t.squeeze(dim=1)[~ended]
            log_prob[~ended, t] = log_prob_t.squeeze(dim=1)[~ended]

            seq_length += (~ended).long()
            ended = ended | (action_t.squeeze(dim=1) == self.char_dict.end_idx).bool()

            if ended.all():
                break

        return action, log_prob, seq_length

    def _get_start_token_vector(self, batch_size, device):
        return torch.LongTensor(batch_size, 1).fill_(self.char_dict.begin_idx).to(device)
"""
