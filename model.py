import os
import json

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical

def sample_smis(model, vocab, device, num_samples):
    vecs, lengths = model.sample(vocab, device, num_samples)
    smis = [vocab.vec2smiles(vecs[idx][:lengths[idx]], rem_eos=True, rem_bos=True) for idx in range(num_samples)]
    return smis

def sample_smis_by_batch(model, vocab, device, num_samples, batch_size):
    offset = 0
    smis = []
    while offset < num_samples:
        cur_batch_size = min(batch_size, num_samples - offset)
        offset += batch_size
        cur_smis = sample_smis(model, vocab, device, cur_batch_size)
        smis += cur_smis

    return smis

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

    def sample(self, vocab, device, num_samples):
        starts = torch.LongTensor(num_samples, 1).fill_(vocab.bos_id).to(device)
        vecs = torch.LongTensor(num_samples, vocab.max_length).fill_(vocab.pad_id).to(device)
        vecs[:, 0] = starts.squeeze(dim=1)

        lengths = torch.LongTensor(num_samples).fill_(1).to(device)
        hiddens = None
        ended = torch.zeros(num_samples, dtype=torch.bool).to(device)

        for time_idx in range(1, vocab.max_length):
            output, _, hiddens = self(starts, lengths=None, hiddens=hiddens)

            # probabilities
            probs = torch.softmax(output, dim=2)

            # sample from probabilities
            distribution = Categorical(probs=probs)
            starts = top_ids = distribution.sample()

            vecs[~ended, time_idx] = top_ids.squeeze(dim=1)[~ended]
            lengths += (~ended).long()
            ended = ended | (top_ids.squeeze(dim=1) == vocab.eos_id).bool()

            if ended.all():
                break

        return vecs, lengths