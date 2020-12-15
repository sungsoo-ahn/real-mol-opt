import random
import numpy as np
import torch
from rdkit import rdBase
from rdkit import Chem
import re

from torch.nn.utils.rnn import pad_sequence
from guacamol.utils.chemistry import canonicalize

class Vocabulary:
    bos_token = '<bos>'
    eos_token = '<eos>'
    pad_token = '<pad>'
    def __init__(self):
        self.tokens = self.ordinary_tokens + [self.bos_token, self.eos_token, self.pad_token]
        self.pattern = self.pattern
        self.regex = re.compile(self.pattern)
        self.token2id = {token: id_ for id_, token in enumerate(self.tokens)}
        self.id2token = {id_: token for id_, token in enumerate(self.tokens)}
        self.max_length = self.max_smiles_length + 2

    def __len__(self):
        return len(self.tokens)

    @property
    def bos_id(self):
        return self.token2id[self.bos_token]

    @property
    def eos_id(self):
        return self.token2id[self.eos_token]

    @property
    def pad_id(self):
        return self.token2id[self.pad_token]

    def id2token(self, id_):
        return self.id2token[id_]

    def string2vec(self, string, add_bos, add_eos):
        ids = [self.token2id[token] for token in self.regex.findall(string)]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        return torch.tensor(ids)

    def vec2string(self, vec, rem_bos, rem_eos):
        ids = vec.tolist()
        try:
            assert ids[0] == self.bos_id
            assert len(ids) == self.max_length or ids[-1] == self.eos_id
        except:
            print(ids)
            assert False

        if len(ids) == 0:
            return None
        if len(ids) > self.max_length:
            return None
        if len(ids) == self.max_length and ids[-1] == self.eos_id:
            return None

        if rem_bos:
            ids = ids[1:]
        if rem_eos:
            ids = ids[:-1]

        string = "".join([self.id2token[id_] for id_ in ids])

        return string

    def vec2smiles(self, vec, rem_bos, rem_eos):
        string = self.vec2string(vec, rem_bos, rem_eos)
        if string is None:
            return None

        smiles = canonicalize(string)
        if smiles is None or len(smiles) == 0:
            return None
        if len(smiles) > self.max_smiles_length:
            return None

        return smiles

    def collate(self, batch):
        strings = batch
        vecs = [self.string2vec(string, add_bos=True, add_eos=True) for string in strings]
        vecs = list(sorted(vecs, key=lambda tsr: tsr.size(0), reverse=True))
        lengths = torch.tensor([vec.size(0) for vec in vecs])
        vecs = pad_sequence(vecs, batch_first=True, padding_value=self.pad_id)
        return vecs, lengths

class ZINCVocabulary(Vocabulary):
    ordinary_tokens = (
        "# $ ( ) * + - . / 0 1 2 3 4 5 6 7 8 9 : = > ? @ @@ Br C Cl F H I N O P S [ \\ ] b c n o p s ~".split()
        )
    pattern = (
        "(\[|\]|Br|Cl?|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@@?|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )
    max_smiles_length = 100

def load_vocab(name):
    if name == "zinc":
        return ZINCVocabulary()