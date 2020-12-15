import random
import numpy as np
import torch
from rdkit import rdBase
from rdkit import Chem
import re

def canonicalize_smiles(smi, raise_error=False):
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.RemoveHs(mol)
        for atom in mol.GetAtoms():
            atom.ClearProp("molAtomMapNumber")
        canonicalized_smi = Chem.MolToSmiles(mol)
        return canonicalized_smi
    except:
        if raise_error:
            assert False
        else:
            return None


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
        if len(ids) == 0:
            return ""
        if rem_bos:
            assert ids[0] == self.bos
            ids = ids[1:]
        if rem_eos:
            assert ids[-1] == self.eos
            ids = ids[:-1]

        string = "".join(map(ids, self.id2token))

        return string

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