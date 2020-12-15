from tqdm import tqdm
from joblib import Parallel, delayed
from rdkit import Chem

from util.smiles.function import smi2tokens
from util.chemistry.benchmarks import load_benchmark

import torch
def canonicalize_smi(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.RemoveHs(mol)
        for atom in mol.GetAtoms():
            atom.ClearProp("molAtomMapNumber")
        canonicalized_smi = Chem.MolToSmiles(mol)
        return canonicalized_smi
    except:
        return None

#def process_smi(smi, quality_score_func, max_len):
#    smi = canonicalize_smi(smi)
#    if smi is None:
#        return None
#    elif len(smi) > max_len:
#        return None
#    elif quality_score_func(smi) < 0.5:
#        return None
#    else:
#        return smi

if __name__ == "__main__":
    DATA_DIR = "./resource/data/zinc"
    SMI_PATH = f"{DATA_DIR}/smis.txt"
    VOCAB_PATH = f"{DATA_DIR}/vocab.txt"
    VEC_PATH = f"{DATA_DIR}/smi_vecs.pt"
    SCORES_PATH = f"{DATA_DIR}/scores.pt"

    with open(SMI_PATH, "r") as f:
        smis = f.read().splitlines()

    with open(VOCAB_PATH, "r") as f:
        vocabs = f.read().splitlines()

    smis = list(sorted(smis))
    vocabs = list(sorted(vocabs))
    with open(SMI_PATH, "w") as f:
        f.write("\n".join(smis))

    with open(VOCAB_PATH, "w") as f:
        f.write("\n".join(vocabs))

    max_smi_len = max([len(smi) for smi in smis])
    print(max_smi_len)

    # preprocess smiles to vector of vocab-indices
    smi_vecs = []
    for smi in tqdm(smis):
        smi = canonicalize_smi(smi)
        if smi is None:
            assert False

        tokens = smi2tokens(smi)
        smi_vec = torch.tensor([vocabs.index(token)+1 for token in tokens])
        smi_vecs.append(smi_vec)

    torch.save(smi_vecs, VEC_PATH)

    # calculate splits according to
    plogp_benchmark = load_benchmark(27)[0]
    plogp_scoring_func = plogp_benchmark.objective.score

    scores = {"penalized_logp": []}
    for smi in tqdm(smis):
        score = plogp_scoring_func(smi)
        scores["penalized_logp"].append(score)

    torch.save(scores, SCORES_PATH)

    #quality_score_func = RDFilter()
    #processed_smis = []
    #for smi in tqdm(smis):
    #    processed_smi = process_smi(smi, quality_score_func, MAX_LEN)
    #    if processed_smi is not None:
    #        processed_smis.append(processed_smi)
    #with open(PROCESSED_PATH, "w") as f:
    #    f.write("\n".join(processed_smis) + "\n")
