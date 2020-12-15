import time
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

import os, sys

from rdkit import Chem
from rdkit.Chem import Descriptors, Mol, rdMolDescriptors
from rdkit.Chem import RDConfig, MolFromSmiles

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

import networkx as nx

from chemistry.guacamol_benchmarks import (
    similarity,
    isomers_c11h24,
    isomers_c9h10n2o2pf2cl,
    median_camphor_menthol,
    median_tadalafil_sildenafil,
    hard_osimertinib,
    hard_fexofenadine,
    ranolazine_mpo,
    perindopril_rings,
    amlodipine_rings,
    sitagliptin_replacement,
    zaleplon_with_other_formula,
    valsartan_smarts,
    decoration_hop,
    scaffold_hop,
    logP_benchmark,
    tpsa_benchmark,
    cns_mpo,
    qed_benchmark,
    isomers_c7h8n2o2,
    pioglitazone_mpo,
    )

from guacamol.common_scoring_functions import (
    TanimotoScoringFunction,
    RdkitScoringFunction,
    CNS_MPO_ScoringFunction,
    IsomerScoringFunction,
    SMARTSScoringFunction,
)
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.goal_directed_score_contributions import uniform_specification
from guacamol.scoring_function import (
    ArithmeticMeanScoringFunction,
    GeometricMeanScoringFunction,
    MoleculewiseScoringFunction,
)
from guacamol.utils.descriptors import (
    num_rotatable_bonds,
    num_aromatic_rings,
    logP,
    qed,
    tpsa,
    bertz,
    mol_weight,
    AtomCounter,
    num_rings,
)

from chemistry.rd_filter import RDFilter

import copy


LOGP_MEAN = 2.4570965532649507
LOGP_STD = 1.4339810636722639
SASCORE_MEAN = 3.0508333383104556
SASCORE_STD = 0.8327034846660627
CYCLEBASIS_CYCLESCORE_MEAN = 0.048152237188108474
CYCLEBASIS_CYCLESCORE_STD = 0.2860582871837183

def penalized_logp(smiles):
    #print(smiles)
    mol = MolFromSmiles(smiles)
    try:
        log_p = Descriptors.MolLogP(mol)
    except:
        print(mol)
        print(smiles)
        return -np.inf

    sa_score = sascorer.calculateScore(mol)

    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    largest_ring_size = max([len(j) for j in cycle_list]) if cycle_list else 0
    cycle_score = max(largest_ring_size - 6, 0)

    log_p = (log_p - LOGP_MEAN) / LOGP_STD
    sa_score = (sa_score - SASCORE_MEAN) / SASCORE_STD
    cycle_score = (cycle_score - CYCLEBASIS_CYCLESCORE_MEAN) / CYCLEBASIS_CYCLESCORE_STD

    return log_p - sa_score - cycle_score

def logp(smiles):
    mol = MolFromSmiles(smiles)
    try:
        log_p = Descriptors.MolLogP(mol)
    except:
        print(mol)
        print(smiles)
        return -np.inf

    log_p = (log_p - LOGP_MEAN) / LOGP_STD

    return log_p

def synth_access(smiles):
    mol = MolFromSmiles(smiles)
    sa_score = sascorer.calculateScore(mol)
    return sa_score

def qed_score(smiles):
    mol = MolFromSmiles(smiles)
    qed_score = qed(mol)
    return qed_score

def load_score_func(name):
    if name == "sa":
        return synth_access
    elif name == "qed":
        return qed_score
    elif name == "rd_filter":
        return RDFilter()
    elif name == "logp":
        return logp
    elif name == "penalized_logp":
        return penalized_logp
