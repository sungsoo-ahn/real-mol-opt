import os
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
from model import sample_smis_by_batch
import neptune

from util import save_json

from functools import total_ordering
import numpy as np
import random

@total_ordering
class PriorityQueueElement:
    def __init__(self, smi, score):
        self.smi = smi
        self.score = score

    def __eq__(self, other):
        return np.isclose(self.score, other.score)

    def __lt__(self, other):
        return self.score < other.score

    def __hash__(self):
        return hash(self.smi)

class PriorityQueue:
    def __init__(self, size):
        self.elems = []
        self.size = size

    def __len__(self):
        return len(self.elems)

    def add_list(self, smis, scores):
        new_elems = [
            PriorityQueueElement(smi=smi, score=score) for smi, score in zip(smis, scores)
        ]
        self.elems.extend(new_elems)
        self.elems = list(set(self.elems))
        if len(self.elems) > self.size:
            self.elems = list(sorted(self.elems, reverse=True))[:self.size]

    def get_elems(self):
        return tuple(map(list, zip(*[(elem.smi, elem.score) for elem in self.elems])))


class HillClimbScheme:
    def run(
        self,
        model,
        optimizer,
        vocab,
        obj_func,
        num_stages,
        queue_size,
        num_steps_per_stage,
        optimize_batch_size,
        kl_div_coef,
        generate_size,
        generate_batch_size,
        device,
        result_dir,
        disable_neptune,
    ):
        os.makedirs(result_dir, exist_ok=True)
        model = model.to(device)
        storage = PriorityQueue(size=queue_size)
        step_train_loss_record = []
        stage_best_obj_record = []

        with torch.no_grad():
            smis = sample_smis_by_batch(model, vocab, device, generate_size, generate_batch_size)

        smis = [smi for smi in smis if smi is not None]
        objs = [obj_func(smi) for smi in smis]
        storage.add_list(smis=smis, scores=objs)
        for stage in tqdm(range(num_stages)):
            train_loader = torch.utils.data.DataLoader(
                dataset=smis,
                batch_size=optimize_batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=vocab.collate,
            )
            for step in range(num_steps_per_stage):
                try:
                    vecs, lengths = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    vecs, lengths = next(train_iter)

                vecs = vecs.to(device)
                lengths = lengths.to(device)

                optimizer.zero_grad()
                loss = self.compute_loss(model, vocab, vecs, lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                step_train_loss_record.append(loss.item())
                save_json(step_train_loss_record, f"{result_dir}/record_step_train_loss.json")
                if not disable_neptune:
                    neptune.log_metric("step_train_loss", loss)

                torch.save(model.state_dict(), f"{result_dir}/checkpoint_{stage:03d}.pt")


            with torch.no_grad():
                smis = sample_smis_by_batch(model, vocab, device, generate_size, generate_batch_size)

            smis = [smi for smi in smis if smi is not None]
            objs = [obj_func(smi) for smi in smis]
            storage.add_list(smis=smis, scores=objs)

            smis, objs = storage.get_elems()
            best_obj = max(objs)
            stage_best_obj_record.append(best_obj)
            if not disable_neptune:
                neptune.log_metric("stage_best_obj", best_obj)

    def compute_loss(self, model, vocab, vecs, lengths):
        in_vecs = vecs[:, :-1]
        out_vecs = vecs[:, 1:]
        logits, _, _ = model(in_vecs, lengths=lengths - 1, hiddens=None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(2)), out_vecs.reshape(-1), ignore_index=vocab.pad_id, reduction="sum"
        )
        loss /= vecs.size(0)

        return loss