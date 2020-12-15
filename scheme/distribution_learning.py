import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from model import sample_smis_by_batch
import neptune

from util import save_json

class DistributionLearningScheme:
    def run(
        self,
        model,
        optimizer,
        vocab,
        train_strings,
        vali_strings,
        train_num_steps,
        train_batch_size,
        eval_freq,
        eval_batch_size,
        eval_generate_size,
        device,
        result_dir,
        disable_neptune,
    ):
        os.makedirs(result_dir, exist_ok=True)
        model = model.to(device)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_strings,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=vocab.collate,
        )
        vali_loader = torch.utils.data.DataLoader(
            dataset=vali_strings, batch_size=eval_batch_size, shuffle=False, num_workers=0, collate_fn=vocab.collate,
        )
        best_vali_loss = np.inf
        step_train_loss_record = []
        eval_vali_loss_record = []
        validity_score_record = []
        for step in tqdm(range(train_num_steps)):
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
            # torch.nn.utils.clip_grad_norm_(num_steps.parameters(), 1.0)
            optimizer.step()

            step_train_loss_record.append(loss.item())
            save_json(step_train_loss_record, f"{result_dir}/record_step_train_loss.json")
            if not disable_neptune:
                neptune.log_metric("step_train_loss", loss)

            if (step + 1) % eval_freq == 0:
                vali_loss = 0.0
                for vecs, lengths in vali_loader:
                    vecs = vecs.to(device)
                    lengths = lengths.to(device)
                    with torch.no_grad():
                        loss = self.compute_loss(model, vocab, vecs, lengths)

                    vali_loss += loss / len(vali_loader)

                torch.save(model.state_dict(), f"{result_dir}/checkpoint_{(step//eval_freq):03d}.pt")

                if vali_loss < best_vali_loss:
                    best_vali_loss = vali_loss
                    torch.save(model.state_dict(), f"{result_dir}/checkpoint_best.pt")

                eval_vali_loss_record.append(vali_loss.item())
                save_json(eval_vali_loss_record, f"{result_dir}/record_eval_vali_loss.json")
                if not disable_neptune:
                    neptune.log_metric("eval_vali_loss", vali_loss)

                with torch.no_grad():
                    smis = sample_smis_by_batch(model, vocab, device, eval_generate_size, eval_batch_size)
                    validity_score = np.mean([smi is not None for smi in smis])
                    validity_score_record.append(validity_score)
                    save_json(validity_score_record, f"{result_dir}/record_validity_score.json")
                    if not disable_neptune:
                        neptune.log_metric("eval_validity_score", validity_score)


    def compute_loss(self, model, vocab, vecs, lengths):
        in_vecs = vecs[:, :-1]
        out_vecs = vecs[:, 1:]
        logits, _, _ = model(in_vecs, lengths=lengths - 1, hiddens=None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(2)), out_vecs.reshape(-1), ignore_index=vocab.pad_id, reduction="sum"
        )
        loss /= vecs.size(0)

        return loss