import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical
import neptune


class DistributionLearningScheme:
    def run(
        self,
        model,
        optimizer,
        vocab,
        train_strings,
        vali_strings,
        num_steps,
        train_batch_size,
        eval_freq,
        eval_batch_size,
        device,
        save_dir,
        disable_neptune,
    ):
        os.makedirs(save_dir, exist_ok=True)
        model = model.to(device)
        def collate(batch):
            strings = batch
            vecs = [vocab.string2vec(string, add_bos=True, add_eos=True) for string in strings]
            vecs = list(sorted(vecs, key=lambda tsr: tsr.size(0), reverse=True))
            lengths = torch.tensor([vec.size(0) for vec in vecs])
            vecs = pad_sequence(vecs, batch_first=True, padding_value=vocab.pad_id)
            return vecs, lengths

        train_loader = torch.utils.data.DataLoader(
            dataset=train_strings,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate,
        )
        vali_loader = torch.utils.data.DataLoader(
            dataset=vali_strings, batch_size=eval_batch_size, shuffle=True, num_workers=0, collate_fn=collate,
        )
        best_vali_loss = np.inf

        for step in tqdm(range(num_steps)):
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

            if not disable_neptune:
                neptune.log_metric("step_train_loss", train_loss)

            if (step + 1) % eval_freq == 0:
                vali_loss = 0.0
                for vecs, lengths in tqdm(vali_loader):
                    vecs = vecs.to(device)
                    lengths = lengths.to(device)
                    with torch.no_grad():
                        loss = self.compute_loss(model, vocab, vecs, lengths)

                    vali_loss += loss / len(vali_loader)

                torch.save(model.state_dict(), f"{save_dir}/{(step//eval_freq):03d}_model.pt")

                if vali_loss < best_vali_loss:
                    best_vali_loss = vali_loss
                    torch.save(model.state_dict(), f"{save_dir}/best_model.pt")

                if not disable_neptune:
                    neptune.log_metric("eval_vali_loss", vali_loss)

    def compute_loss(self, model, vocab, vecs, lengths):
        in_vecs = vecs[:, :-1]
        out_vecs = vecs[:, 1:]
        logits, _, _ = model(in_vecs, lengths=lengths - 1, hiddens=None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(2)), out_vecs.reshape(-1), ignore_index=vocab.pad_id, reduction="sum"
        )
        loss /= vecs.size(0)

        return loss

    def sample(self, model, vocab, device, num_samples):
        starts = torch.LongTensor(num_samples, 1).fill_(vocab.bos_id).to(device)
        vecs = torch.LongTensor(num_samples, vocab.max_length).fill_(vocab.pad_id).to(device)

        lengths = torch.LongTensor(num_samples).fill_(1).to(device)
        hiddens = None
        ended = torch.zeros(num_samples, dtype=torch.bool).to(device)

        for time_idx in range(1, vocab.max_length):
            output, _, hiddens = model(starts, lengths=None, hiddens=hiddens)

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
