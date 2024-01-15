from __future__ import absolute_import, division, print_function

import logging
import random

import numpy as np
import torch

from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForMaskedLM

import DataHolderLM

from collections import OrderedDict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() and not False else "cpu")
    n_gpu = torch.cuda.device_count()
    print('n gpu:', n_gpu)

    seed = 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Prepare model

    model = Model.TableSpecificBERTForMaskedLM()
    model.to(device)
    model.train()

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)

    # Prepare optimizer
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    opimizer = None
    scheduler = None

    for data_num in range(1, 10):
        data_holder = Dataholder.DataHolder(data_number=str(data_num), path='bigbird_inputs')
        data_holder.batch_size = 8

        num_step = int(data_holder.input_ids.shape[0] / data_holder.batch_size)

        if opimizer is None:
            optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-6)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(num_step * 0.1), num_training_steps=num_step
            )

        model.train()
        epoch = 0

        max_grad_norm = 1.0

        total_loss = 0
        tr_step = 0

        num_step = int(data_holder.input_ids.shape[0] / data_holder.batch_size)

        for step in range(num_step):
            batch = data_holder.next_batch()

            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)
            length = 512
            input_ids, attention_mask, token_type_ids, position_ids, label_ids = batch
            prediction_scores, loss = model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids[:, :length, :],
                                            position_ids=None,
                                            labels=label_ids[:, :length])

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            tr_step += 1
            total_loss += loss.item()
            mean_loss = total_loss / tr_step
            print(data_num, ':', step, loss.item(), mean_loss, '/', num_step)

        epoch += 1
        logger.info("** ** * Saving file * ** **")
        output_model_file = "bigbird_tapas_lm.bin"
        torch.save(model.state_dict(), output_model_file)


if __name__ == "__main__":
    main()