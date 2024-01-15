from __future__ import absolute_import, division, print_function

import logging
import random
import sys

import numpy as np
import torch

from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForSequenceClassification

import DataHolder as Dataholder
from modeling_QA import ModelForQuestionAnswering, ModelForQuestionAnsweringV2

from collections import OrderedDict


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() and not False else "cpu")
    n_gpu = torch.cuda.device_count()

    # device = torch.device("cuda:0" if torch.cuda.is_available() and not False else "cpu")
    # n_gpu = 1 #torch.cuda.device_count()

    print('n gpu:', n_gpu)

    seed = 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Prepare model
    model_name = 'roberta'

    train_selection = False
    epoch = 50
    batch_size = 48
    model = ModelForQuestionAnsweringV2()
    model.to(device)
    model.train()

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)
    # input()
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

    data_holder = Dataholder.Dataholder(model_name=model_name)
    data_holder.batch_size = batch_size

    num_step = int(data_holder.input_ids.shape[0] / data_holder.batch_size) * epoch

    if opimizer is None:
        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-6)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(num_step * 0.2), num_training_steps=num_step
        )

    model.train()
    epoch = 0

    max_grad_norm = 1.0

    total_loss = 0
    tr_step = 0

    # num_step = int(data_holder.input_ids.shape[0] / data_holder.batch_size) * 5

    for step in range(num_step):
        if step % 1000 == 0:
            total_loss = 0
            tr_step = 0

        batch = data_holder.next_batch()

        if n_gpu == 1:
            batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, row_mask, position_ids, token_type_ids, labels, labels_column, labels_row, labels_token, labels_agg = batch

        loss_column, loss_row, loss_selection, loss_agg, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            row_mask=row_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            labels_column=labels_column,
            labels_row=labels_row,
            labels_agg=labels_agg,
            labels_token=labels_token,
            labels=labels
        )

        if n_gpu > 1:
            loss_column = loss_column.mean()
            loss_row = loss_row.mean()
            loss_selection = loss_selection.mean()
            loss_agg = loss_agg.mean()

            loss = loss.mean()  # mean() to average on multi-gpu.

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scheduler.step()
        optimizer.step()
        optimizer.zero_grad()

        tr_step += 1
        if torch.isnan(loss).item() is True:
            loss = torch.tensor(0.0)
        total_loss += loss.item()
        mean_loss = total_loss / tr_step
        print(step, loss.item(), 'col:', loss_column.item(), 'row:', loss_row.item(), 'sel:', loss_selection.item(),
              'agg:', loss_agg.item(),
              mean_loss, step, '/', num_step)
        print('-----------------------------')
    epoch += 1
    logger.info("** ** * Saving file * ** **")

    output_model_file = "wtq_robera.bin".format(model_name)

    torch.save(model.state_dict(), output_model_file)


if __name__ == "__main__":
    main()