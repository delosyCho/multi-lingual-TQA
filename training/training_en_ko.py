from __future__ import absolute_import, division, print_function

import logging
import random
import sys

import numpy as np
import torch

from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForSequenceClassification

import DataHolder as Dataholder
import DataHolderKorTQA as Dataholder2

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
    epoch = 100
    batch_size = 24
    model = ModelForQuestionAnsweringV2(set_additional_embedding=True)

    output_model_file = "xlm_roberta_lm.bin"
    loaded_state_dict = torch.load(output_model_file)
    new_state_dict = OrderedDict()
    for n, v in loaded_state_dict.items():
        name = n.replace("module.", "").replace('encoder.roberta.encoder', 'encoder.encoder')\
            .replace('encoder.roberta.embeddings', 'encoder.embeddings')
        # .module이 중간에 포함된 형태라면 (".module","")로 치환
        new_state_dict[name] = v
        print(name, v.shape)
    model.load_state_dict(new_state_dict, strict=False)

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

    data_holder2 = Dataholder2.Dataholder(model_name=model_name)
    data_holder2.batch_size = batch_size

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
        batch2 = data_holder2.next_batch()

        if n_gpu == 1:
            batch = tuple(t.to(device) for t in batch)
            batch2 = tuple(t.to(device) for t in batch2)

        input_ids, attention_mask, position_ids, token_type_ids, labels_column, labels_row, labels_token, labels_agg = batch
        input_ids2, attention_mask2, position_ids2, token_type_ids2, labels_column2, labels_row2, labels_token2, labels_agg2 = batch2

        input_ids = torch.cat([input_ids, input_ids2], dim=0)
        attention_mask = torch.cat([attention_mask, attention_mask2], dim=0)
        position_ids = torch.cat([position_ids, position_ids2], dim=0)
        token_type_ids = torch.cat([token_type_ids, token_type_ids2], dim=0)

        labels_column = torch.cat([labels_column, labels_column2], dim=0)
        labels_row = torch.cat([labels_row, labels_row2], dim=0)
        labels_token = torch.cat([labels_token, labels_token2], dim=0)
        labels_agg = torch.cat([labels_agg, labels_agg2], dim=0)
        # print(labels_column)
        # print(labels_row)
        # print(labels_agg)
        loss_column, loss_row, loss_selection, loss_agg, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            labels_column=labels_column,
            labels_row=labels_row,
            labels_agg=labels_agg,
            labels_token=labels_token,
        )

        if n_gpu > 1:
            loss_column = loss_column.mean()
            loss_row = loss_row.mean()
            loss_selection = loss_selection.mean()
            if loss_agg is not None:
                loss_agg = loss_agg.mean()
            else:
                loss_agg = torch.tensor(0)

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

    output_model_file = "wtq_pretrained-xlm-robera_en_ko.bin".format(model_name)

    torch.save(model.state_dict(), output_model_file)


if __name__ == "__main__":
    main()