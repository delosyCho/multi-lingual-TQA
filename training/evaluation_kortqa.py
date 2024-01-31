from __future__ import absolute_import, division, print_function

import json

import logging
import random
import sys

import numpy as np
import torch

from transformers import AutoTokenizer

import DataHolderKorTQA as DataHolder

from modeling_QA import ModelForQuestionAnswering, ModelForQuestionAnsweringV2
from collections import OrderedDict

import re
from datasets import load_dataset


id_to_table_dict = {}

dataset_dev = load_dataset("wikitablequestions")['validation']
tables_ = dataset_dev['table']
questions_ = dataset_dev['question']
answer_lists_ = dataset_dev['answers']
id_lists_ = dataset_dev['id']

dataset_test = load_dataset("wikitablequestions")['test']
tables_t = dataset_test['table']
questions_t = dataset_test['question']
answer_lists_t = dataset_test['answers']
id_lists_t = dataset_test['id']


dataset = load_dataset("wikitablequestions")['train']
tables = dataset['table']
questions = dataset['question']
answer_lists = dataset['answers']
id_lists = dataset['id']

questions.extend(questions_)
tables.extend(tables_)
answer_lists.extend(answer_lists_)
id_lists.extend(id_lists_)

questions.extend(questions_t)
tables.extend(tables_t)
answer_lists.extend(answer_lists_t)
id_lists.extend(id_lists_t)

ids = id_lists

for i in range(len(tables)):
    if i % 500 == 0:
        print(i)
    table_dict = tables[i]
    table_2d = []
    table_2d.append(table_dict['header'])
    table_2d.extend(table_dict['rows'])

    id_to_table_dict[ids[i]] = table_2d


def convert_to_float(s):
    # 비숫자 문자 제거
    cleaned = re.sub(r'[^\d.]', '', s)

    # 문자열을 실수로 변환
    return float(cleaned)


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
    aggregation_words = [
        'NONE', 'SUM', 'AVERAGE', 'COUNT'
    ]

    count = 0
    em = 0

    prediction_data = {}
    # device = torch.device("cuda" if torch.cuda.is_available() and not False else "cpu")
    # n_gpu = torch.cuda.device_count()

    device = torch.device("cuda:0" if torch.cuda.is_available() and not False else "cpu")
    n_gpu = 1 #torch.cuda.device_count()

    print('n gpu:', n_gpu)

    seed = 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Prepare model
    model = ModelForQuestionAnsweringV2(set_additional_embedding=True)

    output_model_file = "wtq_pretrained-xlm-robera_en_ko.bin"
    loaded_state_dict = torch.load(output_model_file)
    new_state_dict = OrderedDict()
    for n, v in loaded_state_dict.items():
        name = n.replace("module.", "")  # .module이 중간에 포함된 형태라면 (".module","")로 치환
        new_state_dict[name] = v
        print('check:', name, v.shape)
    model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()

    data_holder = DataHolder.Dataholder()

    num_step = data_holder.input_ids_t.shape[0]
    for step in range(num_step):
        batch = data_holder.test_batch()

        input_ids, attention_mask, position_ids, token_type_ids, labels_col, labels_row = batch

        batch = input_ids, attention_mask, token_type_ids, position_ids
        if n_gpu == 1:
            batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, token_type_ids, position_ids = batch
        column_logits, row_logits, _, _ = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        )

        column_index = torch.argmax(column_logits, dim=1)[0].item()
        row_index = torch.argmax(row_logits, dim=1)[0].item()

        try:
            if row_index == labels_row[0] and column_index == labels_col[0]:
                em += 1
        except:
            None
        count += 1

        print(em / count)
        print('---------------------')


if __name__ == "__main__":
    main()