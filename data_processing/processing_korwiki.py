from transformers import AutoTokenizer, TapasTokenizer

import numpy as np
import pandas as pd

import random

from random import randrange
from multiprocessing import Pool

import json
import time

import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# JSON 파일 열기
with open('KorWikiTabular.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

data = data['data']

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
tokenizer_tapas = TapasTokenizer.from_pretrained('google/tapas-base')


# 데이터를 파이썬 객체로 로드한 후 사용
print(data[0])
print(len(data))

# for i in range(len(data)):


def process_data(i):
    max_length = 512
    input_ids = np.zeros(shape=[1, max_length], dtype=np.int32)
    attention_mask = np.zeros(shape=[1, max_length], dtype=np.int16)
    token_type_ids = np.zeros(shape=[1, max_length, 7], dtype=np.int8)
    label_ids = np.full(shape=[1, max_length], fill_value=-100, dtype=np.int32)
    try:
        table_2d = data[i]['TBL']
        question = data[i]['Description']
        caption = data[i]['Caption']

        if len(table_2d[0]) == 2 and caption.find('INFOBOX') != -1:
            new_table_2d = []

            row1 = []
            row2 = []

            for r in range(len(table_2d)):
                row1.append(table_2d[r][0])
                row2.append(table_2d[r][1])
            new_table_2d.append(row1)
            new_table_2d.append(row2)
            table_2d = new_table_2d[:]

        # if len(table_2d) < 3 and i > len(data) // 3:
        #     # print('1', len(table_2d), len(table_2d[0]))
        #     return None

        for r in range(len(table_2d) - 1):
            if len(table_2d[r]) != len(table_2d[r + 1]):
                return None, 0, None, None, None

        table = {}
        num_col = len(table_2d[0])
        num_row = len(table_2d)
        if num_row > 300:
            # print('2')
            return None, 1, None, None, None

        for c in range(num_col):
            cols = []
            for r in range(1, len(table_2d)):
                cols.append(str(table_2d[r][c]))
            table[table_2d[0][c]] = cols
        table = pd.DataFrame.from_dict(table)
    except:
        return None, 2, None, None, None

    num_of_row = len(table_2d)
    num_of_col = len(table_2d[0])
    num_of_cell = num_of_row * num_of_col

    if num_of_cell > 150:
        return None, None, None, None, None

    if num_of_cell < 5:
        # print('4')
        return None, None, None, None, None

    query_tokens = tokenizer.tokenize(question)

    input_tokens = ['<s>']
    input_tokens.extend(query_tokens)
    input_tokens.append('</s>')

    query_length = len(input_tokens)

    row_ids = [0] * len(input_tokens)
    col_ids = [0] * len(input_tokens)
    ranks = [0] * len(input_tokens)
    ranks2 = [0] * len(input_tokens)

    pos_ids = []
    pos_ids.extend(range(len(input_tokens)))

    inputs_tapex = tokenizer_tapas(table=table, queries=question, return_tensors="np")
    token_type_inputs = np.zeros(shape=[len(table_2d) + 1, len(table_2d[0]) + 1, 7], dtype=np.int32)

    for j in range(inputs_tapex['input_ids'].shape[1]):
        if inputs_tapex['token_type_ids'][0, j, 0] == 1:
            r_ix = inputs_tapex['token_type_ids'][0, j, 2]
            c_ix = inputs_tapex['token_type_ids'][0, j, 1]  # -1
            token_type_inputs[r_ix, c_ix, :] = inputs_tapex['token_type_ids'][0, j, :]
        else:
            token_type_inputs[0, 0, :] = inputs_tapex['token_type_ids'][0, 2, :]

    for r, tr in enumerate(table_2d):
        row_statement = 'row ' + str(r) + ' :'
        statement_tokens = tokenizer.tokenize(row_statement)

        for t, token in enumerate(statement_tokens):
            input_tokens.append(token)
            row_ids.append(r)
            col_ids.append(0)
            pos_ids.append(t)
            ranks.append(0)
            ranks2.append(0)

        for c, td in enumerate(tr):
            # print(i, len(tr))
            cell_tokens = tokenizer.tokenize(td)

            for t, token in enumerate(cell_tokens):
                input_tokens.append(token)
                row_ids.append(r)
                col_ids.append(c + 1)
                pos_ids.append(t)
                ranks.append(token_type_inputs[r, c + 1, 4])
                ranks2.append(token_type_inputs[r, c + 1, 5])

            input_tokens.append('<s>')
            row_ids.append(r)
            col_ids.append(c + 1)
            pos_ids.append(len(cell_tokens))
            ranks.append(token_type_inputs[r, c + 1, 4])
            ranks2.append(token_type_inputs[r, c + 1, 5])

    length = len(input_tokens)
    if length > max_length:
        length = max_length

    ids = tokenizer.convert_tokens_to_ids(input_tokens)

    if len(ids) > 600:
        return None, 3, None, None, None

    # print(len(ids))

    count = 0
    for j in range(length):
        input_ids[count, j] = ids[j]
        attention_mask[count, j] = 1
        if j < query_length:
            token_type_ids[count, j, 0] = 0
        else:
            token_type_ids[count, j, 0] = 1
        token_type_ids[count, j, 1] = col_ids[j]
        token_type_ids[count, j, 2] = row_ids[j]
        token_type_ids[count, j, 3] = ranks[j]
        token_type_ids[count, j, 4] = ranks2[j]

    # print(token_type_ids[count, :, 1])
    # input()
    masking_num = randrange(int(num_of_cell * 0.1), int(num_of_cell * 0.2))
    mask_ids = tokenizer.convert_tokens_to_ids(['<mask>'])

    indices_to_mask = []
    for _ in range(masking_num):
        rand_row = randrange(0, num_of_row)
        rand_col = randrange(0, num_of_col)
        indices_to_mask.append([rand_row, rand_col + 1])

    for j in range(length):
        if ([row_ids[j], col_ids[j]] in indices_to_mask) is True:
            label_ids[count, j] = input_ids[count, j]
            input_ids[count, j] = mask_ids[0]
            if random.randint(0, 10) == 2:
                input_ids[count, j] = random.randint(0, 250002)
            # print(label_ids[count, j])
    if i % 1000 == 0:
        print(i)

    return input_ids, attention_mask, token_type_ids, label_ids, (len(table_2d), len(table_2d[0]))


if __name__ == "__main__":
    # 사용할 CPU 코어 수를 설정합니다.
    num_processes = 36  # 적절한 값을 설정하세요
    total_num = 0

    # 작업 병렬 처리를 위해 Pool을 생성합니다.
    start_time = time.time()

    results = []
    # for n in range(10):
    with Pool(processes=num_processes) as pool:
        # 작업을 병렬로 실행합니다. len(data)
        # partition = len(data) // 10
        # start = n * partition
        # end = (n + 1) * partition
        # [start: end]
        results_ = pool.map(process_data, range(len(data)))
        results.extend(results_)
    total = len(results)
    count = 0

    max_length = 512
    input_ids_ = np.zeros(shape=[total, max_length], dtype=np.int32)
    attention_mask_ = np.zeros(shape=[total, max_length], dtype=np.int16)
    token_type_ids_ = np.zeros(shape=[total, max_length, 7], dtype=np.int8)
    label_ids_ = np.full(shape=[total, max_length], fill_value=-100, dtype=np.int32)
    print(len(results))

    false_counts = [0] * 5

    for result in results:
        input_ids, attention_mask, token_type_ids, label_ids, shape = result

        if input_ids is None:
            if attention_mask is not None:
                false_counts[attention_mask] += 1
            continue
        print(shape)
        input_ids_[count] = input_ids
        attention_mask_[count] = attention_mask
        token_type_ids_[count] = token_type_ids
        label_ids_[count] = label_ids
        # print(tokenizer.decode(input_ids_[count, :256]))
        count += 1
    print('count:', count, false_counts)
    np.save('xlm_roberta_tapas/input_ids_lm', input_ids_[:count])
    np.save('xlm_roberta_tapas/attention_mask_lm', attention_mask_[:count])
    np.save('xlm_roberta_tapas/token_type_ids_lm', token_type_ids_[:count])
    np.save('xlm_roberta_tapas/label_ids_lm', label_ids_[:count])
    end_time = time.time()

    # 소요된 시간 계산
    elapsed_time = end_time - start_time

    print(f"작업이 소요된 시간: {elapsed_time} 초")





