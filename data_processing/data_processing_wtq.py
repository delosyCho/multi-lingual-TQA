from transformers import TapasTokenizer, AutoTokenizer

import numpy as np

import pandas as pd
from datasets import load_dataset

dataset_dev = load_dataset("wikitablequestions")['validation']
tables_ = dataset_dev['table']
questions_ = dataset_dev['question']
answer_lists_ = dataset_dev['answers']

dataset = load_dataset("wikitablequestions")['train']
tables = dataset['table']
questions = dataset['question']
answer_lists = dataset['answers']

questions.extend(questions_)
tables.extend(tables_)
answer_lists.extend(answer_lists_)

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
tokenizer_tapas = TapasTokenizer.from_pretrained('google/tapas-base')

max_length = 1024
input_ids = np.zeros(shape=[len(questions), max_length], dtype=np.int32)
attention_mask = np.zeros(shape=[len(questions), max_length], dtype=np.int32)
token_type_ids = np.zeros(shape=[len(questions), max_length, 7], dtype=np.int32)
numeric_values = np.zeros(shape=[len(questions), max_length], dtype=np.float32)
numeric_scales = np.full(shape=[len(questions), max_length], dtype=np.float32, fill_value=1.0)
answer_float = np.zeros(shape=[len(questions)], dtype=np.float32)

count = 0

for i in range(len(tables)):
    if count % 100 == 0:
        print(count)

    question = questions[i]
    answer_text = answer_lists[i][0]

    try:
        float(answer_text)
    except:
        continue

    table_dict = tables[i]
    table_2d = []
    table_2d.append(table_dict['header'])
    table_2d.extend(table_dict['rows'])

    num_col = len(table_2d[0])
    num_row = len(table_2d)
    if num_row > 100:
        continue

    table = {}
    for c in range(num_col):
        cols = []
        for r in range(1, len(table_2d)):
            cols.append(str(table_2d[r][c]))
        header_word = str(table_2d[0][c])
        table[header_word] = cols
    table = pd.DataFrame.from_dict(table)

    try:
        inputs = tokenizer_tapas(table=table, queries=[question], answer_text=answer_text, answer_coordinates=[[]],
                             padding="max_length", max_length=max_length, return_tensors="np")
    except:
        continue

    numeric_value_array = np.zeros(shape=[num_row, num_col], dtype=np.float32)
    numeric_scale_array = np.zeros(shape=[num_row, num_col], dtype=np.float32)

    for j in range(max_length):
        c = inputs['token_type_ids'][0, j, 1]
        r = inputs['token_type_ids'][0, j, 2]

        if c != 0:
            c = c - 1
            numeric_value_array[r, c] = inputs['numeric_values'][0, j]
            numeric_scale_array[r, c] = inputs['numeric_values_scale'][0, j]

    query_tokens = tokenizer.tokenize(question)

    tokens = ['<s>']
    tokens.extend(query_tokens)
    tokens.append('</s>')

    rows = [0] * len(tokens)
    cols = [0] * len(tokens)
    values = [numeric_value_array[0, 0]] * len(tokens)
    scales = [1.0] * len(tokens)
    segments = [0] * len(tokens)

    for r, tr in enumerate(table_2d):
        for c, td in enumerate(tr):
            cell_tokens = tokenizer.tokenize(td)
            cell_tokens.append('</s>')

            for token in cell_tokens:
                tokens.append(token)
                rows.append(r)
                cols.append(c + 1)
                values.append(numeric_value_array[r, c])
                if numeric_scale_array[r, c] == 0:
                    scales.append(1.0)
                else:
                    scales.append(numeric_scale_array[r, c])
                segments.append(1)

    if (0 in scales) is True:
        print(scales)


    ids = tokenizer.convert_tokens_to_ids(tokens)
    length = min(len(ids), max_length)

    for j in range(length):
        input_ids[count, j] = ids[j]
        attention_mask[count, j] = 1
        token_type_ids[count, j, 0] = segments[j]
        token_type_ids[count, j, 1] = cols[j]
        token_type_ids[count, j, 2] = rows[j]
        numeric_values[count, j] = values[j]
        numeric_scales[count, j] = scales[j]
    answer_float[count] = float(answer_text)
    count += 1

    # for tr in table_2d:
    #     print(tr)
    # for r in range(len(table_2d)):
    #     statement = ''
    #     for c in range(len(table_2d[r])):
    #         statement += str(numeric_value_array[r, c]) + '|' + str(numeric_scale_array[r, c]) + ', '
    #     print(statement)
    # print('---------------------------------')

np.save('WTQ_inputs/input_ids', input_ids[:count])
np.save('WTQ_inputs/attention_mask', attention_mask[:count])
np.save('WTQ_inputs/token_type_ids', token_type_ids[:count])
np.save('WTQ_inputs/numeric_values', numeric_values[:count])
np.save('WTQ_inputs/numeric_scales', numeric_scales[:count])
np.save('WTQ_inputs/answer_float', answer_float[:count])
