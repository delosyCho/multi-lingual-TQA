import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


dataset_dev = load_dataset("wikitablequestions")['validation']
tables_ = dataset_dev['table']
questions_ = dataset_dev['question']
answer_lists_ = dataset_dev['answers']
id_lists_ = dataset_dev['id']

dataset = load_dataset("wikitablequestions")['train']
tables = dataset['table']
questions = dataset['question']
answer_lists = dataset['answers']
id_lists = dataset['id']

questions.extend(questions_)
tables.extend(tables_)
answer_lists.extend(answer_lists_)
id_lists.extend(id_lists_)

count = 0
max_length = 512
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
input_ids = np.zeros(shape=[len(questions), max_length], dtype=np.int32)
attention_mask = np.zeros(shape=[len(questions), max_length], dtype=np.int32)
token_type_ids = np.zeros(shape=[len(questions), max_length, 7], dtype=np.int32)
label_row = np.full(shape=[len(questions)], dtype=np.int32, fill_value=-1)
label_col = np.full(shape=[len(questions)], dtype=np.int32, fill_value=0)
label_token = np.full(shape=[len(questions), max_length], dtype=np.int32, fill_value=-1)
labels = np.full(shape=[len(questions), 256], dtype=np.int32, fill_value=-1)
agg_labels = np.full(shape=[len(questions)], dtype=np.int32, fill_value=0)

answer_texts = np.zeros(shape=[len(questions)], dtype='<U100')
id_texts = np.zeros(shape=[len(questions)], dtype='<U100')

id_to_index = {}
for i, id_str in enumerate(id_lists):
    id_to_index[id_str] = i

pseudo_label = np.load('mypseudo/pseudo_labels_sel_3.npy')
id_array = np.load('mypseudo/id_string_selection.npy')
print(pseudo_label.shape, id_array.shape)

for i in range(id_array.shape[0]):
    ix = id_to_index[id_array[i]]

    table_dict = tables[ix]
    table_2d = []
    table_2d.append(table_dict['header'])
    table_2d.extend(table_dict['rows'])

    num_col = len(table_2d[0])
    num_row = len(table_2d)
    original_num_row = len(table_2d)

    question = str(questions[ix])
    query_tokens = tokenizer.tokenize(question)

    tokens = ['<s>']
    tokens.extend(query_tokens)
    tokens.append('</s>')

    rows = [0] * len(tokens)
    cols = [0] * len(tokens)
    # labels = [0] * len(tokens)
    segments = [0] * len(tokens)

    check_tokens = []

    for r, tr in enumerate(table_2d):
        row_statement = 'row ' + str(r) + ' :'
        statement_tokens = tokenizer.tokenize(row_statement)

        for t, token in enumerate(statement_tokens):
            tokens.append(token)
            rows.append(r)
            cols.append(0)
            segments.append(1)

        for c, td in enumerate(tr):
            cell_tokens = tokenizer.tokenize(td)
            cell_tokens.append('</s>')

            for token in cell_tokens:
                tokens.append(token)
                rows.append(r)
                cols.append(c + 1)
                segments.append(1)

    # print(check_tokens, answer_str, '\n')
    # input()
    ids = tokenizer.convert_tokens_to_ids(tokens)
    length = min(len(ids), max_length)

    label_str = str(pseudo_label[i, 0]).replace('\'', '').replace('[', '').replace(']', '')

    try:
        col_ids = int(label_str.split(' row: ')[0].replace('col: ', ''))
    except:
        continue

    row_ids = []
    tks = label_str.split(' row: ')[1].split(' ')
    for tk in tks:
        try:
            val = int(tk)
            row_ids.append(val)
        except:
            None
    print('len:', len(row_ids), answer_lists[ix])
    if len(row_ids) == 0:
        continue

    if len(row_ids) == 1:
        label_row[count] = row_ids[0]
    label_col[count] = col_ids + 1

    if len(row_ids) == 1:
        agg_labels[count] = 0
    else:
        agg_labels[count] = 1

    labels[count, :len(table_2d)] = 0
    for row_id in row_ids:
        labels[count, row_id] = 1
    print(answer_lists[ix], table_2d[row_ids[0]][col_ids])

    answer_texts[count] = answer_lists[ix][0]
    id_texts[count] = id_lists[ix]

    for j in range(length):
        input_ids[count, j] = ids[j]
        attention_mask[count, j] = 1
        token_type_ids[count, j, 0] = segments[j]
        token_type_ids[count, j, 1] = cols[j]
        token_type_ids[count, j, 2] = rows[j]

        if ids[j] > 2:
            if cols[j] == col_ids + 1 and (rows[j] in row_ids) is True:
                label_token[count, j] = 1
            else:
                label_token[count, j] = 0
    count += 1
print(count)

print(count)
# count = 1486

# np.save('WTQ_inputs/input_ids_rl', input_ids[:count])
# np.save('WTQ_inputs/attention_mask_rl', attention_mask[:count])
# np.save('WTQ_inputs/token_type_ids_rl', token_type_ids[:count])
# np.save('WTQ_inputs/labels_rl', labels[:count])
# np.save('WTQ_inputs/labels_col_rl', label_col[:count])
# np.save('WTQ_inputs/labels_row_rl', label_row[:count])
# np.save('WTQ_inputs/agg_labels_rl', agg_labels[:count])
# np.save('WTQ_inputs/answer_texts_rl', answer_texts[:count])
# np.save('WTQ_inputs/id_texts_rl', id_texts[:count])
# exit(100)

pseudo_label = np.load('mypseudo/pseudo_labels_count_3.npy')
id_array = np.load('mypseudo/id_string_rl.npy')
print(pseudo_label.shape, id_array.shape)

for i in range(id_array.shape[0]):
    ix = id_to_index[id_array[i]]

    table_dict = tables[ix]
    table_2d = []
    table_2d.append(table_dict['header'])
    table_2d.extend(table_dict['rows'])

    num_col = len(table_2d[0])
    num_row = len(table_2d)
    original_num_row = len(table_2d)

    question = str(questions[ix])
    query_tokens = tokenizer.tokenize(question)

    tokens = ['<s>']
    tokens.extend(query_tokens)
    tokens.append('</s>')

    rows = [0] * len(tokens)
    cols = [0] * len(tokens)
    # labels = [0] * len(tokens)
    segments = [0] * len(tokens)

    check_tokens = []

    for r, tr in enumerate(table_2d):
        row_statement = 'row ' + str(r) + ' :'
        statement_tokens = tokenizer.tokenize(row_statement)

        for t, token in enumerate(statement_tokens):
            tokens.append(token)
            rows.append(r)
            cols.append(0)
            segments.append(1)

        for c, td in enumerate(tr):
            cell_tokens = tokenizer.tokenize(td)
            cell_tokens.append('</s>')

            for token in cell_tokens:
                tokens.append(token)
                rows.append(r)
                cols.append(c + 1)
                segments.append(1)

    # print(check_tokens, answer_str, '\n')
    # input()
    ids = tokenizer.convert_tokens_to_ids(tokens)
    length = min(len(ids), max_length)
    # print(pseudo_label[i])
    # input()
    label_str = str(pseudo_label[i][0]).replace('\'', '').replace('[', '').replace(']', '')

    try:
        col_ids = int(label_str.split(' row: ')[0].replace('col: ', ''))
    except:
        continue

    row_ids = []
    tks = label_str.split(' row: ')[1].split(' ')
    for tk in tks:
        try:
            val = int(tk)
            row_ids.append(val)
        except:
            None

    if len(row_ids) == len(table_2d) - 1:
        agg_labels[count] = 3
    else:
        agg_labels[count] = 2

    if len(row_ids) == 1:
        label_row[count] = row_ids[0]
    label_col[count] = -1

    labels[count, :len(table_2d)] = 0

    check = []
    for row_id in row_ids:
        labels[count, row_id] = 1
        check.append(table_2d[row_id][col_ids])

    # for tr in table_2d:
    #     print(tr)
    # print()
    # print(check)
    # print(questions[ix])
    # print('---------------')

    for j in range(length):
        input_ids[count, j] = ids[j]
        attention_mask[count, j] = 1
        token_type_ids[count, j, 0] = segments[j]
        token_type_ids[count, j, 1] = cols[j]
        token_type_ids[count, j, 2] = rows[j]

        if ids[j] > 2:
            if cols[j] == col_ids + 1 and (rows[j] in row_ids) is True:
                label_token[count, j] = 1
            else:
                label_token[count, j] = 0
    count += 1
print(count)


pseudo_label = np.load('mypseudo/pseudo_labels_op_3.npy')
id_array = np.load('mypseudo/id_string_rl_op.npy')
print(pseudo_label.shape, id_array.shape)

for i in range(id_array.shape[0]):
    ix = id_to_index[id_array[i]]

    table_dict = tables[ix]
    table_2d = []
    table_2d.append(table_dict['header'])
    table_2d.extend(table_dict['rows'])

    num_col = len(table_2d[0])
    num_row = len(table_2d)
    original_num_row = len(table_2d)

    question = str(questions[ix])
    query_tokens = tokenizer.tokenize(question)

    tokens = ['<s>']
    tokens.extend(query_tokens)
    tokens.append('</s>')

    rows = [0] * len(tokens)
    cols = [0] * len(tokens)
    # labels = [0] * len(tokens)
    segments = [0] * len(tokens)

    check_tokens = []

    for r, tr in enumerate(table_2d):
        row_statement = 'row ' + str(r) + ' :'
        statement_tokens = tokenizer.tokenize(row_statement)

        for t, token in enumerate(statement_tokens):
            tokens.append(token)
            rows.append(r)
            cols.append(0)
            segments.append(1)

        for c, td in enumerate(tr):
            cell_tokens = tokenizer.tokenize(td)
            cell_tokens.append('</s>')

            for token in cell_tokens:
                tokens.append(token)
                rows.append(r)
                cols.append(c + 1)
                segments.append(1)

    # print(check_tokens, answer_str, '\n')
    # input()
    ids = tokenizer.convert_tokens_to_ids(tokens)
    length = min(len(ids), max_length)

    label_str = str(pseudo_label[i, 0]).replace('\'', '').replace('[', '').replace(']', '')

    try:
        col_ids = int(label_str.split(' row: ')[0].replace('col: ', ''))
    except:
        continue

    row_ids = []
    tks = label_str.split(' row: ')[1].split(' ')
    for tk in tks:
        try:
            val = int(tk)
            row_ids.append(val)
        except:
            None

    if len(row_ids) == 1:
        label_row[count] = row_ids[0]
    label_col[count] = col_ids + 1

    agg_labels[count] = 4

    labels[count, :len(table_2d)] = 0
    for row_id in row_ids:
        labels[count, row_id] = 1

    for j in range(length):
        input_ids[count, j] = ids[j]
        attention_mask[count, j] = 1
        token_type_ids[count, j, 0] = segments[j]
        token_type_ids[count, j, 1] = cols[j]
        token_type_ids[count, j, 2] = rows[j]

        if ids[j] > 2:
            if cols[j] == col_ids + 1 and (rows[j] in row_ids) is True:
                label_token[count, j] = 1
            else:
                label_token[count, j] = 0
    count += 1
print(count)
# count = 1486
np.save('WTQ_inputs/input_ids_rl', input_ids[:count])
np.save('WTQ_inputs/attention_mask_rl', attention_mask[:count])
np.save('WTQ_inputs/token_type_ids_rl', token_type_ids[:count])
np.save('WTQ_inputs/labels_rl', labels[:count])
np.save('WTQ_inputs/labels_col_rl', label_col[:count])
np.save('WTQ_inputs/labels_row_rl', label_row[:count])
np.save('WTQ_inputs/labels_token_rl', label_token[:count])
np.save('WTQ_inputs/agg_labels_rl', agg_labels[:count])