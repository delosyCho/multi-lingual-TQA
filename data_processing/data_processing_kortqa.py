import numpy as np

from transformers import AutoTokenizer, TapasTokenizer

max_length = 512

file_name = 'kor_tqa/spec_dataset'
dataset_name = 'spec'

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
tokenizer_tapas = TapasTokenizer.from_pretrained('google/tapas-base')

file = open(file_name, 'r', encoding='utf-8')
lines = file.read().split('\n\t\n')
lines.pop(-1)
file.close()

input_ids = np.zeros(shape=[len(lines), max_length], dtype=np.int32)
attention_mask = np.zeros(shape=[len(lines), max_length], dtype=np.int32)
token_type_ids = np.zeros(shape=[len(lines), max_length, 7], dtype=np.int32)
label_token = np.full(shape=[len(lines), max_length], dtype=np.int32, fill_value=-1)
label_row = np.full(shape=[len(lines)], dtype=np.int32, fill_value=-1)
label_col = np.full(shape=[len(lines)], dtype=np.int32, fill_value=0)
answer_texts = np.zeros(shape=[len(lines)], dtype='<U200')
table_texts = np.zeros(shape=[len(lines)], dtype='<U1000')

count = 0

for i in range(len(lines)):
    tks = lines[i].split('\t')

    question = tks[0]
    answer = tks[1]
    answer_row = int(tks[2])
    answer_col = int(tks[3])
    table_text = tks[4]

    table_lines = table_text.split('[tr]')
    table_2d = []
    for tr in table_lines:
        table_line = tr.split('[td]')
        table_2d.append(table_line)

    num_col = len(table_2d[0])
    num_row = len(table_2d)
    original_num_row = len(table_2d)

    answer_str = str(answer)

    is_number_case = True
    whole_count_case = False

    try:
        answer_value = float(answer)
        if answer_value + 1 == original_num_row:
            if str(question).find('total') == -1 and str(question).find('table') == -1:
                whole_count_case = True
            else:
                continue
        else:
            continue
    except:
        is_number_case = False

    # print(question)
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

    ids = tokenizer.convert_tokens_to_ids(tokens)
    length = min(len(ids), max_length)

    # print(questions[i], ids[:20], tokenizer.convert_ids_to_tokens(ids[:20]))
    # print([0 if x != 2 else 1 for x in ids[0:160]])
    # print('-------------')

    # if np.sum(labels[:length]) == 0:
    #     print('pass')
    #     continue
    # print(selected_answer_positions)
    label_col[count] = answer_col + 1
    label_row[count] = answer_row

    for j in range(length):
        input_ids[count, j] = ids[j]
        attention_mask[count, j] = 1
        token_type_ids[count, j, 0] = segments[j]
        token_type_ids[count, j, 1] = cols[j]
        token_type_ids[count, j, 2] = rows[j]

        if ids[j] > 2:
            if cols[j] == answer_col + 1 and rows[j] == answer_row:
                label_token[count, j] = 1
            else:
                label_token[count, j] = 0
    answer_texts[count] = table_2d[answer_row][answer_col]
    table_texts[count] = table_text
    # for tr in table_2d:
    #     print(tr)
    # print(question)
    # print(answer_texts[count])
    # print()
    count += 1
print(count)

np.save(f'xlm_roberta_kortqa/input_ids_{dataset_name}', input_ids[:count])
np.save(f'xlm_roberta_kortqa/attention_mask_{dataset_name}', attention_mask[:count])
np.save(f'xlm_roberta_kortqa/token_type_ids_{dataset_name}', token_type_ids[:count])
np.save(f'xlm_roberta_kortqa/labels_col_{dataset_name}', label_col[:count])
np.save(f'xlm_roberta_kortqa/labels_row_{dataset_name}', label_row[:count])
np.save(f'xlm_roberta_kortqa/labels_token_{dataset_name}', label_token[:count])
np.save(f'xlm_roberta_kortqa/answer_texts_{dataset_name}', answer_texts[:count])
np.save(f'xlm_roberta_kortqa/answer_texts_{dataset_name}', table_texts[:count])
