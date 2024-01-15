from transformers import TapasTokenizer, AutoTokenizer

import numpy as np

import pandas as pd
from datasets import load_dataset


def is_num(str_word):
    try:
        float(str_word)
        return True
    except:
        return False


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def strip_word(word: str):
    return word.replace('\\\\n', ' ').replace('\\xa0', ' ')


if "__main__" == __name__:
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

    count = 0
    count2 = 0
    count2_ = 0
    count3 = 0
    count_ = 0

    max_length = 512
    answer_length = 64

    input_ids = np.zeros(shape=[len(questions), max_length], dtype=np.int32)
    attention_mask = np.zeros(shape=[len(questions), max_length], dtype=np.int32)
    token_type_ids = np.zeros(shape=[len(questions), max_length, 7], dtype=np.int32)
    labels = np.full(shape=[len(questions), 256], dtype=np.int32, fill_value=-1)
    label_token = np.full(shape=[len(questions), max_length], dtype=np.int32, fill_value=-1)
    label_row = np.full(shape=[len(questions)], dtype=np.int32, fill_value=-1)
    label_col = np.full(shape=[len(questions)], dtype=np.int32, fill_value=0)
    agg_labels = np.full(shape=[len(questions)], dtype=np.int32, fill_value=0)

    for i in range(len(questions)):
        table_dict = tables[i]
        table_2d = []
        table_2d.append(table_dict['header'])
        table_2d.extend(table_dict['rows'])

        num_col = len(table_2d[0])
        num_row = len(table_2d)
        original_num_row = len(table_2d)

        question = str(questions[i])
        answers = answer_lists[i]
        answer = answers[0]
        for a in range(1, len(answers)):
            answer += '|' + answers[a]
        answer_str = str(answer)

        is_number_case = True
        whole_count_case = False

        try:
            answer_value = float(answer)
            if answer_value + 1 == original_num_row:
                if str(question).find('total') == -1 and str(question).find('table') == -1:
                    whole_count_case = True
                else:
                    count3 += 1
                    continue
            else:
                count3 += 1
                continue
        except:
            is_number_case = False

        supervision = True
        for word in answers:
            is_exist = False
            for tl in table_2d:
                for td in tl:
                    if str(td).lower() == str(word).lower():
                        is_exist = True

            if is_exist is False:
                supervision = False

        # if whole_count_case is True:
        #    continue

        if is_number_case is True:
            continue

        if supervision is False:
            count2 += 1
            continue

        is_okay = True
        selected_answer_positions = []
        for answer_text in answers:
            check_count = 0
            is_check = False
            for r, tl in enumerate(table_2d):
                for c, td in enumerate(tl):
                    if str(td).lower() == str(answer_text).lower():
                        is_check = True
                        selected_answer_positions.append((r, c))
                        check_count += 1
            # print(count)
            if check_count > 1:
                is_okay = False

            if is_check is False:
                is_okay = False

        if is_okay is False and whole_count_case is False:
            count2_ += 1
            continue

        if whole_count_case is True:
            answer = 'col: ' + 'num row' + ' , row: '
            for j in range(1, len(table_2d)):
                answer += str(j)
                if j < len(table_2d) - 1:
                    answer += ', '
            answer += 'agg: count'
        else:
            try:
                answer = 'col: ' + table_2d[0][selected_answer_positions[0][1]] + ' , row: '
                for position in selected_answer_positions:
                    answer += '' + str(position[0]) + ', '
                answer += 'agg: se'
            except:
                continue

        answer_rows = []
        for selected_answer_position in selected_answer_positions:
            answer_rows.append(selected_answer_position[0])

        if max(answer_rows) > 256:
            continue

        row_ids = answer_rows
        labels[count, :len(table_2d)] = 0
        for answer_row in answer_rows:
            labels[count, answer_row] = 1

        if len(answer_rows) == 1:
            label_row[count] = answer_rows[0]
            agg_labels[count] = 0
        else:
            agg_labels[count] = 1

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
        label_col[count] = selected_answer_positions[0][1] + 1

        for j in range(length):
            input_ids[count, j] = ids[j]
            attention_mask[count, j] = 1
            token_type_ids[count, j, 0] = segments[j]
            token_type_ids[count, j, 1] = cols[j]
            token_type_ids[count, j, 2] = rows[j]

            if ids[j] > 2:
                if cols[j] == selected_answer_positions[0][1] + 1 and (rows[j] in row_ids) is True:
                    label_token[count, j] = 1
                else:
                    label_token[count, j] = 0
        # print(token_type_ids[count, :, 2])
        count += 1
    print(count)

    np.save('WTQ_inputs/input_ids_sel', input_ids[:count])
    np.save('WTQ_inputs/attention_mask_sel', attention_mask[:count])
    np.save('WTQ_inputs/token_type_ids_sel', token_type_ids[:count])
    np.save('WTQ_inputs/labels', labels[:count])
    np.save('WTQ_inputs/labels_col', label_col[:count])
    np.save('WTQ_inputs/labels_row', label_row[:count])
    np.save('WTQ_inputs/labels_token', label_token[:count])
    np.save('WTQ_inputs/agg_labels', agg_labels[:count])

    count = 0

    # test dataset
    dataset_test = load_dataset("wikitablequestions")['test']
    tables = dataset_test['table']
    questions = dataset_test['question']
    answer_lists = dataset_test['answers']
    id_list = dataset_test['id']

    input_ids = np.zeros(shape=[len(questions), max_length], dtype=np.int32)
    attention_mask = np.zeros(shape=[len(questions), max_length], dtype=np.int32)
    token_type_ids = np.zeros(shape=[len(questions), max_length, 7], dtype=np.int32)
    answer_texts = np.zeros(shape=[len(questions)], dtype='<U100')
    id_texts = np.zeros(shape=[len(questions)], dtype='<U100')

    for i in range(len(questions)):
        table_dict = tables[i]
        table_2d = []
        table_2d.append(table_dict['header'])
        table_2d.extend(table_dict['rows'])

        num_col = len(table_2d[0])
        num_row = len(table_2d)
        original_num_row = len(table_2d)

        if num_row > 300:
            continue

        question = str(questions[i])
        answers = answer_lists[i]
        answer = answers[0]
        for a in range(1, len(answers)):
            answer += '|' + answers[a]
        answer_str = str(answer)

        query_tokens = tokenizer.tokenize(question)

        tokens = ['<s>']
        tokens.extend(query_tokens)
        tokens.append('</s>')

        rows = [0] * len(tokens)
        cols = [0] * len(tokens)
        labels = [0] * len(tokens)
        segments = [0] * len(tokens)

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

        for j in range(length):
            input_ids[count, j] = ids[j]
            attention_mask[count, j] = 1
            token_type_ids[count, j, 0] = segments[j]
            token_type_ids[count, j, 1] = cols[j]
            token_type_ids[count, j, 2] = rows[j]
        answer_texts[count] = answer_str
        id_texts[i] = id_list[i]
        count += 1
    print(count)
    np.save('WTQ_inputs/input_ids_sel_dev', input_ids[:count])
    np.save('WTQ_inputs/attention_mask_sel_dev', attention_mask[:count])
    np.save('WTQ_inputs/token_type_ids_sel_dev', token_type_ids[:count])
    np.save('WTQ_inputs/answer_texts', answer_texts[:count])
    np.save('WTQ_inputs/id_texts', id_texts[:count])

    # test dataset
    count= 0
    dataset_test = load_dataset("wikitablequestions")['validation']
    tables = dataset_test['table']
    questions = dataset_test['question']
    answer_lists = dataset_test['answers']
    id_list = dataset_test['id']

    input_ids = np.zeros(shape=[len(questions), max_length], dtype=np.int32)
    attention_mask = np.zeros(shape=[len(questions), max_length], dtype=np.int32)
    token_type_ids = np.zeros(shape=[len(questions), max_length, 7], dtype=np.int32)
    answer_texts = np.zeros(shape=[len(questions)], dtype='<U100')
    id_texts = np.zeros(shape=[len(questions)], dtype='<U100')

    for i in range(len(questions)):
        table_dict = tables[i]
        table_2d = []
        table_2d.append(table_dict['header'])
        table_2d.extend(table_dict['rows'])

        num_col = len(table_2d[0])
        num_row = len(table_2d)
        original_num_row = len(table_2d)

        if num_row > 300:
            continue

        question = str(questions[i])
        answers = answer_lists[i]
        answer = answers[0]
        for a in range(1, len(answers)):
            answer += '|' + answers[a]
        answer_str = str(answer)

        query_tokens = tokenizer.tokenize(question)

        tokens = ['<s>']
        tokens.extend(query_tokens)
        tokens.append('</s>')

        rows = [0] * len(tokens)
        cols = [0] * len(tokens)
        labels = [0] * len(tokens)
        segments = [0] * len(tokens)

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

        for j in range(length):
            input_ids[count, j] = ids[j]
            attention_mask[count, j] = 1
            token_type_ids[count, j, 0] = segments[j]
            token_type_ids[count, j, 1] = cols[j]
            token_type_ids[count, j, 2] = rows[j]
        answer_texts[count] = answer_str
        id_texts[i] = id_list[i]
        count += 1
    print(count)
    np.save('WTQ_inputs/input_ids_sel_dev2', input_ids[:count])
    np.save('WTQ_inputs/attention_mask_sel_dev2', attention_mask[:count])
    np.save('WTQ_inputs/token_type_ids_sel_dev2', token_type_ids[:count])
    np.save('WTQ_inputs/answer_texts2', answer_texts[:count])
    np.save('WTQ_inputs/id_texts2', id_texts[:count])