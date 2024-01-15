import random
import torch
import numpy as np
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("roberta-large")


class Dataholder:
    def __init__(self, model_name='roberta'):
        path = 'WTQ_inputs/'
        self.input_ids = np.load(path + 'input_ids.npy')
        self.attention_mask = np.load(path + 'attention_mask.npy')
        self.token_type_ids = np.load(path + 'token_type_ids.npy')
        self.input_value = np.load(path + 'numeric_values.npy')
        self.input_scale = np.load(path + 'numeric_scales.npy')
        self.answer_float = np.load(path + 'answer_float.npy')

        self.input_ids_sel = np.load(path + 'input_ids_sel.npy')
        self.attention_mask_sel = np.load(path + 'attention_mask_sel.npy')
        self.token_type_ids_sel = np.load(path + 'token_type_ids_sel.npy')
        # for i in range(100):
        #     print(self.token_type_ids_sel[i, :, 2])
        # input()
        self.labels = np.load(path + 'labels.npy')
        self.labels_column = np.load(path + 'labels_col.npy')
        self.labels_row = np.load(path + 'labels_row.npy')
        self.labels_agg = np.load(path + 'agg_labels.npy')
        self.labels_token = np.load(path + 'labels_token.npy')

        self.input_ids_rl = np.load(path + 'input_ids_rl.npy')
        self.attention_mask_rl = np.load(path + 'attention_mask_rl.npy')
        self.token_type_ids_rl = np.load(path + 'token_type_ids_rl.npy')
        self.labels_rl = np.load(path + 'labels_rl.npy')
        self.labels_column_rl = np.load(path + 'labels_col_rl.npy')
        self.labels_row_rl = np.load(path + 'labels_row_rl.npy')
        self.labels_agg_rl = np.load(path + 'agg_labels_rl.npy')
        self.labels_token_rl = np.load(path + 'labels_token_rl.npy')
        self.answer_texts_rl = np.load(path + 'answer_texts_rl.npy')
        self.id_texts_rl = np.load(path + 'id_texts_rl.npy')

        self.input_ids_t = np.load(path + 'input_ids_sel_dev.npy')
        self.attention_mask_t = np.load(path + 'attention_mask_sel_dev.npy')
        self.token_type_ids_t = np.load(path + 'token_type_ids_sel_dev.npy')
        self.answer_texts = np.load(path + 'answer_texts.npy')
        self.id_texts = np.load(path + 'id_texts.npy')

        self.input_ids_t2 = np.load(path + 'input_ids_sel_dev2.npy')
        self.attention_mask_t2 = np.load(path + 'attention_mask_sel_dev2.npy')
        self.token_type_ids_t2 = np.load(path + 'token_type_ids_sel_dev2.npy')
        self.answer_texts2 = np.load(path + 'answer_texts2.npy')
        self.id_texts2 = np.load(path + 'id_texts2.npy')

        self.b_ix = 0
        self.b_ix2 = 0

        self.r_ix = np.array(
            range(self.input_ids_rl.shape[0]), dtype=np.int32
        )
        np.random.shuffle(self.r_ix)

        self.r_ix2 = np.array(
            range(self.input_ids_sel.shape[0]), dtype=np.int32
        )
        np.random.shuffle(self.r_ix2)
        self.batch_size = 16

    def next_batch(self):
        length = self.input_ids_rl.shape[1]
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        attention_mask = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        position_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        token_type_ids = np.zeros(shape=[self.batch_size, length, 7], dtype=np.int32)
        row_mask = np.zeros(shape=[self.batch_size, self.labels.shape[1]], dtype=np.int32)

        labels_token = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        labels = np.zeros(shape=[self.batch_size, self.labels.shape[1]], dtype=np.int32)
        labels_column = np.zeros(shape=[self.batch_size], dtype=np.int32)
        labels_row = np.zeros(shape=[self.batch_size], dtype=np.int32)
        labels_agg = np.zeros(shape=[self.batch_size], dtype=np.int32)

        # self.batch_size // 2
        for i in range(0, self.batch_size // 2):
            if self.b_ix + self.batch_size >= self.input_ids_rl.shape[0]:
                self.b_ix = 0
                np.random.shuffle(self.r_ix)

            ix = self.r_ix[self.b_ix]
            self.b_ix += 1

            input_ids[i] = self.input_ids_rl[ix]
            attention_mask[i] = self.attention_mask_rl[ix]
            position_ids[i] = range(length)
            token_type_ids[i] = self.token_type_ids_rl[ix]

            labels[i] = self.labels_rl[ix]
            labels_column[i] = self.labels_column_rl[ix]
            labels_row[i] = self.labels_row_rl[ix]
            labels_token[i] = self.labels_token_rl[ix]
            labels_agg[i] = self.labels_agg_rl[ix]

            max_row = np.max(token_type_ids[i, :, 2])
            if labels_row[i] >= max_row:
                labels_row[i] = -1

            for r in range(max_row):
                row_mask[i, r] = 1

        # self.batch_size // 2
        for i in range(self.batch_size // 2, self.batch_size):
            if self.b_ix2 + self.batch_size >= self.input_ids_sel.shape[0]:
                self.b_ix2 = 0
                np.random.shuffle(self.r_ix2)

            ix = self.r_ix2[self.b_ix2]
            self.b_ix2 += 1

            input_ids[i] = self.input_ids_sel[ix]
            attention_mask[i] = self.attention_mask_sel[ix]
            position_ids[i] = range(length)
            token_type_ids[i] = self.token_type_ids_sel[ix]

            labels[i] = self.labels[ix]
            labels_column[i] = self.labels_column[ix]
            labels_row[i] = self.labels_row[ix]
            labels_token[i] = self.labels_token[ix]
            labels_agg[i] = self.labels_agg[ix]
            # labels_row[i] = -1
            max_row = np.max(token_type_ids[i, :, 2])
            for r in range(max_row):
                row_mask[i, r] = 1
            # print(token_type_ids[i, :, 0])
        # print(labels_row[0], labels[0])
        # print(labels_row)
        # print(labels_column)
        # print(labels_row)
        # print('---')
        return torch.tensor(np.array(input_ids), dtype=torch.long), \
               torch.tensor(np.array(attention_mask), dtype=torch.long), \
               torch.tensor(np.array(row_mask), dtype=torch.long), \
               torch.tensor(np.array(position_ids), dtype=torch.long), \
               torch.tensor(np.array(token_type_ids), dtype=torch.long), \
               torch.tensor(np.array(labels), dtype=torch.long), \
               torch.tensor(np.array(labels_column), dtype=torch.long), \
               torch.tensor(np.array(labels_row), dtype=torch.long), \
               torch.tensor(np.array(labels_token), dtype=torch.long), \
               torch.tensor(np.array(labels_agg), dtype=torch.long)

    def next_batch_sel(self):
        length = self.input_ids.shape[1]
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        attention_mask = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        row_mask = np.zeros(shape=[self.batch_size, self.labels.shape[1]], dtype=np.int32)
        position_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        token_type_ids = np.zeros(shape=[self.batch_size, length, 7], dtype=np.int32)
        labels = np.zeros(shape=[self.batch_size, self.labels.shape[1]], dtype=np.int32)
        labels_column = np.zeros(shape=[self.batch_size], dtype=np.int32)
        labels_row = np.zeros(shape=[self.batch_size], dtype=np.int32)

        for i in range(self.batch_size):
            if self.b_ix2 + self.batch_size >= self.input_ids_sel.shape[0]:
                self.b_ix2 = 0
                np.random.shuffle(self.r_ix2)

            ix = self.r_ix2[self.b_ix2]
            self.b_ix2 += 1

            if self.labels_column[ix] > 256:
                continue

            input_ids[i] = self.input_ids_sel[ix]
            attention_mask[i] = self.attention_mask_sel[ix]
            position_ids[i] = range(length)
            token_type_ids[i] = self.token_type_ids_sel[ix]
            labels[i] = self.labels[ix]
            labels_column[i] = self.labels_column[ix]
            labels_row[i] = self.labels_row[ix]

            max_row = np.max(token_type_ids[i, :, 2])
            for r in range(max_row):
                row_mask[i, r] = 1

        # print(labels[i, :512])
        return torch.tensor(np.array(input_ids), dtype=torch.long), \
               torch.tensor(np.array(attention_mask), dtype=torch.long), \
               torch.tensor(np.array(row_mask), dtype=torch.long), \
               torch.tensor(np.array(position_ids), dtype=torch.long), \
               torch.tensor(np.array(token_type_ids), dtype=torch.long), \
               torch.tensor(np.array(labels), dtype=torch.float), \
               torch.tensor(np.array(labels_column), dtype=torch.long), \
               torch.tensor(np.array(labels_row), dtype=torch.long)

    def next_batch_test(self):
        self.batch_size = 1

        length = self.input_ids_t.shape[1]
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        attention_mask = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        position_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        token_type_ids = np.zeros(shape=[self.batch_size, length, 7], dtype=np.int32)
        answer_texts = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        for i in range(self.batch_size):
            if self.b_ix2 + self.batch_size >= self.input_ids_sel.shape[0]:
                self.b_ix2 = 0
                np.random.shuffle(self.r_ix2)

            ix = self.b_ix

            input_ids[i] = self.input_ids_t[ix]
            attention_mask[i] = self.attention_mask_t[ix]
            position_ids[i] = range(length)
            token_type_ids[i] = self.token_type_ids_t[ix]
            answer_text = self.answer_texts[ix]
            id_text = self.id_texts[ix]
            self.b_ix += 1

        # print(labels[i, :512])
        return torch.tensor(np.array(input_ids), dtype=torch.long), \
               torch.tensor(np.array(attention_mask), dtype=torch.long), \
               torch.tensor(np.array(position_ids), dtype=torch.long), \
               torch.tensor(np.array(token_type_ids), dtype=torch.long), \
               answer_text, id_text

    def next_batch_test2(self):
        self.batch_size = 1

        length = self.input_ids_t.shape[1]
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        attention_mask = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        position_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        token_type_ids = np.zeros(shape=[self.batch_size, length, 7], dtype=np.int32)
        answer_texts = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        for i in range(self.batch_size):
            if self.b_ix2 + self.batch_size >= self.input_ids_sel.shape[0]:
                self.b_ix2 = 0
                np.random.shuffle(self.r_ix2)

            ix = self.b_ix

            input_ids[i] = self.input_ids_rl[ix]
            attention_mask[i] = self.attention_mask_rl[ix]
            position_ids[i] = range(length)
            token_type_ids[i] = self.token_type_ids_rl[ix]
            answer_text = self.answer_texts_rl[ix]
            id_text = self.id_texts_rl[ix]
            self.b_ix += 1

        # print(labels[i, :512])
        return torch.tensor(np.array(input_ids), dtype=torch.long), \
               torch.tensor(np.array(attention_mask), dtype=torch.long), \
               torch.tensor(np.array(position_ids), dtype=torch.long), \
               torch.tensor(np.array(token_type_ids), dtype=torch.long), \
               answer_text, id_text