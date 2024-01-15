import random
import torch
import numpy as np
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("roberta-large")


class Dataholder:
    def __init__(self, model_name='roberta'):
        self.input_ids = np.load('xlm_roberta_tapas/input_ids_lm.npy')
        self.attention_mask = np.load('xlm_roberta_tapas/attention_mask_lm.npy')
        self.token_type_ids = np.load('xlm_roberta_tapas/token_type_ids_lm.npy')
        self.labels = np.load('xlm_roberta_tapas/label_ids_lm.npy')

        self.b_ix = 0
        self.b_ix2 = 0

        self.r_ix = np.array(
            range(self.input_ids.shape[0]), dtype=np.int32
        )
        np.random.shuffle(self.r_ix)

        self.r_ix2 = np.array(
            range(self.input_ids.shape[0]), dtype=np.int32
        )
        np.random.shuffle(self.r_ix2)
        self.batch_size = 16

    def next_batch(self):
        indexes = []

        length = self.input_ids.shape[1]
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        attention_mask = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        token_type_ids = np.zeros(shape=[self.batch_size, length, 7], dtype=np.int32)
        labels = np.zeros(shape=[self.batch_size, self.labels.shape[1]], dtype=np.int32)

        # self.batch_size // 2
        for i in range(0, self.batch_size // 2):
            if self.b_ix + self.batch_size >= self.input_ids.shape[0]:
                self.b_ix = 0
                np.random.shuffle(self.r_ix)

            ix = self.r_ix[self.b_ix]
            self.b_ix += 1

            indexes.append(ix)

            input_ids[i] = self.input_ids[ix]
            attention_mask[i] = self.attention_mask[ix]
            token_type_ids[i] = self.token_type_ids[ix]
            labels[i] = self.labels[ix]
        # print(np.max(input_ids), np.min(input_ids), np.max(attention_mask), np.min(attention_mask), np.max(token_type_ids), np.min(token_type_ids), np.max(labels), np.min(labels))

        return torch.tensor(np.array(input_ids), dtype=torch.long), \
               torch.tensor(np.array(attention_mask), dtype=torch.long), \
               torch.tensor(np.array(token_type_ids), dtype=torch.long), \
               torch.tensor(np.array(labels), dtype=torch.long)