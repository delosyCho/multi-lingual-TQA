import torch
import torch.nn as nn

import math

from transformers import XLMRobertaForMaskedLM

from transformers import TapasConfig, AutoModel
from torch.nn.functional import one_hot
from torch.nn import CrossEntropyLoss


LayerNorm = nn.LayerNorm


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def Linear(i_dim, o_dim, bias=True):
    m = nn.Linear(i_dim, o_dim, bias)
    nn.init.normal_(m.weight, std=0.02)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class Pooler(nn.Module):
    def __init__(self, hidden_size, index: int):
        super().__init__()
        self.index = index
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        index = self.index

        first_token_tensor = hidden_states[:, index]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ProjectionLayer(nn.Module):
    def __init__(self, hidden_size=1024):
        super(ProjectionLayer, self).__init__()
        self.projection_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_sates, token_type_ids, max_length=256, with_cell_shape=False):
        one_hot_to = one_hot(token_type_ids, num_classes=max_length).type(torch.float32)
        one_hot_from = torch.transpose(one_hot_to, 1, 2)

        cell_hidden = torch.matmul(one_hot_from, hidden_sates)
        cell_hidden = gelu(self.projection_linear(cell_hidden))
        if with_cell_shape is True:
            return cell_hidden

        seq_hidden = torch.matmul(one_hot_to, cell_hidden)

        return seq_hidden


class ModelForMaskingLM(nn.Module):
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            print('!!! 1')
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            print('!!! 2')
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            print('!!! 3')
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def __init__(self):
        super(ModelForMaskingLM, self).__init__()
        config = TapasConfig.from_pretrained('google/tapas-base-finetuned-wtq')
        self.encoder = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base')

        hidden_size = 768
        max_vocab_sizes = 512

        self.initializer_range = 0.003
        self.seg_embedding = nn.Embedding(max_vocab_sizes, hidden_size)
        self.seg_embedding.apply(self._init_weights)

        self.col_embedding = nn.Embedding(max_vocab_sizes, hidden_size)
        self.col_embedding.apply(self._init_weights)

        self.row_embedding = nn.Embedding(max_vocab_sizes, hidden_size)
        self.row_embedding.apply(self._init_weights)

        self.rank_embedding = nn.Embedding(max_vocab_sizes, hidden_size)
        self.rank_embedding.apply(self._init_weights)

        self.rank_inv_embedding = nn.Embedding(max_vocab_sizes, hidden_size)
        self.rank_inv_embedding.apply(self._init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None,):
        inputs_embeds = self.encoder.roberta.get_input_embeddings()(input_ids)
        inputs_embeds += self.col_embedding(token_type_ids[:, :, 1])
        inputs_embeds += self.row_embedding(token_type_ids[:, :, 2])
        inputs_embeds += self.rank_embedding(token_type_ids[:, :, 3])
        inputs_embeds += self.rank_inv_embedding(token_type_ids[:, :, 4])

        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(input_ids),
            labels=labels
        )
        total_loss = outputs.loss

        return total_loss








