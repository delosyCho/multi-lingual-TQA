import torch
import torch.nn as nn

import math

from modeling_tapas import TapasForQuestionAnswering

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


class ModelForQuestionAnswering(nn.Module):
    def __init__(self):
        super(ModelForQuestionAnswering, self).__init__()
        config = TapasConfig.from_pretrained('google/tapas-base-finetuned-wtq')

        self.qa_model = TapasForQuestionAnswering(config=config)

        self.row_hidden_layer = ProjectionLayer(hidden_size=config.hidden_size)
        self.col_hidden_layer = ProjectionLayer(hidden_size=config.hidden_size)

        self.encoder = AutoModel.from_pretrained('roberta-base')
        self.classifier_token = nn.Linear(config.hidden_size * 3, 2)

        self.hidden_column = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden_row = nn.Linear(config.hidden_size, config.hidden_size)

        self.classifier_column = nn.Linear(config.hidden_size, 1)

        # set position embedding
        # 3. 원래 포지션 임베딩 가져오기
        original_position_embeddings = self.encoder.embeddings.position_embeddings.weight.detach().clone()

        # 원래 포지션 임베딩의 크기 (예: 512) 확인
        original_max_position, embedding_size = original_position_embeddings.size()

        # 4. 새로운 포지션 임베딩 설정 (예: 1024)
        new_max_position = 1024
        new_position_embeddings = torch.nn.Embedding(new_max_position, embedding_size)
        # 원래의 임베딩 값을 새로운 임베딩에 복사
        with torch.no_grad():
            new_position_embeddings.weight[:original_max_position] = original_position_embeddings.clone()

        # 5. 모델에 새로운 포지션 임베딩 설정
        self.encoder.embeddings.position_embeddings = new_position_embeddings

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                row_mask=None,
                labels=None,
                labels_column=None,
                labels_row=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=torch.zeros_like(input_ids)
        )
        loss_fct = CrossEntropyLoss(ignore_index=-1)

        column_ids = token_type_ids[:, :, 1]
        row_ids = token_type_ids[:, :, 2]

        sequence_output = outputs.last_hidden_state

        token_mask = torch.where(input_ids < 10, torch.tensor(0), torch.tensor(1))
        token_mask = torch.unsqueeze(token_mask, dim=-1)

        # col
        one_hot_to = one_hot(column_ids, num_classes=256).type(torch.float32) # [B, S, C]
        one_hot_from = torch.transpose(one_hot_to, 1, 2) # [B, C, S]

        col_hidden = torch.matmul(one_hot_from, sequence_output * token_mask)
        col_hidden = gelu(self.hidden_column(col_hidden))
        prediction_column = self.classifier_column(col_hidden).squeeze(-1).contiguous()

        # row
        one_hot_to = one_hot(row_ids, num_classes=256).type(torch.float32)  # [B, S, C]
        one_hot_from = torch.transpose(one_hot_to, 1, 2)  # [B, C, S]

        row_hidden = torch.matmul(one_hot_from, sequence_output * token_mask)
        row_hidden = gelu(self.hidden_row(row_hidden))
        prediction_row = self.classifier_column(row_hidden).squeeze(-1).contiguous()

        dist_per_row = torch.distributions.Bernoulli(logits=prediction_row)

        if labels is None:
            return prediction_column, dist_per_row.probs

        weight = torch.where(
            labels == 0,
            torch.ones_like(labels, dtype=torch.float32),
            10.0 * torch.ones_like(labels, dtype=torch.float32),
        )

        EPSILON_ZERO_DIVISION = 1e-10
        input_mask_float = row_mask.float()
        selection_loss_per_token = -dist_per_row.log_prob(labels) * weight
        selection_loss_per_example = torch.sum(selection_loss_per_token * input_mask_float, dim=1) / (
                torch.sum(input_mask_float, dim=1) + EPSILON_ZERO_DIVISION
        )
        # print(torch.sum(torch.sum(selection_loss_per_token * input_mask_float, dim=1) ))

        loss_selection = torch.mean(selection_loss_per_example)
        loss_row = loss_fct(prediction_row, labels_row)
        loss_column = loss_fct(prediction_column, labels_column)

        return loss_column, loss_row, loss_selection, loss_column + loss_row + loss_selection









