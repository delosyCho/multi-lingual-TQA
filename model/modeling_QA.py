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
        self.dropout = nn.Dropout(0.4)
        self.qa_model = TapasForQuestionAnswering(config=config)

        self.row_hidden_layer = ProjectionLayer(hidden_size=config.hidden_size)
        self.col_hidden_layer = ProjectionLayer(hidden_size=config.hidden_size)

        self.encoder = AutoModel.from_pretrained('roberta-base')
        self.classifier_token = nn.Linear(config.hidden_size * 3, 2)

        self.hidden_agg = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden_column = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden_row = nn.Linear(config.hidden_size, config.hidden_size)

        self.classifier_column = nn.Linear(config.hidden_size, 1)
        self.classifier_row = nn.Linear(config.hidden_size, 2)

        self.classifier_agg = nn.Linear(config.hidden_size, 6)

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
                labels_row=None,
                labels_agg=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=torch.zeros_like(input_ids)
        )
        pooled_output = outputs.pooler_output

        loss_fct = CrossEntropyLoss(ignore_index=-1)

        column_ids = token_type_ids[:, :, 1]
        row_ids = token_type_ids[:, :, 2]

        sequence_output = self.dropout(outputs.last_hidden_state)

        token_mask = torch.where(input_ids > 2, torch.tensor(1), torch.tensor(0))
        token_mask = torch.unsqueeze(token_mask, dim=-1)

        col_mask = torch.where(input_ids == 2, torch.tensor(1), torch.tensor(0))
        col_mask = torch.unsqueeze(col_mask, dim=-1)

        head_mask = torch.where(row_ids != 0, torch.tensor(0), torch.tensor(1))
        head_mask = torch.unsqueeze(head_mask, dim=-1)

        # agg
        pooled_output = gelu(self.hidden_agg(pooled_output))
        prediction_agg = self.classifier_agg(pooled_output).contiguous()

        # col
        one_hot_to = one_hot(column_ids, num_classes=256).type(torch.float32) # [B, S, C]
        one_hot_from = torch.transpose(one_hot_to, 1, 2) # [B, C, S]
        a = col_mask * head_mask
        # print(a[0, :150, 0])
        col_hidden = torch.matmul(one_hot_from, sequence_output * col_mask * head_mask)
        col_hidden = self.dropout(gelu(self.hidden_column(col_hidden)))

        prediction_column = self.classifier_column(col_hidden).squeeze(-1).contiguous()
        # probs_column = torch.softmax(prediction_column, dim=-1)
        # probs_column = torch.unsqueeze(probs_column, dim=-1) # [B, C, 1]
        # probs_column = torch.matmul(one_hot_to, probs_column)

        # row
        one_hot_to = one_hot(row_ids, num_classes=256).type(torch.float32)  # [B, S, C]
        one_hot_from = torch.transpose(one_hot_to, 1, 2)  # [B, C, S]

        row_hidden = torch.matmul(one_hot_from, sequence_output * token_mask)
        row_hidden = self.dropout(gelu(self.hidden_row(row_hidden)))

        prediction_row = self.classifier_row(row_hidden) #.contiguous()

        dist_per_row = torch.distributions.Bernoulli(logits=prediction_row)

        if labels is None:
            prediction_row = torch.softmax(prediction_row, dim=-1)
            return prediction_column, torch.softmax(prediction_row, dim=-1)[:, :, 1], dist_per_row.probs, prediction_agg

        # print(torch.softmax(prediction_row, dim=-1)[0, :12, 1])
        # print(labels[0, :12])
        # print('----------')

        # weight = torch.where(
        #     labels == 0,
        #     torch.ones_like(labels, dtype=torch.float32),
        #     10.0 * torch.ones_like(labels, dtype=torch.float32),
        # )
        #
        # EPSILON_ZERO_DIVISION = 1e-10
        # input_mask_float = row_mask.float()
        # selection_loss_per_token = -dist_per_row.log_prob(labels) * weight
        # selection_loss_per_example = torch.sum(selection_loss_per_token * input_mask_float, dim=1) / (
        #         torch.sum(input_mask_float, dim=1) + EPSILON_ZERO_DIVISION
        # )
        # print(torch.sum(torch.sum(selection_loss_per_token * input_mask_float, dim=1) ))

        # loss_selection = torch.mean(selection_loss_per_example)
        loss_row = loss_fct(prediction_row[:, :, 1], labels_row)
        loss_row2 = loss_fct(prediction_row.view(-1, 2), labels.view(-1))

        loss_column = loss_fct(prediction_column, labels_column)
        loss_agg = loss_fct(prediction_agg, labels_agg)

        total_loss = loss_column + loss_agg + loss_row + loss_row2

        return loss_column, loss_row, loss_row2, loss_agg, total_loss


class ModelForQuestionAnsweringV2(nn.Module):
    def __init__(self):
        super(ModelForQuestionAnsweringV2, self).__init__()
        config = TapasConfig.from_pretrained('google/tapas-base-finetuned-wtq')
        self.dropout = nn.Dropout(0.4)
        self.qa_model = TapasForQuestionAnswering(config=config)

        self.row_hidden_layer = ProjectionLayer(hidden_size=config.hidden_size)
        self.col_hidden_layer = ProjectionLayer(hidden_size=config.hidden_size)

        self.encoder = AutoModel.from_pretrained('roberta-base')
        self.classifier_token = nn.Linear(config.hidden_size * 3, 2)

        self.hidden_agg = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden_column = nn.Linear(config.hidden_size, config.hidden_size)

        self.hidden_token_column = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden_row = nn.Linear(config.hidden_size, config.hidden_size)

        self.hidden_combined1 = nn.Linear(config.hidden_size * 3, config.hidden_size * 3)
        self.hidden_combined2 = nn.Linear(config.hidden_size * 3, config.hidden_size * 3)
        self.hidden_combined3 = nn.Linear(config.hidden_size * 3, config.hidden_size * 3)

        self.classifier_column = nn.Linear(config.hidden_size, 1)
        self.classifier_combined = nn.Linear(config.hidden_size * 3, 2)

        self.classifier_agg = nn.Linear(config.hidden_size, 6)

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
                labels_row=None,
                labels_token=None,
                labels_agg=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=torch.zeros_like(input_ids)
        )
        pooled_output = outputs.pooler_output

        loss_fct = CrossEntropyLoss(ignore_index=-1)

        column_ids = token_type_ids[:, :, 1]
        row_ids = token_type_ids[:, :, 2]

        sequence_output = self.dropout(outputs.last_hidden_state)

        token_mask = torch.where(input_ids > 2, torch.tensor(1), torch.tensor(0))
        token_mask = torch.unsqueeze(token_mask, dim=-1)

        col_mask = torch.where(input_ids == 2, torch.tensor(1), torch.tensor(0))
        col_mask = torch.unsqueeze(col_mask, dim=-1)

        head_mask = torch.where(row_ids != 0, torch.tensor(0), torch.tensor(1))
        head_mask = torch.unsqueeze(head_mask, dim=-1)

        # agg
        pooled_output = gelu(self.hidden_agg(pooled_output))
        prediction_agg = self.classifier_agg(pooled_output).contiguous()

        # col
        one_hot_to = one_hot(column_ids, num_classes=256).type(torch.float32) # [B, S, C]
        one_hot_from = torch.transpose(one_hot_to, 1, 2) # [B, C, S]
        a = col_mask * head_mask
        # print(a[0, :150, 0])
        col_hidden = torch.matmul(one_hot_from, sequence_output * col_mask * head_mask)
        col_hidden = self.dropout(gelu(self.hidden_column(col_hidden)))
        prediction_column = self.classifier_column(col_hidden).squeeze(-1).contiguous()

        col_hidden = torch.matmul(one_hot_from, sequence_output * col_mask * head_mask)
        col_hidden = self.dropout(gelu(self.hidden_token_column(col_hidden)))
        col_hidden = torch.matmul(one_hot_to, col_hidden)

        # probs_column = torch.softmax(prediction_column, dim=-1)
        # probs_column = torch.unsqueeze(probs_column, dim=-1) # [B, C, 1]
        # probs_column = torch.matmul(one_hot_to, probs_column)

        # row
        one_hot_to = one_hot(row_ids, num_classes=256).type(torch.float32)  # [B, S, C]
        one_hot_from = torch.transpose(one_hot_to, 1, 2)  # [B, C, S]

        row_hidden = torch.matmul(one_hot_from, sequence_output * token_mask)
        row_hidden = self.dropout(gelu(self.hidden_row(row_hidden)))
        row_hidden = torch.matmul(one_hot_to, row_hidden)

        hidden_combined = torch.cat([col_hidden, row_hidden, sequence_output], dim=-1)
        hidden_combined = self.dropout(gelu(self.hidden_combined1(hidden_combined)))
        hidden_combined = self.dropout(gelu(self.hidden_combined2(hidden_combined)))
        hidden_combined = self.dropout(gelu(self.hidden_combined3(hidden_combined)))

        predictions = self.classifier_combined(hidden_combined)
        prediction_row = torch.matmul(one_hot_from, predictions)[:, :, 1]

        if labels is None:
            predictions = torch.softmax(predictions, dim=-1)
            return prediction_column, prediction_row, predictions[:, :, 1], prediction_agg
        # print(one_hot_to.shape, predictions.shape)

        loss_row = loss_fct(prediction_row, labels_row)
        loss_row2 = loss_fct(predictions.view(-1, 2), labels_token.view(-1))

        loss_column = loss_fct(prediction_column, labels_column)
        loss_agg = loss_fct(prediction_agg, labels_agg)

        total_loss = loss_column + loss_agg + loss_row + loss_row2

        return loss_column, loss_row, loss_row2, loss_agg, total_loss








