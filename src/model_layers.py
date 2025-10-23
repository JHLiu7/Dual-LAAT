import torch
import torch.nn as nn
import numpy as np

from typing import Dict, Optional, List
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import AutoModel 



def perform_attention(x_text, x_code, projection_text, projection_code, **kwargs):
    # Perform attention
    representation_text = torch.tanh(projection_text(x_text))
    representation_code = torch.tanh(projection_code(x_code))

    attention_weight = torch.softmax(
        representation_code.matmul(representation_text.transpose(1, 2)), dim=2
    )
    weighted_text = attention_weight @ x_text
    return weighted_text


def perform_attention_dim(x_text, x_code, projection_text, projection_code, text_attention_mask=None):
    # Perform attention
    representation_text = projection_text(x_text)
    representation_code = torch.tanh(projection_code(x_code))

    attention_score = representation_code.matmul(torch.tanh(representation_text).transpose(1, 2))
    if text_attention_mask is not None:
        attention_score = attention_score.masked_fill(
            text_attention_mask.unsqueeze(1) == 0, float('-inf')
        )

    attention_weight = torch.softmax(attention_score, dim=2)
    weighted_text = attention_weight @ representation_text
    return weighted_text


class DualCNN(nn.Module):
    def __init__(
        self,
        word_embeddings: np.ndarray,

        kernel_size: int = 10,
        num_filters: int = 128,
        dropout: float = 0.1,

        num_mha_heads: int = 1,
        projection_dim: int = 128,

        dual_attention: bool = False,
        dual_attention_lambda: float = 0.5
    ):
        super().__init__()

        self.dual_attention = dual_attention
        self.dual_attention_lambda = dual_attention_lambda

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(word_embeddings, dtype=torch.float32), freeze=False)
        self.embedding_dropout = nn.Dropout(dropout)

        self.cnn_text = nn.Conv1d(in_channels=word_embeddings.shape[1], out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.cnn_code = nn.Conv1d(in_channels=word_embeddings.shape[1], out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        
        self.text_dim = num_filters
        self.code_dim = num_filters
        self.encode_dim = self.text_dim * num_mha_heads

        self.projection_text_layers = nn.ModuleList([nn.Linear(self.text_dim, projection_dim) for _ in range(num_mha_heads)])
        self.projection_code_layers = nn.ModuleList([nn.Linear(self.code_dim, projection_dim) for _ in range(num_mha_heads)])

        self.final_layer = nn.Sequential(
            nn.Linear(self.encode_dim, self.encode_dim),
            nn.ReLU(),
            nn.Linear(self.encode_dim, 1)
        )

        self.init_weights([self.cnn_text, self.cnn_code, *self.projection_text_layers, *self.projection_code_layers, *self.final_layer])

    def init_weights(self, layers):
        for layer in layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)

    def forward(self, input_text, input_code):
        x_text = self.embedding(input_text)
        x_text = self.embedding_dropout(x_text)
        x_code = self.embedding(input_code)

        if not self.dual_attention:
            x_text = self.cnn_text(x_text.transpose(1, 2)).transpose(1, 2)

            x_code = self.cnn_code(x_code.transpose(1, 2)).transpose(1, 2)
            x_code = x_code.max(dim=1)[0]

            mha_outputs = [
                perform_attention(x_text, x_code, projection_text, projection_code)
                for projection_text, projection_code in zip(self.projection_text_layers, self.projection_code_layers)
            ]
            mha_outputs = torch.cat(mha_outputs, dim=2)

            logits = self.final_layer(mha_outputs).squeeze(2)
            return logits
        
        else:

            # Encode with CNN
            x_text = self.cnn_text(x_text.transpose(1, 2)).transpose(1, 2)
            x_code = self.cnn_code(x_code.transpose(1, 2)).transpose(1, 2)

            # Get query representation
            x_code_query = x_code.max(dim=1)[0]
            x_text_query = x_text.max(dim=1)[0]

            # Text to Code Attention
            mha_outputs_tc = [
                perform_attention(x_text, x_code_query, projection_text, projection_code)
                for projection_text, projection_code in zip(self.projection_text_layers, self.projection_code_layers)
            ]
            mha_outputs_tc = torch.cat(mha_outputs_tc, dim=2)

            # Code to Text Attention
            mha_outputs_ct = [
                perform_attention(x_code, x_text_query, projection_code, projection_text)
                for projection_text, projection_code in zip(self.projection_text_layers, self.projection_code_layers)
            ]
            mha_outputs_ct = torch.cat(mha_outputs_ct, dim=2)
            mha_outputs_ct = mha_outputs_ct.transpose(0, 1) # (batch_size, label_num, dim)

            # Combine attention results and classify
            logits = self.final_layer(
                self.dual_attention_lambda * mha_outputs_tc + (1 - self.dual_attention_lambda) * mha_outputs_ct
            ).squeeze(2)
            return logits
    

class DualRNN(nn.Module):
    
    def __init__(
        self,
        word_embeddings: np.ndarray,

        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        dropout: float = 0.1,
        rnn_type: str = 'lstm',

        num_mha_heads: int = 1,
        projection_dim: int = 128
    ):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(word_embeddings, dtype=torch.float32), freeze=False)
        self.embedding_dropout = nn.Dropout(dropout)


        if rnn_type.lower() == 'lstm':
            self.rnn_text = nn.LSTM(input_size=word_embeddings.shape[1], hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, bidirectional=True, dropout=dropout, batch_first=True)
            self.rnn_code = nn.LSTM(input_size=word_embeddings.shape[1], hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        elif rnn_type.lower() == 'gru':
            self.rnn_text = nn.GRU(input_size=word_embeddings.shape[1], hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, bidirectional=True, dropout=dropout, batch_first=True)
            self.rnn_code = nn.GRU(input_size=word_embeddings.shape[1], hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Use 'lstm' or 'gru'.")

        self.text_dim = rnn_hidden_size * 2
        self.code_dim = rnn_hidden_size * 2
        self.encode_dim = self.text_dim * num_mha_heads

        self.projection_text_layers = nn.ModuleList([nn.Linear(self.text_dim, projection_dim) for _ in range(num_mha_heads)])
        self.projection_code_layers = nn.ModuleList([nn.Linear(self.code_dim, projection_dim) for _ in range(num_mha_heads)])

        self.final_layer = nn.Sequential(
            nn.Linear(self.encode_dim, self.encode_dim),
            nn.ReLU(),
            nn.Linear(self.encode_dim, 1)
        )

        self.init_weights([self.rnn_text, self.rnn_code, *self.projection_text_layers, *self.projection_code_layers, *self.final_layer])

    def init_weights(self, layers):
        for layer in layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)

    def forward(self, x_text, x_code):
        x_text = self.embedding(x_text)
        x_text = self.embedding_dropout(x_text)
        x_text, _ = self.rnn_text(x_text)

        x_code = self.embedding(x_code)
        _, x_out = self.rnn_code(x_code)
        if isinstance(x_out, tuple):
            x_out = x_out[0]
        x_code = x_out.transpose(0, 1).flatten(1, 2)

        mha_outputs = [
            perform_attention(x_text, x_code, projection_text, projection_code)
            for projection_text, projection_code in zip(self.projection_text_layers, self.projection_code_layers)
        ]
        mha_outputs = torch.cat(mha_outputs, dim=2)
        logits = self.final_layer(mha_outputs).squeeze(2)
        return logits


