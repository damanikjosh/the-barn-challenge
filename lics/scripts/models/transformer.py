import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            input_dim,
            num_heads,
            dropout=0.0,
            bias=False,
            encoder_decoder_attention=False,
            causal=False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = input_dim // num_heads
        self.encoder_decoder_attention = encoder_decoder_attention
        self.causal = causal
        self.k_proj = nn.Linear(input_dim, input_dim, bias=bias)
        self.v_proj = nn.Linear(input_dim, input_dim, bias=bias)
        self.q_proj = nn.Linear(input_dim, input_dim, bias=bias)
        self.out_proj = nn.Linear(input_dim, input_dim, bias=bias)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim,)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def multi_head_scaled_dot_product(self,
                                      query: torch.Tensor,
                                      key: torch.Tensor,
                                      value: torch.Tensor,
                                      attention_mask: torch.BoolTensor):
        attn_weights = torch.matmul(query, key.transpose(-1, -2) / math.sqrt(self.input_dim))
        if attention_mask is not None:
            if self.causal:
                attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(0).unsqueeze(1), float("-inf"))
            else:
                attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        concat_attn_output_shape = attn_output.size()[:-2] + (self.input_dim,)
        attn_output = attn_output.view(*concat_attn_output_shape)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            attention_mask: torch.BoolTensor):
        q = self.q_proj(query)
        if self.encoder_decoder_attention:
            k = self.k_proj(key)
            v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attn_output, attn_weights = self.multi_head_scaled_dot_product(q, k, v, attention_mask)
        return attn_output, attn_weights


class PositionWiseFeedForward(nn.Module):

    def __init__(self, input_dim: int, d_ff: int, dropout: float = 0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.activation = nn.ReLU()
        self.w_1 = nn.Linear(input_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, input_dim)
        self.dropout = dropout

    def forward(self, x):
        residual = x
        x = self.activation(self.w_1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.w_2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x + residual


class EmbeddingLidar(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.len_lidar = 720
        self.num_patch = config.num_patch
        self.dim_patch = self.len_lidar // self.num_patch
        self.model_dim = config.model_dim
        self.dropout = config.dropout
        self.pos_embed = nn.Parameter(torch.randn(self.num_patch, self.model_dim))

        self.linear = nn.Linear(self.dim_patch, self.model_dim)

    def forward(self, inputs):
        x = inputs.view([-1, self.num_patch, self.dim_patch])
        x = self.linear(x)
        x = x + self.pos_embed
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_dim = config.input_dim
        self.ffn_dim = config.ffn_dim
        self.self_attn = MultiHeadAttention(
            input_dim=self.input_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.input_dim)
        self.dropout = config.dropout
        self.activation_fn = nn.ReLU()
        self.PositionWiseFeedForward = PositionWiseFeedForward(self.input_dim, self.ffn_dim, config.dropout)
        self.final_layer_norm = nn.LayerNorm(self.input_dim)

    def forward(self, x, encoder_padding_mask):
        residual = x
        x, attn_weights = self.self_attn(query=x, key=x, attention_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        x = self.PositionWiseFeedForward(x)
        x = self.final_layer_norm(x)
        return x, attn_weights


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = config.dropout

        self.embedding = EmbeddingLidar(config)

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])

    def forward(self, inputs, attention_mask=None):
        x = self.embedding(inputs)
        self_attn_scores = []
        for encoder_layer in self.layers:
            x, attn = encoder_layer(x, attention_mask)
            self_attn_scores.append(attn.detach())

        return x, self_attn_scores


class DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_dim = config.input_dim
        self.ffn_dim = config.ffn_dim
        self.dropout = config.dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.input_dim)
        self.encoder_attn = MultiHeadAttention(
            input_dim=self.input_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.input_dim)
        self.PositionWiseFeedForward = PositionWiseFeedForward(self.input_dim, self.ffn_dim, config.dropout)
        self.final_layer_norm = nn.LayerNorm(self.input_dim)

    def forward(
            self,
            x,
            encoder_hidden_states,
            encoder_attention_mask=None,
    ):
        residual = x
        x, cross_attn_weights = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)
        x = self.PositionWiseFeedForward(x)
        x = self.final_layer_norm(x)

        return (
            x,
            cross_attn_weights,
        )


class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = config.dropout
        self.model_dim = config.model_dim
        self.linear = nn.Linear(1, self.model_dim)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])

    # input_ids and some mask variable should be removed? -jw-    
    def forward(
            self,
            inputs,
            encoder_hidden_states,
    ):
        x = inputs
        x = self.linear(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        cross_attention_scores = []
        for idx, decoder_layer in enumerate(self.layers):
            x, layer_cross_attn = decoder_layer(
                x,
                encoder_hidden_states,
            )
            cross_attention_scores.append(layer_cross_attn.detach())
        return x, cross_attention_scores


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model_dim = config.model_dim
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.prediction_head = nn.Linear(self.model_dim * 2, 2)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'weight' in name:
                    nn.init.normal_(param.data, mean=0, std=0.01)
                else:
                    nn.init.constant_(param.data, 0)

    def forward(self, src, trg):
        encoder_output, encoder_attention_scores = self.encoder(
            inputs=src
        )
        decoder_output, decoder_attention_scores = self.decoder(
            trg,
            encoder_output
        )
        decoder_output = decoder_output.reshape((-1, self.model_dim * 2))
        decoder_output = self.prediction_head(decoder_output)

        return decoder_output, encoder_attention_scores, decoder_attention_scores
