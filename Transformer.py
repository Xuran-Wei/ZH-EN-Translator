import math

import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维用cos

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 不参与梯度更新

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # 添加调试信息
            # print(f"Attention - scores shape: {scores.shape}, mask shape: {mask.shape}")

            # 确保掩码与注意力分数形状匹配
            # scores: [batch_size, n_heads, seq_len, seq_len]
            # mask: [batch_size, 1, seq_len, seq_len] 需要扩展到多头

            if mask.dim() == 4:
                if mask.size(1) == 1:  # 如果掩码只有1个头
                    mask = mask.expand(-1, query.size(1), -1, -1)  # 扩展到多头
                elif mask.size(1) != query.size(1):
                    # 如果头数不匹配，重新扩展
                    mask = mask.expand(-1, query.size(1), -1, -1)

            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, value)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 分别获取每个序列的长度
        Q_len = query.size(1)
        K_len = key.size(1)
        V_len = value.size(1)

        # 线性变换
        Q = self.w_q(query)  # [batch_size, Q_len, d_model]
        K = self.w_k(key)  # [batch_size, K_len, d_model]
        V = self.w_v(value)  # [batch_size, V_len, d_model]

        # 重塑为多头格式
        Q = Q.view(batch_size, Q_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, Q_len, d_k]
        K = K.view(batch_size, K_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, K_len, d_k]
        V = V.view(batch_size, V_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, V_len, d_k]

        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # 恢复原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, Q_len, self.d_model)
        output = self.w_o(attn_output)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力 + 残差 + 归一化
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差 + 归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 掩码自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 编码器-解码器注意力
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model

        # 词嵌入层（将索引映射为d_model维度向量）
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码层
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 堆叠N个编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 词嵌入后的dropout（防止过拟合）
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 1. 词嵌入：索引→向量，并乘以√d_model（平衡词嵌入与位置编码的尺度）
        x = self.embedding(x) * math.sqrt(self.d_model)
        # 2. 叠加位置编码
        x = self.pos_encoding(x)
        # 3. 应用dropout
        x = self.dropout(x)
        # 4. 经过N个编码器层
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model

        # 目标序列的词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码层（与编码器共享结构，参数独立）
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 堆叠N个解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 词嵌入后的dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 1. 词嵌入 + 缩放
        x = self.embedding(x) * math.sqrt(self.d_model)
        # 2. 叠加位置编码
        x = self.pos_encoding(x)
        # 3. 应用dropout
        x = self.dropout(x)
        # 4. 经过N个解码器层
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout)
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.linear(dec_output)
        return output


def create_mask(src, tgt, pad_idx):
    # 源序列掩码: [batch_size, 1, 1, src_len]
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # 目标序列掩码
    tgt_len = tgt.size(1)

    # 填充掩码: [batch_size, 1, tgt_len, 1]
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)

    # 未来掩码: [1, 1, tgt_len, tgt_len]
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=src.device)).bool()
    tgt_sub_mask = tgt_sub_mask.unsqueeze(0).unsqueeze(0)

    # 组合掩码: [batch_size, 1, tgt_len, tgt_len]
    tgt_mask = tgt_pad_mask & tgt_sub_mask

    return src_mask, tgt_mask