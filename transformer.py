"""
Kiến trúc Transformer từ đầu cho bài toán dịch máy Seq2Seq
Bao gồm:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
- Transformer Encoder Layer
- Transformer Decoder Layer
- Transformer Model hoàn chỉnh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_k)
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query tensor [batch_size, n_heads, seq_len, d_k]
            K: Key tensor [batch_size, n_heads, seq_len, d_k]
            V: Value tensor [batch_size, n_heads, seq_len, d_k]
            mask: Attention mask [batch_size, 1, seq_len, seq_len] hoặc None
        Returns:
            output: Attention output [batch_size, n_heads, seq_len, d_k]
            attention_weights: Attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        # Tính QK^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Áp dụng mask nếu có
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Nhân với V
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    Sử dụng nhiều heads để học các loại attention khác nhau
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear layers cho Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len, seq_len] hoặc None
        Returns:
            output: [batch_size, seq_len, d_model]
        """

        
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        k_seq_len = K.size(1)
        v_seq_len = V.size(1)
        

        
        # Linear transformation và chia thành heads
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        

        
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, k_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, v_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Áp dụng attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        # Output length should match Q's length (seq_len), not K/V's length
        output_seq_len = attn_output.size(2)  # This should be seq_len (from Q)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, output_seq_len, self.d_model
        )
        
        # Output projection
        output = self.W_o(attn_output)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Tạo ma trận positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN)
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    - Multi-Head Self-Attention
    - Add & Layer Normalization
    - Feed-Forward Network
    - Add & Layer Normalization
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len, seq_len] hoặc None
        """
        # Self-attention với residual connection và layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward với residual connection và layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    - Masked Multi-Head Self-Attention
    - Add & Layer Normalization
    - Multi-Head Cross-Attention (Encoder-Decoder Attention)
    - Add & Layer Normalization
    - Feed-Forward Network
    - Add & Layer Normalization
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input [batch_size, tgt_len, d_model]
            enc_output: Encoder output [batch_size, src_len, d_model]
            src_mask: Source mask [batch_size, 1, 1, src_len]
            tgt_mask: Target mask [batch_size, 1, tgt_len, tgt_len]
        """
        # Masked self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Cross-attention (Encoder-Decoder Attention)
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class Transformer(nn.Module):
    """
    Transformer Model hoàn chỉnh cho Seq2Seq
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8,
                 n_encoder_layers=6, n_decoder_layers=6, d_ff=2048, max_len=5000,
                 dropout=0.1, pad_idx=0):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def generate_mask(self, src, tgt=None):
        """
        Tạo attention masks
        - src_mask: che các padding tokens trong source
        - tgt_mask: che các padding tokens và future tokens trong target
        """
        # Source mask: [batch_size, 1, 1, src_len]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        if tgt is not None:
            # Target mask: [batch_size, 1, tgt_len, tgt_len]
            tgt_len = tgt.size(1)
            tgt_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(3)  # Padding mask
            # Causal mask: che các future tokens
            causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
            tgt_mask = tgt_mask & causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            tgt_mask = None
        
        return src_mask, tgt_mask
    
    def encode(self, src):
        """
        Encoder forward pass
        Args:
            src: [batch_size, src_len]
        Returns:
            enc_output: [batch_size, src_len, d_model]
        """
        # Embedding + positional encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        
        # Encoder layers
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, mask=None)
        
        return enc_output
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        """
        Decoder forward pass
        Args:
            tgt: [batch_size, tgt_len]
            enc_output: [batch_size, src_len, d_model]
            src_mask: [batch_size, 1, 1, src_len]
            tgt_mask: [batch_size, 1, tgt_len, tgt_len]
        Returns:
            dec_output: [batch_size, tgt_len, d_model]
        """
        # Embedding + positional encoding
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # Decoder layers
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        return dec_output
    
    def forward(self, src, tgt):
        """
        Forward pass hoàn chỉnh
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
        Returns:
            output: [batch_size, tgt_len, tgt_vocab_size]
        """
        # Tạo masks
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # Encode
        enc_output = self.encode(src)
        
        # Decode
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        # Output projection
        output = self.output_projection(dec_output)
        
        return output


def create_transformer_model(src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8,
                            n_encoder_layers=6, n_decoder_layers=6, d_ff=2048,
                            dropout=0.1, pad_idx=0):
    """
    Factory function để tạo Transformer model
    """
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        dropout=dropout,
        pad_idx=pad_idx
    )
    
    # Khởi tạo weights
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model

