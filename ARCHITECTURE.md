# Kiến Trúc Transformer - Tài Liệu Kỹ Thuật

## 1. Tổng Quan

Mô hình Transformer được xây dựng từ đầu theo kiến trúc trong paper "Attention Is All You Need" (Vaswani et al., 2017), được sử dụng cho bài toán dịch máy Seq2Seq từ tiếng Việt sang tiếng Anh.

## 2. Các Thành Phần Cốt Lõi

### 2.1. Scaled Dot-Product Attention

**Công thức:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Đặc điểm:**
- Scaling factor `√d_k` giúp tránh gradient quá nhỏ khi `d_k` lớn
- Độ phức tạp: O(n²) với n là độ dài sequence
- Cho phép mô hình tập trung vào các phần quan trọng của input

**Implementation:**
- File: `transformer.py` - Class `ScaledDotProductAttention`
- Input: Q, K, V tensors [batch_size, n_heads, seq_len, d_k]
- Output: Attention output và attention weights

### 2.2. Multi-Head Attention

**Công thức:**
```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O
where headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)
```

**Đặc điểm:**
- Cho phép mô hình học nhiều loại quan hệ khác nhau đồng thời
- Mỗi head có thể tập trung vào các khía cạnh khác nhau
- Thường sử dụng 8 heads với d_model = 512

**Implementation:**
- File: `transformer.py` - Class `MultiHeadAttention`
- Sử dụng `ScaledDotProductAttention` cho mỗi head
- Concatenate và project output

### 2.3. Positional Encoding

**Công thức Sinusoidal:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Đặc điểm:**
- Mã hóa vị trí tuyệt đối và tương đối
- Không cần học (fixed encoding)
- Cho phép mô hình hiểu thứ tự của tokens

**Implementation:**
- File: `transformer.py` - Class `PositionalEncoding`
- Pre-computed và cached
- Thêm vào embeddings trước khi đưa vào encoder/decoder

### 2.4. Encoder Layer

**Cấu trúc:**
```
EncoderLayer:
  1. Multi-Head Self-Attention
     └─> Add & Layer Normalization
  2. Feed-Forward Network
     └─> Add & Layer Normalization
```

**Đặc điểm:**
- Self-attention: Q, K, V đều từ cùng một input
- Residual connections giúp gradient flow tốt hơn
- Layer normalization giúp training ổn định

**Implementation:**
- File: `transformer.py` - Class `EncoderLayer`
- Stack nhiều layers (thường 6 layers)

### 2.5. Decoder Layer

**Cấu trúc:**
```
DecoderLayer:
  1. Masked Multi-Head Self-Attention
     └─> Add & Layer Normalization
  2. Multi-Head Cross-Attention (Encoder-Decoder Attention)
     └─> Add & Layer Normalization
  3. Feed-Forward Network
     └─> Add & Layer Normalization
```

**Đặc điểm:**
- **Masked Self-Attention**: Che các future tokens (causal mask)
- **Cross-Attention**: Q từ decoder, K và V từ encoder output
- Cho phép decoder "nhìn" vào encoder output khi generate

**Implementation:**
- File: `transformer.py` - Class `DecoderLayer`
- Mask được tạo tự động trong `generate_mask()`

### 2.6. Feed-Forward Network (FFN)

**Công thức:**
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

**Đặc điểm:**
- 2-layer MLP với ReLU activation
- Thường d_ff = 4 * d_model (ví dụ: 2048 với d_model = 512)
- Áp dụng cho mỗi vị trí độc lập

**Implementation:**
- File: `transformer.py` - Class `FeedForward`

## 3. Kiến Trúc Hoàn Chỉnh

### 3.1. Encoder Stack

```
Input Embedding + Positional Encoding
    ↓
[Encoder Layer 1]
    ↓
[Encoder Layer 2]
    ↓
    ...
    ↓
[Encoder Layer N]
    ↓
Encoder Output
```

### 3.2. Decoder Stack

```
Output Embedding + Positional Encoding
    ↓
[Decoder Layer 1]
    ↓
[Decoder Layer 2]
    ↓
    ...
    ↓
[Decoder Layer N]
    ↓
Linear Projection
    ↓
Output Probabilities
```

### 3.3. Full Model

```
Source Sequence
    ↓
Encoder → Encoder Output
    ↓
Target Sequence (shifted right)
    ↓
Decoder → Decoder Output
    ↓
Linear → Target Vocabulary Probabilities
```

## 4. Các Kỹ Thuật Cải Tiến

### 4.1. Learning Rate Scheduling với Warmup

**Công thức:**
```
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

**Lợi ích:**
- Tăng dần learning rate trong giai đoạn đầu
- Giúp mô hình ổn định khi bắt đầu training
- Sau warmup, learning rate giảm dần

**Implementation:**
- File: `utils.py` - Class `WarmupScheduler`

### 4.2. Gradient Clipping

**Mục đích:**
- Ngăn gradient explosion
- Giúp training ổn định hơn

**Implementation:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4.3. Label Smoothing (Có thể thêm)

**Công thức:**
```
smooth_label = (1 - ε) * one_hot + ε / vocab_size
```

**Lợi ích:**
- Giảm overfitting
- Cải thiện generalization

### 4.4. Beam Search Decoding

**Thuật toán:**
1. Khởi tạo beam với SOS token
2. Với mỗi step:
   - Expand tất cả sequences trong beam
   - Lấy top-k candidates từ mỗi sequence
   - Chọn top-k sequences tốt nhất
3. Dừng khi tất cả sequences kết thúc hoặc đạt max_len
4. Chọn sequence tốt nhất (normalized by length)

**Lợi ích:**
- Tốt hơn greedy search
- Xem xét nhiều khả năng cùng lúc
- Trade-off giữa chất lượng và tốc độ

**Implementation:**
- File: `evaluate.py` - Class `BeamSearchDecoder`

## 5. Hyperparameters Mặc Định

```python
d_model = 512              # Model dimension
n_heads = 8                # Number of attention heads
n_encoder_layers = 6       # Number of encoder layers
n_decoder_layers = 6       # Number of decoder layers
d_ff = 2048                # Feed-forward dimension
dropout = 0.1              # Dropout rate
max_len = 128              # Maximum sequence length
batch_size = 32            # Batch size
learning_rate = 1e-4       # Learning rate
warmup_steps = 4000        # Warmup steps
```

## 6. Độ Phức Tạp

- **Time Complexity**: O(n²) cho attention với n là sequence length
- **Space Complexity**: O(n²) cho attention matrices
- **Parameters**: ~65M với cấu hình mặc định

## 7. So Sánh với RNN/LSTM

| Đặc điểm | Transformer | RNN/LSTM |
|----------|-------------|----------|
| Parallelization | ✅ Có thể parallelize | ❌ Sequential |
| Long-range dependencies | ✅ Tốt | ⚠️ Khó khăn |
| Training speed | ✅ Nhanh hơn | ❌ Chậm hơn |
| Memory | ⚠️ O(n²) | ✅ O(n) |
| Interpretability | ✅ Attention weights | ❌ Hidden states |

## 8. Tài Liệu Tham Khảo

1. Vaswani et al. (2017). "Attention Is All You Need". NIPS.
2. The Annotated Transformer: http://nlp.seas.harvard.edu/annotated-transformer/
3. PyTorch Transformer Tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

