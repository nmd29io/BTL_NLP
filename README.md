# Transformer Seq2Seq Machine Translation

Dự án xây dựng mô hình dịch máy Seq2Seq sử dụng kiến trúc Transformer từ đầu cho bài toán dịch Việt-Anh.

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Cài Đặt](#cài-đặt)
- [Sử Dụng](#sử-dụng)
- [Kiến Trúc](#kiến-trúc)
- [Kết Quả](#kết-quả)
- [Tài Liệu](#tài-liệu)

## 🎯 Tổng Quan

Dự án này triển khai đầy đủ kiến trúc Transformer từ các thành phần cơ bản nhất:
- ✅ Scaled Dot-Product Attention & Multi-Head Attention
- ✅ Positional Encoding (Sinusoidal)
- ✅ Transformer Encoder & Decoder Layers
- ✅ Training với Warmup Learning Rate Scheduler
- ✅ Beam Search & Greedy Search Decoding
- ✅ Evaluation với BLEU Score

## 📁 Cấu Trúc Dự Án

```
BTL_NLP/
├── data/                    # Thư mục chứa dữ liệu và vocabulary
├── models/                  # Lưu mô hình đã huấn luyện
├── results/                 # Kết quả, đồ thị và báo cáo
├── data_processing.py       # Xử lý dữ liệu (tokenization, vocabulary, dataloader)
├── transformer.py           # Kiến trúc Transformer từ đầu
├── train.py                 # Script huấn luyện
├── evaluate.py              # Script đánh giá với Beam Search
├── utils.py                 # Các hàm tiện ích (scheduler, visualization)
├── report.py                # Tạo báo cáo kết quả
├── demo.py                  # Demo dịch câu
├── main.py                  # Script chạy toàn bộ pipeline
├── requirements.txt         # Dependencies
├── README.md                # File này
├── QUICKSTART.md            # Hướng dẫn sử dụng nhanh
└── ARCHITECTURE.md          # Tài liệu kỹ thuật chi tiết
```

## 🔧 Cài Đặt

```bash
# Clone hoặc tải dự án
cd BTL_NLP

# Cài đặt dependencies
pip install -r requirements.txt
```

## 🚀 Sử Dụng

### Cách 1: Chạy toàn bộ pipeline (Khuyến nghị)

```bash
python main.py --mode all
```

### Cách 2: Chạy từng bước

```bash
# 1. Xử lý dữ liệu
python data_processing.py

# 2. Huấn luyện mô hình
python train.py

# 3. Đánh giá mô hình
python evaluate.py

# 4. Tạo báo cáo
python report.py

# 5. Demo dịch câu
python demo.py --sentences "xin chào" "tôi là sinh viên"
```

Xem thêm chi tiết trong [QUICKSTART.md](QUICKSTART.md)

## 🏗️ Kiến Trúc

### Các Thành Phần Chính

1. **Scaled Dot-Product Attention**
   - Tính toán attention scores với scaling factor √d_k
   - Công thức: `Attention(Q, K, V) = softmax(QK^T / √d_k) V`

2. **Multi-Head Attention**
   - Sử dụng nhiều attention heads (mặc định: 8 heads)
   - Mỗi head học các loại quan hệ khác nhau

3. **Positional Encoding (Sinusoidal)**
   - Mã hóa vị trí tuyệt đối và tương đối
   - Không cần học (fixed encoding)

4. **Transformer Encoder Layer**
   - Multi-Head Self-Attention
   - Feed-Forward Network
   - Residual Connections & Layer Normalization

5. **Transformer Decoder Layer**
   - Masked Multi-Head Self-Attention
   - Multi-Head Cross-Attention (Encoder-Decoder)
   - Feed-Forward Network
   - Residual Connections & Layer Normalization

6. **Decoding**
   - Beam Search (ưu tiên)
   - Greedy Search

Xem chi tiết kỹ thuật trong [ARCHITECTURE.md](ARCHITECTURE.md)

## 📊 Kết Quả

Sau khi huấn luyện, kết quả sẽ được lưu tại thư mục `results/`:

- `training_history.png`: Đồ thị Loss và Perplexity
- `evaluation_report.txt`: BLEU Score và metrics
- `final_report.md`: Báo cáo tổng hợp
- `beam_predictions.txt`: Kết quả dịch với Beam Search
- `greedy_predictions.txt`: Kết quả dịch với Greedy Search

## 📚 Tài Liệu

- [QUICKSTART.md](QUICKSTART.md) - Hướng dẫn sử dụng nhanh
- [ARCHITECTURE.md](ARCHITECTURE.md) - Tài liệu kỹ thuật chi tiết
- [Paper gốc](https://arxiv.org/abs/1706.03762) - "Attention Is All You Need"

## ⚙️ Hyperparameters Mặc Định

```python
d_model = 512
n_heads = 8
n_encoder_layers = 6
n_decoder_layers = 6
d_ff = 2048
dropout = 0.1
batch_size = 32
learning_rate = 1e-4
warmup_steps = 4000
beam_size = 5
```

## 🎓 Các Kỹ Thuật Cải Tiến

1. **Warmup Learning Rate Scheduler**: Tăng dần LR trong giai đoạn đầu
2. **Gradient Clipping**: Ngăn gradient explosion
3. **Beam Search Decoding**: Tìm kiếm tốt hơn greedy search
4. **Early Stopping**: Dừng sớm khi validation loss không cải thiện
5. **Layer Normalization**: Giúp training ổn định hơn

## 📝 Ghi Chú

- Dự án được xây dựng hoàn toàn từ đầu, không sử dụng pre-built Transformer từ thư viện
- Tất cả các thành phần (Attention, Positional Encoding, Encoder, Decoder) đều được implement từ đầu
- Code được comment chi tiết bằng tiếng Việt để dễ hiểu

## 👥 Đóng Góp

Dự án này được phát triển cho mục đích học tập và nghiên cứu.

