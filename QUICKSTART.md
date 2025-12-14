# Hướng Dẫn Sử Dụng Nhanh

## Cài Đặt

```bash
cd BTL_NLP
pip install -r requirements.txt
```

## Chạy Toàn Bộ Pipeline

### Cách 1: Sử dụng main.py (Khuyến nghị)

```bash
# Chạy tất cả: xử lý dữ liệu -> huấn luyện -> đánh giá -> báo cáo
python main.py --mode all

# Hoặc chạy từng bước:
python main.py --mode data      # Chỉ xử lý dữ liệu
python main.py --mode train     # Chỉ huấn luyện
python main.py --mode eval      # Chỉ đánh giá
```

### Cách 2: Chạy từng script riêng

```bash
# 1. Xử lý dữ liệu
python data_processing.py

# 2. Huấn luyện mô hình
python train.py

# 3. Đánh giá mô hình
python evaluate.py

# 4. Tạo báo cáo
python report.py
```

## Demo Nhanh

Sau khi huấn luyện xong, bạn có thể test mô hình:

```bash
python demo.py --sentences "xin chào" "tôi là sinh viên" "hôm nay trời đẹp"
```

## Tùy Chỉnh Hyperparameters

### Training với cấu hình tùy chỉnh:

```bash
python train.py \
    --d_model 512 \
    --n_heads 8 \
    --n_encoder_layers 6 \
    --n_decoder_layers 6 \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 1e-4
```

### Evaluation với beam size khác:

```bash
python evaluate.py \
    --model_path models/best_model.pt \
    --beam_size 10
```

## Cấu Trúc Thư Mục Sau Khi Chạy

```
BTL_NLP/
├── data/
│   ├── src_vocab.pkl          # Vocabulary tiếng Việt
│   └── tgt_vocab.pkl          # Vocabulary tiếng Anh
├── models/
│   ├── best_model.pt          # Mô hình tốt nhất
│   └── checkpoint_epoch_*.pt  # Checkpoints
└── results/
    ├── training_history.png   # Đồ thị loss/perplexity
    ├── training_history.pkl   # Dữ liệu training
    ├── evaluation_report.txt  # Báo cáo đánh giá
    ├── beam_predictions.txt   # Kết quả beam search
    ├── greedy_predictions.txt  # Kết quả greedy search
    ├── references.txt         # References
    └── final_report.md        # Báo cáo tổng hợp
```

## Lưu Ý

1. **Dữ liệu**: Script sẽ tự động tải IWSLT dataset. Nếu không tải được, sẽ sử dụng dữ liệu mẫu để demo.

2. **GPU**: Nếu có GPU, mô hình sẽ tự động sử dụng GPU để tăng tốc.

3. **Thời gian huấn luyện**: 
   - Với CPU: ~vài giờ cho 50 epochs
   - Với GPU: ~30-60 phút cho 50 epochs

4. **Bộ nhớ**: 
   - Dữ liệu nhỏ: ~2-4GB RAM
   - Dữ liệu lớn: Có thể cần 8GB+ RAM

## Troubleshooting

### Lỗi: "CUDA out of memory"
- Giảm `batch_size` (ví dụ: `--batch_size 16`)
- Giảm `max_len` (ví dụ: `--max_len 64`)
- Giảm `d_model` hoặc `d_ff`

### Lỗi: "Module not found"
- Chạy: `pip install -r requirements.txt`

### Lỗi khi tải dữ liệu
- Kiểm tra kết nối internet
- Script sẽ tự động fallback sang dữ liệu mẫu

## Ví Dụ Sử Dụng

### Huấn luyện mô hình nhỏ (nhanh hơn):
```bash
python train.py \
    --d_model 256 \
    --n_heads 4 \
    --n_encoder_layers 3 \
    --n_decoder_layers 3 \
    --d_ff 1024 \
    --batch_size 64 \
    --num_epochs 20
```

### Huấn luyện mô hình lớn (chất lượng tốt hơn):
```bash
python train.py \
    --d_model 512 \
    --n_heads 8 \
    --n_encoder_layers 6 \
    --n_decoder_layers 6 \
    --d_ff 2048 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 5e-5
```

