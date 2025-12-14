# Hướng Dẫn Training Transformer với IWSLT trong 1 Giờ

## Tổng Quan

Script `train_fast.py` được tối ưu hóa để train mô hình Transformer trên dataset IWSLT Vi-En trong vòng 1 giờ với các tối ưu sau:

### Tối Ưu Hóa

1. **Model nhỏ hơn**:
   - `d_model`: 256 (thay vì 512)
   - `n_heads`: 4 (thay vì 8)
   - `n_encoder_layers`: 3 (thay vì 6)
   - `n_decoder_layers`: 3 (thay vì 6)
   - `d_ff`: 1024 (thay vì 2048)

2. **Batch size lớn hơn**: 128 (thay vì 32)

3. **Sequence length ngắn hơn**: 64 (thay vì 128)

4. **Mixed Precision Training (AMP)**: Sử dụng `torch.cuda.amp` để tăng tốc độ training

5. **Tối ưu Data Loading**:
   - `num_workers`: 4 (nếu có GPU)
   - `pin_memory`: True (nếu có GPU)

6. **Time-based Training**: Tự động dừng sau 1 giờ

## Cách Sử Dụng

### Cách 1: Sử dụng script có sẵn (Windows)

```bash
run_train_1h.bat
```

### Cách 2: Sử dụng script có sẵn (Linux/Mac)

```bash
chmod +x run_train_1h.sh
./run_train_1h.sh
```

### Cách 3: Chạy trực tiếp với Python

```bash
python train_fast.py --max_time_hours 1.0 --batch_size 128 --d_model 256 --n_heads 4 --n_encoder_layers 3 --n_decoder_layers 3 --d_ff 1024 --max_len 64 --use_amp
```

### Cách 4: Tùy chỉnh tham số

```bash
python train_fast.py \
    --max_time_hours 1.0 \
    --batch_size 128 \
    --d_model 256 \
    --n_heads 4 \
    --n_encoder_layers 3 \
    --n_decoder_layers 3 \
    --d_ff 1024 \
    --max_len 64 \
    --learning_rate 1e-4 \
    --warmup_steps 2000 \
    --use_amp \
    --num_workers 4 \
    --iwslt_dir iwslt_en_vi
```

## Các Tham Số Quan Trọng

- `--max_time_hours`: Thời gian tối đa để train (mặc định: 1.0 giờ)
- `--batch_size`: Batch size (mặc định: 128)
- `--d_model`: Kích thước embedding (mặc định: 256)
- `--n_heads`: Số lượng attention heads (mặc định: 4)
- `--n_encoder_layers`: Số lớp encoder (mặc định: 3)
- `--n_decoder_layers`: Số lớp decoder (mặc định: 3)
- `--d_ff`: Kích thước feed-forward (mặc định: 1024)
- `--max_len`: Độ dài tối đa của sequence (mặc định: 64)
- `--use_amp`: Bật mixed precision training (mặc định: True)
- `--num_workers`: Số worker cho data loading (mặc định: 4)

## Kết Quả

Sau khi training, bạn sẽ có:

1. **Model tốt nhất**: `models/best_model.pt`
2. **Model cuối cùng**: `models/final_model.pt`
3. **Checkpoints**: `models/checkpoint_epoch_*.pt`
4. **Training history**: `results/training_history.png` và `results/training_history.pkl`

## Lưu Ý

- Script sẽ tự động dừng sau 1 giờ (hoặc thời gian bạn chỉ định)
- Nếu có GPU, script sẽ sử dụng GPU và mixed precision training
- Nếu không có GPU, script vẫn chạy được nhưng sẽ chậm hơn
- Dữ liệu IWSLT cần được đặt trong thư mục `iwslt_en_vi/` với các file:
  - `train.vi` và `train.en`
  - `tst2012.vi` và `tst2012.en` (validation)
  - `tst2013.vi` và `tst2013.en` (test)

## Đánh Giá Model

Sau khi training xong, bạn có thể đánh giá model bằng:

```bash
python evaluate.py --model_path models/best_model.pt
```

## So Sánh với Training Thông Thường

| Tham số | Training Thông Thường | Training 1 Giờ |
|---------|----------------------|----------------|
| d_model | 512 | 256 |
| n_heads | 8 | 4 |
| n_layers | 6 | 3 |
| d_ff | 2048 | 1024 |
| batch_size | 32 | 128 |
| max_len | 128 | 64 |
| Mixed Precision | Không | Có |
| Thời gian | Nhiều giờ | 1 giờ |

## Tối Ưu Thêm (Nếu Cần)

Nếu bạn muốn train nhanh hơn nữa:

1. Giảm `d_model` xuống 128
2. Giảm `n_heads` xuống 2
3. Giảm `n_layers` xuống 2
4. Tăng `batch_size` lên 256 (nếu GPU đủ bộ nhớ)
5. Giảm `max_len` xuống 32

Tuy nhiên, điều này sẽ làm giảm chất lượng model.

