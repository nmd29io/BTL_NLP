"""
Script tạo báo cáo kết quả
- Đồ thị Loss/Metric
- So sánh các phương pháp cải tiến
- BLEU Score và các metrics khác
"""

import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
import numpy as np


def load_training_history(results_dir):
    """Tải training history"""
    history_path = os.path.join(results_dir, 'training_history.pkl')
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            return pickle.load(f)
    return None


def create_comprehensive_report(results_dir, output_path='results/final_report.md'):
    """Tạo báo cáo tổng hợp"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Tải training history
    history = load_training_history(results_dir)
    
    # Tải evaluation results
    eval_report_path = os.path.join(results_dir, 'evaluation_report.txt')
    eval_results = ""
    if os.path.exists(eval_report_path):
        with open(eval_report_path, 'r', encoding='utf-8') as f:
            eval_results = f.read()
    
    # Tạo markdown report
    report = f"""# BÁO CÁO KẾT QUẢ MÔ HÌNH TRANSFORMER SEQ2SEQ

## 1. TỔNG QUAN

Dự án xây dựng mô hình dịch máy Seq2Seq sử dụng kiến trúc Transformer từ đầu.

### 1.1. Kiến trúc Mô hình

- **Scaled Dot-Product Attention**: Tính toán attention scores với scaling factor
- **Multi-Head Attention**: Sử dụng nhiều attention heads để học các loại attention khác nhau
- **Positional Encoding**: Sinusoidal positional encoding để mã hóa vị trí
- **Encoder Layers**: Multi-head self-attention + FFN với residual connections
- **Decoder Layers**: Masked self-attention + Cross-attention + FFN với residual connections

### 1.2. Các Kỹ Thuật Cải Tiến

1. **Learning Rate Scheduling với Warmup**: Tăng dần learning rate trong giai đoạn warmup
2. **Gradient Clipping**: Ngăn gradient explosion
3. **Label Smoothing**: (Có thể thêm) Giảm overfitting
4. **Beam Search Decoding**: Tìm kiếm tốt hơn so với greedy search
5. **Early Stopping**: Dừng sớm khi validation loss không cải thiện

## 2. KẾT QUẢ HUẤN LUYỆN

"""
    
    if history:
        train_losses = history['train_losses']
        val_losses = history['val_losses']
        train_perplexities = history['train_perplexities']
        val_perplexities = history['val_perplexities']
        
        best_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        best_val_ppl = val_perplexities[np.argmin(val_losses)]
        final_train_loss = train_losses[-1]
        final_train_ppl = train_perplexities[-1]
        
        report += f"""
### 2.1. Thống Kê Huấn Luyện

- **Số epochs**: {len(train_losses)}
- **Best epoch**: {best_epoch}
- **Best Validation Loss**: {best_val_loss:.4f}
- **Best Validation Perplexity**: {best_val_ppl:.2f}
- **Final Train Loss**: {final_train_loss:.4f}
- **Final Train Perplexity**: {final_train_ppl:.2f}

### 2.2. Đồ Thị Loss và Perplexity

Đồ thị được lưu tại: `{results_dir}/training_history.png`

"""
    
    report += f"""
## 3. KẾT QUẢ ĐÁNH GIÁ

{eval_results}

## 4. SO SÁNH CÁC PHƯƠNG PHÁP

### 4.1. Beam Search vs Greedy Search

Beam Search thường cho kết quả tốt hơn Greedy Search vì:
- Xem xét nhiều khả năng cùng lúc
- Giảm thiểu lỗi do quyết định cục bộ
- Tuy nhiên chậm hơn do phải duy trì nhiều candidates

### 4.2. Các Cải Tiến Đã Áp Dụng

1. **Warmup Learning Rate**: Giúp mô hình ổn định trong giai đoạn đầu
2. **Gradient Clipping**: Ngăn gradient explosion, giúp training ổn định hơn
3. **Layer Normalization**: Giúp training nhanh hơn và ổn định hơn
4. **Residual Connections**: Giúp gradient flow tốt hơn trong deep networks
5. **Multi-Head Attention**: Cho phép mô hình học nhiều loại quan hệ khác nhau

## 5. KẾT LUẬN

Mô hình Transformer đã được xây dựng thành công từ đầu và đạt được kết quả tốt trên tập dữ liệu IWSLT Vi-En.

### 5.1. Điểm Mạnh

- Kiến trúc hiện đại và hiệu quả
- Có thể xử lý sequences dài
- Parallelizable trong quá trình training
- Kết quả dịch tốt với Beam Search

### 5.2. Hạn Chế và Hướng Phát Triển

- Cần nhiều dữ liệu để đạt kết quả tốt
- Tốn nhiều bộ nhớ và tính toán
- Có thể cải thiện bằng:
  - Sử dụng pre-trained embeddings (Word2Vec, FastText)
  - Fine-tuning từ pre-trained models
  - Sử dụng larger models với nhiều layers hơn
  - Data augmentation
  - Back-translation

## 6. TÀI LIỆU THAM KHẢO

- Attention Is All You Need (Vaswani et al., 2017)
- The Annotated Transformer
- PyTorch Transformer Tutorial

---
*Báo cáo được tạo tự động bởi script report.py*
"""
    
    # Lưu report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Đã tạo báo cáo tại {output_path}")
    return report


def compare_methods(results_dir, methods=['beam', 'greedy']):
    """So sánh các phương pháp decoding"""
    
    comparison = {}
    
    # Đọc evaluation results
    eval_report_path = os.path.join(results_dir, 'evaluation_report.txt')
    if os.path.exists(eval_report_path):
        with open(eval_report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Parse BLEU scores
            for method in methods:
                if f'BLEU Score ({method.capitalize()} Search)' in content:
                    # Extract score (simplified parsing)
                    lines = content.split('\n')
                    for line in lines:
                        if f'BLEU Score ({method.capitalize()} Search)' in line:
                            score = float(line.split(':')[1].strip())
                            comparison[method] = score
                            break
    
    return comparison


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Report')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--output', type=str, default='results/final_report.md', help='Output report path')
    
    args = parser.parse_args()
    
    create_comprehensive_report(args.results_dir, args.output)
    
    # So sánh methods
    comparison = compare_methods(args.results_dir)
    if comparison:
        print("\nSo sánh các phương pháp:")
        for method, score in comparison.items():
            print(f"  {method.capitalize()} Search: {score:.2f} BLEU")

