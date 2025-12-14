"""
Script huấn luyện mô hình Transformer tối ưu cho 1 giờ
- Model nhỏ hơn để train nhanh
- Mixed precision training
- Batch size lớn
- Time-based training (dừng sau 1 giờ)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse
from datetime import datetime, timedelta
import time

from data_processing import prepare_data
from transformer import create_transformer_model
from utils import get_device, WarmupScheduler, save_checkpoint, plot_training_history, calculate_perplexity, count_parameters, train_epoch, validate








def train_model(config, data=None):
    """Huấn luyện mô hình với cấu hình tối ưu cho 1 giờ"""
    
    # Device
    device = get_device()
    print(f"Sử dụng device: {device}")
    
    # Mixed precision training
    use_amp = config.get('use_amp', True) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("✓ Sử dụng Mixed Precision Training (AMP)")
    
    # Chuẩn bị dữ liệu
    print("\n" + "="*60)
    print("CHUẨN BỊ DỮ LIỆU")
    print("="*60)
    
    if data is None:
        data = prepare_data(
            data_dir=config['data_dir'],
            max_len=config['max_len'],
            min_freq=config['min_freq'],
            batch_size=config['batch_size'],
            iwslt_dir=config.get('iwslt_dir', 'iwslt_en_vi'),
            num_workers=config.get('num_workers', 4 if torch.cuda.is_available() else 0)
        )
    else:
        print("✓ Sử dụng dữ liệu đã được nạp sẵn.")
    
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    src_vocab = data['src_vocab']
    tgt_vocab = data['tgt_vocab']
    
    # Tạo mô hình
    print("\n" + "="*60)
    print("KHỞI TẠO MÔ HÌNH")
    print("="*60)
    model = create_transformer_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_encoder_layers=config['n_encoder_layers'],
        n_decoder_layers=config['n_decoder_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        pad_idx=src_vocab.word2idx[src_vocab.PAD_TOKEN]
    )
    
    model = model.to(device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"✓ Đã phát hiện {torch.cuda.device_count()} GPUs!")
        print(f"  -> Đang kích hoạt chế độ DataParallel để chạy song song trên tất cả GPU.")
        model = nn.DataParallel(model)
    else:
        print(f"  -> Chạy trên 1 GPU/CPU ({device})")
        
    num_params = count_parameters(model)
    print(f"Số lượng tham số: {num_params:,}")
    
    # Loss function (ignore padding tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.word2idx[src_vocab.PAD_TOKEN])
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.get('weight_decay', 0.0001)
    )
    
    # Learning rate scheduler với warmup
    scheduler = WarmupScheduler(
        optimizer,
        d_model=config['d_model'],
        warmup_steps=config['warmup_steps']
        base_lr=config['learning_rate'] 
    )
    
    # Tạo thư mục lưu mô hình
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Training history
    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Time-based training
    max_time_seconds = config.get('max_time_hours', 1.0) * 3600
    start_time = time.time()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if config.get('resume_from'):
        print(f"\nĐang tổi checkpoint từ {config['resume_from']}...")
        if os.path.exists(config['resume_from']):
            from utils import load_checkpoint
            # Load checkpoint
            loaded_epoch, loaded_loss, loaded_config = load_checkpoint(
                model, optimizer, scheduler, config['resume_from'], device,
                reset_scheduler=config.get('reset_scheduler', False)
            )
            start_epoch = loaded_epoch
            print(f"✓ Đã khôi phục training từ epoch {start_epoch}")
            
            # Cập nhật config từ checkpoint nếu có (để đảm bảo nhất quán)
            if loaded_config:
                print("Lưu ý: Đang sử dụng cấu hình gốc từ checkpoint (có thể khác với tham số dòng lệnh)")
        else:
            print(f"⚠️ Không tìm thấy checkpoint tại {config['resume_from']}. Bắt đầu train mới.")
    
    print("\n" + "="*60)
    print("BẮT ĐẦU HUẤN LUYỆN")
    print(f"Giới hạn thời gian: {config.get('max_time_hours', 1.0):.2f} giờ")
    print("="*60)
    
                            
    def cleanup_checkpoints(model_dir, keep_n):
        """Xóa bớt các checkpoint cũ, chỉ giữ lại N cái mới nhất"""
        checkpoints = []
        for f in os.listdir(model_dir):
            if f.startswith('checkpoint_epoch_') and f.endswith('.pt'):
                path = os.path.join(model_dir, f)
                # Parse epoch number from filename
                try:
                    epoch_num = int(f.split('_')[-1].split('.')[0])
                    checkpoints.append((epoch_num, path))
                except ValueError:
                    continue
        
        # Sort by epoch (newest first)
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Remove old checkpoints
        if len(checkpoints) > keep_n:
            print(f"\nĐang dọn dẹp checkpoint (giữ lại {keep_n} bản mới nhất)...")
            for _, path in checkpoints[keep_n:]:
                os.remove(path)
                print(f"  - Đã xóa: {os.path.basename(path)}")

    # Training loop
    epoch = start_epoch
    should_stop = False
    
    try:
        while not should_stop:
            epoch += 1
            
            # Kiểm tra thời gian trước khi bắt đầu epoch
            elapsed = time.time() - start_time
            if elapsed >= max_time_seconds:
                print(f"\nĐã đạt giới hạn thời gian ({config.get('max_time_hours', 1.0):.2f} giờ)")
                break
            
            print(f"\nEpoch {epoch}")
            print("-" * 60)
            
            # Train
            remaining_time = max_time_seconds - elapsed
            train_loss, time_stopped = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, 
                config['clip_grad'], scaler, remaining_time, start_time
            )
            
            if time_stopped:
                print("\nĐừng lo: Đã hết thời gian (Stop by Time Limit). Đang lưu model hiện tại...")
                should_stop = True
                
                # Lưu final model
                final_model_path = os.path.join(config['model_dir'], 'final_model.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, train_loss, final_model_path, config=config)
                
                # Nếu chưa có best model (chưa xong epoch 1), dùng luôn final model làm best
                best_model_path = os.path.join(config['model_dir'], 'best_model.pt')
                if not os.path.exists(best_model_path):
                    print("Chưa có best model, sao chép final model thành best model để đánh giá.")
                    import shutil
                    shutil.copy(final_model_path, best_model_path)

            
            train_ppl = calculate_perplexity(train_loss)
            train_losses.append(train_loss)
            train_perplexities.append(train_ppl)
            
            # Validate (chỉ validate nếu còn thời gian)
            if not should_stop:
                elapsed = time.time() - start_time
                if elapsed < max_time_seconds * 0.95:  # Dành 5% thời gian cho validation
                    val_loss = validate(model, val_loader, criterion, device)
                    val_ppl = calculate_perplexity(val_loss)
                    val_losses.append(val_loss)
                    val_perplexities.append(val_ppl)
                    
                    print(f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
                    print(f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
                    print(f"Learning Rate: {scheduler.get_lr():.2e}")
                    print(f"Thời gian đã dùng: {(time.time() - start_time)/3600:.2f} giờ")
                    
                    # Lưu best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_path = os.path.join(config['model_dir'], 'best_model.pt')
                        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_model_path, config=config)
                    else:
                        patience_counter += 1
                    
                    # Early stopping
                    if patience_counter >= config.get('patience', 10):
                        print(f"\nEarly stopping tại epoch {epoch}")
                        should_stop = True
                else:
                    print(f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
                    print(f"Learning Rate: {scheduler.get_lr():.2e}")
                    print(f"Thời gian đã dùng: {(time.time() - start_time)/3600:.2f} giờ")
                    # Lưu model cuối cùng
                    final_model_path = os.path.join(config['model_dir'], 'final_model.pt')
                    save_checkpoint(model, optimizer, scheduler, epoch, train_loss, final_model_path, config=config)
            
            # Lưu checkpoint định kỳ
            if epoch % config.get('save_every', 5) == 0 and not should_stop:
                checkpoint_path = os.path.join(config['model_dir'], f'checkpoint_epoch_{epoch}.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, 
                              val_losses[-1] if val_losses else train_loss, 
                              checkpoint_path, config=config)
                
                # Cleanup old checkpoints
                cleanup_checkpoints(config['model_dir'], config.get('keep_last_n', 5))

    except KeyboardInterrupt:
        print("\n\n⚠️ Đã ngắt thủ công (Ctrl+C)!")
        print("Đang lưu model hiện tại (interrupted_model.pt)...")
        interrupted_path = os.path.join(config['model_dir'], 'interrupted_model.pt')
        save_checkpoint(model, optimizer, scheduler, epoch, 0.0, interrupted_path, config=config)
        
        # Nếu chưa có best_model thì dùng luôn cái này
        best_model_path = os.path.join(config['model_dir'], 'best_model.pt')
        if not os.path.exists(best_model_path):
            print("Chưa có best model, dùng tạm model bị ngắt này để đánh giá.")
            import shutil
            shutil.copy(interrupted_path, best_model_path)
        
        # Vẫn cho phép vẽ đồ thị
        should_stop = True
    
    # Lưu đồ thị training history
    if len(train_losses) > 0:
        plot_training_history(
            train_losses, 
            val_losses if val_losses else train_losses, 
            train_perplexities, 
            val_perplexities if val_perplexities else train_perplexities,
            save_path=os.path.join(config['results_dir'], 'training_history.png')
        )
        
        # Lưu training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_perplexities': train_perplexities,
            'val_perplexities': val_perplexities
        }
        import pickle
        with open(os.path.join(config['results_dir'], 'training_history.pkl'), 'wb') as f:
            pickle.dump(history, f)
    
    print("\n" + "="*60)
    print("HOÀN THÀNH HUẤN LUYỆN")
    print("="*60)
    total_time = (time.time() - start_time) / 3600
    print(f"Tổng thời gian: {total_time:.2f} giờ")
    print(f"Tổng số epochs: {epoch}")
    if val_losses:
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"Best Validation Perplexity: {calculate_perplexity(best_val_loss):.2f}")
    
    if train_losses:
        print(f"Final Train Loss: {train_losses[-1]:.4f}")
        print(f"Final Train Perplexity: {train_perplexities[-1]:.2f}")
    else:
        print("Chưa hoàn thành epoch nào.")
    
    return model, data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer Model (Optimized for 1 hour)')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--iwslt_dir', type=str, default='iwslt_en_vi', help='IWSLT data directory')
    
    # Model hyperparameters (nhỏ hơn để train nhanh)
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension (reduced from 512)')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads (reduced from 8)')
    parser.add_argument('--n_encoder_layers', type=int, default=3, help='Number of encoder layers (reduced from 6)')
    parser.add_argument('--n_decoder_layers', type=int, default=3, help='Number of decoder layers (reduced from 6)')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward dimension (reduced from 2048)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (increased)')
    parser.add_argument('--max_len', type=int, default=64, help='Maximum sequence length (reduced from 128)')
    parser.add_argument('--min_freq', type=int, default=2, help='Minimum word frequency')
    parser.add_argument('--num_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='Warmup steps (reduced)')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    
    # Optimization
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--max_time_hours', type=float, default=1.0, help='Maximum training time in hours')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--keep_last_n', type=int, default=2, help='Keep only N latest checkpoints (default: 2)')
    parser.add_argument('--reset_scheduler', action='store_true', help='Reset optimizer and scheduler states when resuming')
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'model_dir': args.model_dir,
        'results_dir': args.results_dir,
        'iwslt_dir': args.iwslt_dir,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_encoder_layers': args.n_encoder_layers,
        'n_decoder_layers': args.n_decoder_layers,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'max_len': args.max_len,
        'min_freq': args.min_freq,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'warmup_steps': args.warmup_steps,
        'clip_grad': args.clip_grad,
        'patience': args.patience,
        'save_every': args.save_every,
        'weight_decay': args.weight_decay,
        'use_amp': args.use_amp,
        'num_workers': args.num_workers,
        'num_workers': args.num_workers,
        'max_time_hours': args.max_time_hours,
        'resume_from': args.resume_from,
        'keep_last_n': args.keep_last_n,
        'reset_scheduler': args.reset_scheduler
    }
    
    train_model(config)

