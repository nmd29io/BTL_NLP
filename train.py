"""
Script huấn luyện mô hình Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse
from datetime import datetime

from data_processing import prepare_data
from transformer import create_transformer_model
from utils import get_device, WarmupScheduler, save_checkpoint, plot_training_history, calculate_perplexity, count_parameters


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, clip_grad=1.0):
    """Huấn luyện một epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt_input)
        
        # Reshape cho loss calculation
        output = output.view(-1, output.size(-1))
        tgt_output = tgt_output.view(-1)
        
        # Tính loss (ignore padding tokens)
        loss = criterion(output, tgt_output)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_lr():.2e}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Đánh giá trên validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            
            # Forward pass
            output = model(src, tgt_input)
            
            # Reshape cho loss calculation
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.view(-1)
            
            # Tính loss
            loss = criterion(output, tgt_output)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train_model(config):
    """Huấn luyện mô hình với cấu hình cho trước"""
    
    # Device
    device = get_device()
    print(f"Sử dụng device: {device}")
    
    # Chuẩn bị dữ liệu
    print("\n" + "="*60)
    print("CHUẨN BỊ DỮ LIỆU")
    print("="*60)
    data = prepare_data(
        data_dir=config['data_dir'],
        max_len=config['max_len'],
        min_freq=config['min_freq'],
        batch_size=config['batch_size']
    )
    
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
    num_params = count_parameters(model)
    print(f"Số lượng tham số: {num_params:,}")
    
    # Loss function (ignore padding tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.word2idx[src_vocab.PAD_TOKEN])
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # Learning rate scheduler với warmup
    scheduler = WarmupScheduler(
        optimizer,
        d_model=config['d_model'],
        warmup_steps=config['warmup_steps']
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
    
    print("\n" + "="*60)
    print("BẮT ĐẦU HUẤN LUYỆN")
    print("="*60)
    
    # Training loop
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, config['clip_grad'])
        train_ppl = calculate_perplexity(train_loss)
        train_losses.append(train_loss)
        train_perplexities.append(train_ppl)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_ppl = calculate_perplexity(val_loss)
        val_losses.append(val_loss)
        val_perplexities.append(val_ppl)
        
        print(f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
        print(f"Learning Rate: {scheduler.get_lr():.2e}")
        
        # Lưu best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(config['model_dir'], 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping tại epoch {epoch}")
            break
        
        # Lưu checkpoint định kỳ
        if epoch % config['save_every'] == 0:
            checkpoint_path = os.path.join(config['model_dir'], f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
    
    # Lưu đồ thị training history
    plot_training_history(
        train_losses, val_losses, train_perplexities, val_perplexities,
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
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Perplexity: {calculate_perplexity(best_val_loss):.2f}")
    
    return model, data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer Model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--n_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--min_freq', type=int, default=2, help='Minimum word frequency')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Warmup steps')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'model_dir': args.model_dir,
        'results_dir': args.results_dir,
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
        'save_every': args.save_every
    }
    
    train_model(config)

