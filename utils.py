"""
Các hàm tiện ích cho training và evaluation
"""

import torch
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, 
                clip_grad=1.0, scaler=None, max_time=None, start_time=None):
    """Huấn luyện một epoch với mixed precision"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        # Kiểm tra thời gian
        if max_time and start_time:
            elapsed = time.time() - start_time
            if elapsed >= max_time:
                print(f"\\nĐã đạt giới hạn thời gian ({max_time/3600:.2f} giờ)")
                return total_loss / max(num_batches, 1), True  # Trả về True để báo dừng
        
        src = batch['src'].to(device, non_blocking=True)
        tgt_input = batch['tgt_input'].to(device, non_blocking=True)
        tgt_output = batch['tgt_output'].to(device, non_blocking=True)
        
        # Forward pass với mixed precision
        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision training
            with torch.amp.autocast('cuda'):
                output = model(src, tgt_input)
                output = output.view(-1, output.size(-1))
                tgt_output_flat = tgt_output.view(-1)
                loss = criterion(output, tgt_output_flat)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            output = model(src, tgt_input)
            output = output.view(-1, output.size(-1))
            tgt_output_flat = tgt_output.view(-1)
            loss = criterion(output, tgt_output_flat)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'lr': f'{scheduler.get_lr():.2e}',
            'avg_loss': f'{total_loss/num_batches:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    return avg_loss, False  # False = không dừng


def validate(model, val_loader, criterion, device):
    """Đánh giá trên validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            src = batch['src'].to(device, non_blocking=True)
            tgt_input = batch['tgt_input'].to(device, non_blocking=True)
            tgt_output = batch['tgt_output'].to(device, non_blocking=True)
            
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


def get_device():
    """Lấy device (GPU nếu có, ngược lại CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WarmupScheduler:
    """
    Learning Rate Scheduler với warmup
    LR = warmup_factor * min(step / warmup_steps, 1) * base_lr
    """
    
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.base_lr = None
        self.scale_factor = d_model ** -0.5
    
    def step(self):
        """Cập nhật learning rate"""
        self.current_step += 1
        
        if self.base_lr is None:
            self.base_lr = self.optimizer.param_groups[0]['lr']
        
        # Warmup schedule
        lr = self.base_lr * self.scale_factor * min(
            self.current_step ** (-0.5),
            self.current_step * self.warmup_steps ** (-1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        """Lấy learning rate hiện tại"""
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        """Lưu trạng thái scheduler"""
        return {
            'current_step': self.current_step,
            'base_lr': self.base_lr,
            'warmup_steps': self.warmup_steps,
            'd_model': self.d_model
        }

    def load_state_dict(self, state_dict):
        """Tải trạng thái scheduler"""
        self.current_step = state_dict['current_step']
        self.base_lr = state_dict['base_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.d_model = state_dict['d_model']


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, config=None):
    """Lưu checkpoint và config (nếu có)"""
    
    # Unwrap model nếu là DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
        
    state = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    if config:
        state['config'] = config
        
    torch.save(state, path)
    print(f"✓ Đã lưu checkpoint tại {path}")


def load_checkpoint(model, optimizer, scheduler, path, device, reset_scheduler=False):
    """Tải checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    # Xử lý tương thích key giữa DataParallel và model thường
    state_dict = checkpoint['model_state_dict']
    
    # Nếu model hiện tại là DataParallel
    if isinstance(model, torch.nn.DataParallel):
        # Nếu checkpoint không có 'module.' (do lưu ở chế độ 1 GPU hoặc đã unwrap) -> load vào model.module
        if not list(state_dict.keys())[0].startswith('module.'):
            model.module.load_state_dict(state_dict)
        else:
             # Nếu checkpoint có 'module.' -> load thẳng
            model.load_state_dict(state_dict)
    else:
        # Nếu model hiện tại là Single GPU/CPU
        # Nếu checkpoint có 'module.' (legacy save từ DataParallel) -> remove prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    if optimizer and not reset_scheduler:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif reset_scheduler:
        print("➤ Đã reset optimizer state.")
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        if not reset_scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("➤ Đã reset scheduler state (Learning Rate sẽ bắt đầu lại từ đầu).")
            
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    config = checkpoint.get('config', None)
    
    print(f"✓ Đã tải checkpoint từ epoch {epoch}, loss: {loss:.4f}")
    if config:
         print(f"  - Config tìm thấy: d_model={config.get('d_model')}, layers={config.get('n_encoder_layers')}")
         
    return epoch, loss, config


def plot_training_history(train_losses, val_losses, train_perplexities=None, 
                         val_perplexities=None, save_path='results/training_history.png'):
    """Vẽ đồ thị loss và perplexity"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(train_losses, label='Train Loss', color='blue')
    axes[0].plot(val_losses, label='Validation Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot perplexity
    if train_perplexities and val_perplexities:
        axes[1].plot(train_perplexities, label='Train Perplexity', color='blue')
        axes[1].plot(val_perplexities, label='Validation Perplexity', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Perplexity')
        axes[1].set_title('Training and Validation Perplexity')
        axes[1].legend()
        axes[1].grid(True)
    else:
        # Tính perplexity từ loss nếu không có
        train_ppl = [np.exp(loss) for loss in train_losses]
        val_ppl = [np.exp(loss) for loss in val_losses]
        axes[1].plot(train_ppl, label='Train Perplexity', color='blue')
        axes[1].plot(val_ppl, label='Validation Perplexity', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Perplexity')
        axes[1].set_title('Training and Validation Perplexity')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu đồ thị tại {save_path}")
    plt.close()


def calculate_perplexity(loss):
    """Tính perplexity từ loss"""
    return np.exp(loss)


def count_parameters(model):
    """Đếm số lượng tham số của mô hình"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

