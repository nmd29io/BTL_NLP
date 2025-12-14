"""
Các hàm tiện ích cho training và evaluation
"""

import torch
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os


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
    
    def step(self):
        """Cập nhật learning rate"""
        self.current_step += 1
        
        if self.base_lr is None:
            self.base_lr = self.optimizer.param_groups[0]['lr']
        
        # Warmup schedule
        lr = self.base_lr * min(
            self.current_step ** (-0.5),
            self.current_step * self.warmup_steps ** (-1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        """Lấy learning rate hiện tại"""
        return self.optimizer.param_groups[0]['lr']


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Lưu checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"✓ Đã lưu checkpoint tại {path}")


def load_checkpoint(model, optimizer, path, device):
    """Tải checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"✓ Đã tải checkpoint từ epoch {epoch}, loss: {loss:.4f}")
    return epoch, loss


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

