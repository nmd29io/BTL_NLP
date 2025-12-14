# train_simple.py
"""
Script training đơn giản, dễ theo dõi
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import json
from tqdm import tqdm

from config import SMALL_CONFIG, ModelConfig
from transformer import Transformer
from data_processing import prepare_data

class SimpleTrainer:
    """Trainer đơn giản, dễ debug"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Device: {self.device}")
        
        # Load data
        print("\n📁 Loading data...")
        self.data = prepare_data(
            data_dir=config.data_dir,
            max_len=config.max_len,
            batch_size=config.batch_size,
            min_freq=config.min_freq
        )
        
        self.src_vocab = self.data['src_vocab']
        self.tgt_vocab = self.data['tgt_vocab']
        self.train_loader = self.data['train_loader']
        self.val_loader = self.data['val_loader']
        
        # Create model
        print("\n🤖 Creating model...")
        self.model = Transformer(
            src_vocab_size=len(self.src_vocab),
            tgt_vocab_size=len(self.tgt_vocab),
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_encoder_layers=config.n_encoder_layers,
            n_decoder_layers=config.n_decoder_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            pad_idx=self.src_vocab.word2idx[self.src_vocab.PAD_TOKEN]
        ).to(self.device)
        
        print(f"✅ Model parameters: {self.count_parameters():,}")
        
        # Optimizer & Loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.src_vocab.word2idx[self.src_vocab.PAD_TOKEN]
        )
        
        # Scheduler đơn giản
        self.scheduler = self.create_scheduler()
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_ppl': [],
            'val_ppl': [],
            'learning_rates': []
        }
        
        # Create directories
        os.makedirs(config.model_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def create_scheduler(self):
        """Tạo scheduler đơn giản (no warmup để dễ debug)"""
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)
    
    def train_epoch(self):
        """Train một epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            src = batch['src'].to(self.device)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input)
            
            # Reshape cho loss
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.view(-1)
            
            # Loss
            loss = self.criterion(output, tgt_output)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
            
            # Update
            self.optimizer.step()
            
            # Stats
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            avg_loss = total_loss / num_batches
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'avg': f'{avg_loss:.3f}',
                'lr': f'{current_lr:.2e}'
            })
        
        return total_loss / num_batches
    
    def validate(self):
        """Validation"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                
                output = self.model(src, tgt_input)
                output = output.view(-1, output.size(-1))
                tgt_output = tgt_output.view(-1)
                
                loss = self.criterion(output, tgt_output)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def calculate_perplexity(self, loss):
        """Tính perplexity từ loss"""
        return torch.exp(torch.tensor(loss)).item()
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Lưu checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': {
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'n_encoder_layers': self.config.n_encoder_layers,
                'n_decoder_layers': self.config.n_decoder_layers,
                'd_ff': self.config.d_ff,
                'learning_rate': self.config.learning_rate
            }
        }
        
        if is_best:
            path = os.path.join(self.config.model_dir, 'best_model.pt')
            torch.save(checkpoint, path)
            print(f"💾 Saved best model to {path}")
        else:
            path = os.path.join(self.config.model_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, path)
    
    def train(self, max_epochs=10):
        """Training loop chính"""
        print("\n" + "="*50)
        print("🚀 STARTING TRAINING")
        print("="*50)
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        try:
            for epoch in range(1, max_epochs + 1):
                print(f"\n📊 EPOCH {epoch}/{max_epochs}")
                print("-" * 40)
                
                # Train
                train_loss = self.train_epoch()
                train_ppl = self.calculate_perplexity(train_loss)
                
                # Validate
                val_loss = self.validate()
                val_ppl = self.calculate_perplexity(val_loss)
                
                # Update scheduler
                self.scheduler.step()
                
                # Save history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_ppl'].append(train_ppl)
                self.history['val_ppl'].append(val_ppl)
                self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                
                # Print stats
                print(f"\n📈 Epoch {epoch} Summary:")
                print(f"  Train Loss: {train_loss:.4f} | PPL: {train_ppl:.2f}")
                print(f"  Val Loss:   {val_loss:.4f} | PPL: {val_ppl:.2f}")
                print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
                print(f"  Time elapsed: {(time.time() - start_time)/60:.1f} min")
                
                # Save checkpoint
                if epoch % self.config.save_every == 0:
                    self.save_checkpoint(epoch, val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, is_best=True)
                    print(f"🎉 New best model! Loss: {val_loss:.4f}")
                
                # Early stopping
                if epoch > self.config.patience:
                    recent_losses = self.history['val_loss'][-self.config.patience:]
                    if min(recent_losses) > best_val_loss:
                        print(f"🛑 Early stopping at epoch {epoch}")
                        break
                
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted!")
        
        # Save final model
        self.save_checkpoint(epoch, val_loss)
        
        # Save history
        history_path = os.path.join(self.config.results_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Plot
        self.plot_training()
        
        total_time = (time.time() - start_time) / 60
        print(f"\n✅ Training completed in {total_time:.1f} minutes")
        
        return self.history
    
    def plot_training(self):
        """Vẽ đồ thị training"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Loss
            axes[0].plot(self.history['train_loss'], label='Train', marker='o')
            axes[0].plot(self.history['val_loss'], label='Val', marker='s')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training & Validation Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # Perplexity
            axes[1].plot(self.history['train_ppl'], label='Train', marker='o')
            axes[1].plot(self.history['val_ppl'], label='Val', marker='s')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Perplexity')
            axes[1].set_title('Perplexity')
            axes[1].legend()
            axes[1].grid(True)
            
            # Learning Rate
            axes[2].plot(self.history['learning_rates'], marker='o', color='green')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_title('Learning Rate Schedule')
            axes[2].grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(self.config.results_dir, 'training_plot.png')
            plt.savefig(plot_path, dpi=150)
            plt.close()
            
            print(f"📊 Plot saved to {plot_path}")
            
        except ImportError:
            print("⚠️ Matplotlib not installed, skipping plot")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Transformer Training')
    
    # Model args
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_encoder_layers', type=int, default=3)
    parser.add_argument('--n_decoder_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_dir', type=str, default='models_simple')
    parser.add_argument('--results_dir', type=str, default='results_simple')
    
    args = parser.parse_args()
    
    # Convert args to config
    config = TrainingConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_len=args.max_len,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        clip_grad=args.clip_grad,
        max_epochs=args.max_epochs,
        patience=args.patience,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        results_dir=args.results_dir
    )
    
    # Create trainer and train
    trainer = SimpleTrainer(config)
    trainer.train(max_epochs=config.max_epochs)

if __name__ == '__main__':
    main()