"""
Script chính để chạy toàn bộ pipeline:
1. Xử lý dữ liệu
2. Huấn luyện mô hình
3. Đánh giá mô hình
4. Tạo báo cáo
"""

import os
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='Transformer Seq2Seq Pipeline')
    parser.add_argument('--mode', type=str, choices=['data', 'train', 'eval', 'all'], 
                       default='all', help='Mode to run')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--model_path', type=str, default='models/best_model.pt', 
                       help='Path to model for evaluation')
    
    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_encoder_layers', type=int, default=6)
    parser.add_argument('--n_decoder_layers', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--min_freq', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=10)
    
    # Evaluation hyperparameters
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--decoding_mode', type=str, default='both', choices=['beam', 'greedy', 'both'])
    parser.add_argument('--max_time_hours', type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Tạo thư mục
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    data = None
    if args.mode in ['data', 'all']:
        print("="*60)
        print("BƯỚC 1: XỬ LÝ DỮ LIỆU")
        print("="*60)
        from data_processing import prepare_data
        data = prepare_data(
            data_dir=args.data_dir,
            max_len=args.max_len,
            min_freq=args.min_freq,
            batch_size=args.batch_size
        )
        print("\n✓ Hoàn thành xử lý dữ liệu!\n")
    
    if args.mode in ['train', 'all']:
        print("="*60)
        print("BƯỚC 2: HUẤN LUYỆN MÔ HÌNH")
        print("="*60)
        from train import train_model
        
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
            'save_every': args.save_every,
            'max_time_hours': args.max_time_hours
        }
        
        train_model(config, data=data)
        print("\n✓ Hoàn thành huấn luyện!\n")
    
    if args.mode in ['eval', 'all']:
        print("="*60)
        print("BƯỚC 3: ĐÁNH GIÁ MÔ HÌNH")
        print("="*60)
        from evaluate import evaluate
        
        config = {
            'model_path': args.model_path,
            'data_dir': args.data_dir,
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
            'beam_size': args.beam_size,
            'decoding_mode': args.decoding_mode
        }
        
        evaluate(config, data=data)
        print("\n✓ Hoàn thành đánh giá!\n")
    
    if args.mode == 'all':
        print("="*60)
        print("BƯỚC 4: TẠO BÁO CÁO")
        print("="*60)
        from report import create_comprehensive_report
        create_comprehensive_report(args.results_dir)
        print("\n✓ Hoàn thành tạo báo cáo!\n")
        
        print("="*60)
        print("HOÀN THÀNH TẤT CẢ CÁC BƯỚC!")
        print("="*60)
        print(f"Kết quả được lưu tại: {args.results_dir}")
        print(f"Mô hình được lưu tại: {args.model_dir}")


if __name__ == '__main__':
    main()

