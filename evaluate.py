"""
Script đánh giá mô hình Transformer
- Beam Search Decoding
- Greedy Search Decoding
- BLEU Score
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import argparse
from typing import List, Tuple
from collections import defaultdict

from data_processing import Vocabulary, TranslationDataset
from transformer import Transformer
from utils import get_device, load_checkpoint
from sacrebleu import BLEU


class BeamSearchDecoder:
    """
    Beam Search Decoder cho Transformer
    """
    
    def __init__(self, model, tgt_vocab, device, beam_size=5, max_len=128):
        self.model = model
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.beam_size = beam_size
        self.max_len = max_len
        self.sos_idx = tgt_vocab.word2idx[tgt_vocab.SOS_TOKEN]
        self.eos_idx = tgt_vocab.word2idx[tgt_vocab.EOS_TOKEN]
        self.pad_idx = tgt_vocab.word2idx[tgt_vocab.PAD_TOKEN]
    
    def decode(self, src, src_mask):
        """
        Beam Search Decoding
        Args:
            src: [1, src_len]
            src_mask: [1, 1, 1, src_len]
        Returns:
            best_sequence: List of token indices
        """
        self.model.eval()
        
        batch_size = src.size(0)
        src_len = src.size(1)
        
        # Encode source
        with torch.no_grad():
            enc_output = self.model.encode(src)
        
        # Khởi tạo beam: (sequence, score, hidden_state)
        beams = [([self.sos_idx], 0.0)]
        finished = []
        
        for step in range(self.max_len):
            candidates = []
            
            for sequence, score in beams:
                # Nếu đã kết thúc, thêm vào finished
                if sequence[-1] == self.eos_idx:
                    finished.append((sequence, score / len(sequence)))
                    continue
                
                # Tạo input cho decoder
                tgt_input = torch.tensor([sequence], device=self.device)
                tgt_len = tgt_input.size(1)
                
                # Tạo target mask
                tgt_mask = self.model.generate_mask(tgt_input, tgt_input)[1]
                
                # Decode
                with torch.no_grad():
                    dec_output = self.model.decode(tgt_input, enc_output, src_mask, tgt_mask)
                    logits = self.model.output_projection(dec_output)
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                
                # Lấy top-k candidates
                top_k_log_probs, top_k_indices = torch.topk(log_probs, self.beam_size)
                
                for i in range(self.beam_size):
                    token_idx = top_k_indices[0, i].item()
                    log_prob = top_k_log_probs[0, i].item()
                    new_sequence = sequence + [token_idx]
                    new_score = score + log_prob
                    candidates.append((new_sequence, new_score))
            
            # Chọn top beam_size candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:self.beam_size]
            
            # Nếu tất cả beams đã kết thúc
            if all(seq[-1] == self.eos_idx for seq, _ in beams):
                break
        
        # Thêm các beams còn lại vào finished
        for sequence, score in beams:
            if sequence[-1] == self.eos_idx:
                finished.append((sequence, score / len(sequence)))
        
        # Chọn sequence tốt nhất
        if finished:
            finished.sort(key=lambda x: x[1], reverse=True)
            best_sequence = finished[0][0]
        else:
            # Nếu không có sequence kết thúc, chọn sequence dài nhất
            beams.sort(key=lambda x: len(x[0]), reverse=True)
            best_sequence = beams[0][0]
        
        return best_sequence


class GreedyDecoder:
    """
    Greedy Search Decoder
    """
    
    def __init__(self, model, tgt_vocab, device, max_len=128):
        self.model = model
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.max_len = max_len
        self.sos_idx = tgt_vocab.word2idx[tgt_vocab.SOS_TOKEN]
        self.eos_idx = tgt_vocab.word2idx[tgt_vocab.EOS_TOKEN]
        self.pad_idx = tgt_vocab.word2idx[tgt_vocab.PAD_TOKEN]
    
    def decode(self, src, src_mask):
        """
        Greedy Decoding
        Args:
            src: [1, src_len]
            src_mask: [1, 1, 1, src_len]
        Returns:
            sequence: List of token indices
        """
        self.model.eval()
        
        # Encode source
        with torch.no_grad():
            enc_output = self.model.encode(src)
        
        # Khởi tạo với SOS token
        sequence = [self.sos_idx]
        
        for step in range(self.max_len):
            # Tạo input cho decoder
            tgt_input = torch.tensor([sequence], device=self.device)
            tgt_mask = self.model.generate_mask(tgt_input, tgt_input)[1]
            
            # Decode
            with torch.no_grad():
                dec_output = self.model.decode(tgt_input, enc_output, src_mask, tgt_mask)
                logits = self.model.output_projection(dec_output)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            
            sequence.append(next_token)
            
            # Dừng nếu gặp EOS
            if next_token == self.eos_idx:
                break
        
        return sequence


def evaluate_model(model, test_loader, src_vocab, tgt_vocab, device, 
                  decoder_type='beam', beam_size=5, max_len=128):
    """
    Đánh giá mô hình trên test set
    """
    # Chọn decoder
    if decoder_type == 'beam':
        decoder = BeamSearchDecoder(model, tgt_vocab, device, beam_size, max_len)
    else:
        decoder = GreedyDecoder(model, tgt_vocab, device, max_len)
    
    predictions = []
    references = []
    
    print(f"\nĐang đánh giá với {decoder_type} decoding...")
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            src = batch['src'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            
            # Tạo source mask
            src_mask, _ = model.generate_mask(src)
            
            # Decode từng câu trong batch
            for i in range(src.size(0)):
                src_single = src[i:i+1]
                src_mask_single = src_mask[i:i+1]
                
                # Decode
                pred_sequence = decoder.decode(src_single, src_mask_single)
                
                # Chuyển thành text
                pred_tokens = tgt_vocab.decode(pred_sequence)
                pred_text = ' '.join(pred_tokens)
                
                # Lấy reference
                ref_sequence = tgt_output[i].cpu().tolist()
                ref_tokens = tgt_vocab.decode(ref_sequence)
                ref_text = ' '.join(ref_tokens)
                
                predictions.append(pred_text)
                references.append([ref_text])  # BLEU expects list of references
    
    return predictions, references


def calculate_bleu_score(predictions: List[str], references: List[List[str]]):
    """
    Tính BLEU score
    """
    bleu = BLEU()
    score = bleu.corpus_score(predictions, references)
    return score.score, score


def evaluate(config, data=None):
    """Đánh giá mô hình"""
    
    # Device
    device = get_device()
    print(f"Sử dụng device: {device}")
    
    if data is None:
        # Load vocabulary manually if data not provided
        import pickle
        with open(os.path.join(config['data_dir'], 'src_vocab.pkl'), 'rb') as f:
            src_vocab_data = pickle.load(f)
            src_vocab = Vocabulary()
            src_vocab.word2idx = src_vocab_data['word2idx']
            src_vocab.idx2word = src_vocab_data['idx2word']
            src_vocab.word_count = src_vocab_data['word_count']
        
        with open(os.path.join(config['data_dir'], 'tgt_vocab.pkl'), 'rb') as f:
            tgt_vocab_data = pickle.load(f)
            tgt_vocab = Vocabulary()
            tgt_vocab.word2idx = tgt_vocab_data['word2idx']
            tgt_vocab.idx2word = tgt_vocab_data['idx2word']
            tgt_vocab.word_count = tgt_vocab_data['word_count']
        
        # Load test data
        from data_processing import prepare_data
        data_res = prepare_data(
            data_dir=config['data_dir'],
            max_len=config['max_len'],
            min_freq=config['min_freq'],
            batch_size=config['batch_size']
        )
        test_loader = data_res['test_loader']
    else:
        print("✓ Sử dụng dữ liệu đã được nạp sẵn.")
        src_vocab = data['src_vocab']
        tgt_vocab = data['tgt_vocab']
        test_loader = data['test_loader']
    
    # Load model
    print(f"\nĐang tải mô hình từ {config['model_path']}...")
    model = Transformer(
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
    
    checkpoint = torch.load(config['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("✓ Đã tải mô hình")
    
    # Đánh giá với Beam Search
    beam_predictions = []
    if config.get('decoding_mode', 'both') in ['beam', 'both']:
        print("\n" + "="*60)
        print(f"ĐÁNH GIÁ VỚI BEAM SEARCH (Beam Size: {config['beam_size']})")
        print("="*60)
        beam_predictions, references = evaluate_model(
            model, test_loader, src_vocab, tgt_vocab, device,
            decoder_type='beam', beam_size=config['beam_size'], max_len=config['max_len']
        )
        beam_bleu_score, beam_bleu = calculate_bleu_score(beam_predictions, references)
        print(f"BLEU Score (Beam Search): {beam_bleu_score:.2f}")
        print(f"Chi tiết: {beam_bleu}")
    
    # Đánh giá với Greedy Search
    greedy_predictions = []
    if config.get('decoding_mode', 'both') in ['greedy', 'both']:
        print("\n" + "="*60)
        print("ĐÁNH GIÁ VỚI GREEDY SEARCH")
        print("="*60)
        # Nếu chưa chạy beam search thì reference chưa được tạo, cần tạo lại (bên trong evaluate_model vẫn trả về)
        greedy_predictions, ref_greedy = evaluate_model(
            model, test_loader, src_vocab, tgt_vocab, device,
            decoder_type='greedy', max_len=config['max_len']
        )
        if not references: # Nếu chưa có references từ bước beam search
            references = ref_greedy
            
        greedy_bleu_score, greedy_bleu = calculate_bleu_score(greedy_predictions, references)
        print(f"BLEU Score (Greedy Search): {greedy_bleu_score:.2f}")
        print(f"Chi tiết: {greedy_bleu}")
    
    # Lưu kết quả
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Lưu predictions
    with open(os.path.join(config['results_dir'], 'beam_predictions.txt'), 'w', encoding='utf-8') as f:
        for pred in beam_predictions:
            f.write(pred + '\n')
    
    with open(os.path.join(config['results_dir'], 'greedy_predictions.txt'), 'w', encoding='utf-8') as f:
        for pred in greedy_predictions:
            f.write(pred + '\n')
    
    with open(os.path.join(config['results_dir'], 'references.txt'), 'w', encoding='utf-8') as f:
        for ref in references:
            f.write(ref[0] + '\n')
    
    # Lưu báo cáo
    report_path = os.path.join(config['results_dir'], 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH\n")
        f.write("="*60 + "\n\n")
        f.write(f"Mô hình: {config['model_path']}\n")
        f.write(f"Beam Size: {config['beam_size']}\n")
        f.write(f"Max Length: {config['max_len']}\n\n")
        f.write("KẾT QUẢ:\n")
        f.write(f"BLEU Score (Beam Search): {beam_bleu_score:.2f}\n")
        f.write(f"BLEU Score (Greedy Search): {greedy_bleu_score:.2f}\n\n")
        f.write("CHI TIẾT BEAM SEARCH:\n")
        f.write(str(beam_bleu) + "\n\n")
        f.write("CHI TIẾT GREEDY SEARCH:\n")
        f.write(str(greedy_bleu) + "\n")
    
    print(f"\n✓ Đã lưu kết quả tại {config['results_dir']}")
    print(f"✓ Đã lưu báo cáo tại {report_path}")
    
    # In một số ví dụ
    print("\n" + "="*60)
    print("MỘT SỐ VÍ DỤ DỊCH")
    print("="*60)
    for i in range(min(5, len(beam_predictions))):
        print(f"\nVí dụ {i+1}:")
        print(f"Reference: {references[i][0]}")
        print(f"Beam Search: {beam_predictions[i]}")
        print(f"Greedy Search: {greedy_predictions[i]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Transformer Model')
    parser.add_argument('--model_path', type=str, default='models/best_model.pt', 
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
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
    parser.add_argument('--beam_size', type=int, default=5, help='Beam search size')
    parser.add_argument('--decoding_mode', type=str, default='both', choices=['beam', 'greedy', 'both'],
                       help='Decoding mode: beam, greedy, or both')
    
    args = parser.parse_args()
    
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
        'beam_size': args.beam_size
    }
    
    evaluate(config)

