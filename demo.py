"""
Script demo để test mô hình nhanh
"""

import torch
from data_processing import Vocabulary, tokenize
from transformer import create_transformer_model
from evaluate import BeamSearchDecoder, GreedyDecoder
from utils import get_device
import pickle
import os


def demo_translation(model_path='models/best_model.pt', 
                    data_dir='data',
                    sentences=['xin chào', 'tôi là sinh viên', 'hôm nay trời đẹp'],
                    decoder_type='beam',
                    beam_size=5,
                    d_model=256, n_heads=4, n_encoder_layers=3, n_decoder_layers=3, d_ff=1024):
    """
    Demo dịch một số câu
    """
    device = get_device()
    print(f"Sử dụng device: {device}")
    
    # Load vocabulary
    print("\nĐang tải vocabulary...")
    with open(os.path.join(data_dir, 'src_vocab.pkl'), 'rb') as f:
        src_vocab_data = pickle.load(f)
        src_vocab = Vocabulary()
        src_vocab.word2idx = src_vocab_data['word2idx']
        src_vocab.idx2word = src_vocab_data['idx2word']
    
    with open(os.path.join(data_dir, 'tgt_vocab.pkl'), 'rb') as f:
        tgt_vocab_data = pickle.load(f)
        tgt_vocab = Vocabulary()
        tgt_vocab.word2idx = tgt_vocab_data['word2idx']
        tgt_vocab.idx2word = tgt_vocab_data['idx2word']
    
    # Load checkpoint first to check for config
    print("Đang tải mô hình...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check for config in checkpoint
    config = checkpoint.get('config', None)
    if config:
        print("✓ Tìm thấy cấu hình trong checkpoint, đang sử dụng...")
        d_model = config.get('d_model', d_model)
        n_heads = config.get('n_heads', n_heads)
        n_encoder_layers = config.get('n_encoder_layers', n_encoder_layers)
        n_decoder_layers = config.get('n_decoder_layers', n_decoder_layers)
        d_ff = config.get('d_ff', d_ff)
    
    from transformer import Transformer
    
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        dropout=0.1,
        pad_idx=src_vocab.word2idx[src_vocab.PAD_TOKEN]
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✓ Đã tải mô hình thành công\n")
    
    # Chọn decoder
    if decoder_type == 'beam':
        decoder = BeamSearchDecoder(model, tgt_vocab, device, beam_size)
        print(f"Sử dụng Beam Search (beam_size={beam_size})")
    else:
        decoder = GreedyDecoder(model, tgt_vocab, device)
        print("Sử dụng Greedy Search")
    
    print("="*60)
    print("DỊCH CÁC CÂU")
    print("="*60)
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\n{i}. Câu tiếng Việt: {sentence}")
        
        # Tokenize và encode
        tokens = tokenize(sentence, 'vi')
        src_indices = [src_vocab.word2idx[src_vocab.SOS_TOKEN]] + \
                     src_vocab.encode(tokens) + \
                     [src_vocab.word2idx[src_vocab.EOS_TOKEN]]
        
        # Padding nếu cần
        max_len = 128
        if len(src_indices) < max_len:
            src_indices.extend([src_vocab.word2idx[src_vocab.PAD_TOKEN]] * 
                             (max_len - len(src_indices)))
        else:
            src_indices = src_indices[:max_len]
        
        # Chuyển thành tensor
        src = torch.tensor([src_indices], device=device)
        src_mask, _ = model.generate_mask(src)
        
        # Decode
        with torch.no_grad():
            pred_sequence = decoder.decode(src, src_mask)
        
        # Decode thành text
        pred_tokens = tgt_vocab.decode(pred_sequence)
        
        # Loại bỏ <sos> nếu có (thường xuất hiện ở đầu do beam search)
        if pred_tokens and pred_tokens[0] == tgt_vocab.SOS_TOKEN:
            pred_tokens = pred_tokens[1:]
            
        pred_text = ' '.join(pred_tokens)
        
        print(f"   Dịch sang tiếng Anh: {pred_text}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo Translation')
    parser.add_argument('--model_path', type=str, default='models/best_model.pt')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--decoder', type=str, choices=['beam', 'greedy'], default='beam')
    parser.add_argument('--beam_size', type=int, default=5)
    
    # Model architecture arguments
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_encoder_layers', type=int, default=3)
    parser.add_argument('--n_decoder_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=1024)
    
    parser.add_argument('--sentences', type=str, nargs='+', 
                       default=['xin chào', 'tôi là sinh viên', 'hôm nay trời đẹp'])
    
    args = parser.parse_args()
    
    demo_translation(
        model_path=args.model_path,
        data_dir=args.data_dir,
        sentences=args.sentences,
        decoder_type=args.decoder,
        beam_size=args.beam_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_ff=args.d_ff
    )

