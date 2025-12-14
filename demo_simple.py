# demo_simple.py
"""
Demo đơn giản để test model
"""
import torch
import pickle
import os
from data_processing import Vocabulary, tokenize
from transformer import Transformer
from config import ModelConfig

def load_vocab(data_dir):
    """Load vocabulary"""
    with open(os.path.join(data_dir, 'src_vocab.pkl'), 'rb') as f:
        src_data = pickle.load(f)
        src_vocab = Vocabulary()
        src_vocab.word2idx = src_data['word2idx']
        src_vocab.idx2word = src_data['idx2word']
    
    with open(os.path.join(data_dir, 'tgt_vocab.pkl'), 'rb') as f:
        tgt_data = pickle.load(f)
        tgt_vocab = Vocabulary()
        tgt_vocab.word2idx = tgt_data['word2idx']
        tgt_vocab.idx2word = tgt_data['idx2word']
    
    return src_vocab, tgt_vocab

def greedy_decode(model, src, src_vocab, tgt_vocab, device, max_len=50):
    """Greedy decoding đơn giản"""
    model.eval()
    
    # Tạo mask
    src_mask = (src != src_vocab.word2idx[src_vocab.PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
    
    # Encode
    with torch.no_grad():
        enc_output = model.encode(src, src_mask)
    
    # Decode từng token
    batch_size = src.size(0)
    tgt = torch.ones(batch_size, 1).fill_(tgt_vocab.word2idx[tgt_vocab.SOS_TOKEN]).long().to(device)
    
    for i in range(max_len):
        tgt_mask = model.make_tgt_mask(tgt)
        
        with torch.no_grad():
            dec_output = model.decode(tgt, enc_output, src_mask, tgt_mask)
            output = model.output_projection(dec_output[:, -1:, :])
            next_token = output.argmax(-1)
        
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # Dừng nếu tất cả sentences đã có EOS
        if (next_token == tgt_vocab.word2idx[tgt_vocab.EOS_TOKEN]).all():
            break
    
    return tgt

def demo():
    """Chạy demo"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load vocab
    print("Loading vocab...")
    src_vocab, tgt_vocab = load_vocab('data')
    
    # Load checkpoint
    checkpoint_path = 'models_simple/best_model.pt'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'models/best_model.pt'
    
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Lấy config từ checkpoint
    config = checkpoint.get('config', {})
    print(f"Model config: {config}")
    
    # Tạo model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 4),
        n_encoder_layers=config.get('n_encoder_layers', 3),
        n_decoder_layers=config.get('n_decoder_layers', 3),
        d_ff=config.get('d_ff', 1024),
        dropout=0.1,
        pad_idx=src_vocab.word2idx[src_vocab.PAD_TOKEN]
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Các câu test
    test_sentences = [
        "xin chào",
        "tôi là sinh viên", 
        "hôm nay trời đẹp",
        "bạn có khỏe không",
        "cảm ơn bạn"
    ]
    
    print("\n" + "="*60)
    print("DEMO TRANSLATION")
    print("="*60)
    
    for sent in test_sentences:
        # Tokenize
        tokens = tokenize(sent, 'vi')
        
        # Encode
        src_ids = [src_vocab.word2idx[src_vocab.SOS_TOKEN]] + \
                  src_vocab.encode(tokens, max_len=30) + \
                  [src_vocab.word2idx[src_vocab.EOS_TOKEN]]
        
        # Padding
        if len(src_ids) < 32:
            src_ids.extend([src_vocab.word2idx[src_vocab.PAD_TOKEN]] * (32 - len(src_ids)))
        
        # To tensor
        src_tensor = torch.LongTensor([src_ids]).to(device)
        
        # Decode
        with torch.no_grad():
            tgt_ids = greedy_decode(model, src_tensor, src_vocab, tgt_vocab, device)
        
        # Convert to text
        tgt_tokens = tgt_vocab.decode(tgt_ids[0].cpu().tolist())
        translation = ' '.join(tgt_tokens)
        
        print(f"\n🇻🇳 Vietnamese: {sent}")
        print(f"🇬🇧 English: {translation}")
        print("-" * 40)

if __name__ == '__main__':
    demo()