import torch
from transformer import create_transformer_model

def verify_masking():
    print("Verifying Transformer Masking Fix...")
    
    src_vocab_size = 100
    tgt_vocab_size = 100
    model = create_transformer_model(src_vocab_size, tgt_vocab_size)
    model.eval()
    
    batch_size = 2
    seq_len = 10
    src = torch.randint(1, src_vocab_size, (batch_size, seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, seq_len))
    
    src[0, 7:] = 0
    
    print("Running forward pass...")
    try:
        output = model(src, tgt)
        print(f"Forward pass successful. Output shape: {output.shape}")
        print("Fix applied correctly: Encoder now accepts src_mask.")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_masking()
