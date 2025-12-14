import torch
from transformer import create_transformer_model
import sys

print("Python version:", sys.version)
print("Torch version:", torch.__version__)

try:
    model = create_transformer_model(100, 100)
    model.eval()
    print("Model created successfully")
    
    src = torch.randint(1, 100, (2, 10))
    tgt = torch.randint(1, 100, (2, 10))
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    
    # Test encode specifically
    print("Testing encode...")
    enc_out = model.encode(src, src_mask)
    print("Encode successful, shape:", enc_out.shape)
    
    # Test full forward
    print("Testing forward...")
    out = model(src, tgt)
    print("Forward successful, shape:", out.shape)
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
